import os
import sys
import json
import time
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests

def _ensure_imports():
    base = Path(__file__).resolve().parents[1]
    if str(base) not in sys.path:
        sys.path.append(str(base))

_ensure_imports()

from datasets import Dataset  # type: ignore
from work.config import get_llm, get_embeddings, get_vector_store
from work.tools import retrieve_kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
FAITH_THRESH = float(os.getenv("FAITH_THRESH", "0.69"))
RELEV_THRESH = float(os.getenv("RELEV_THRESH", "0.59"))
API_URL = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat")
REPORT_PATH = Path(__file__).resolve().parent / "ragas_report.json"
SIM_THRESH = float(os.getenv("SIM_THRESH", "0.7"))
TOP_K = int(os.getenv("TOP_K", "2"))

def _load_ragas():
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        try:
            from ragas.llms import LangchainLLMWrapper as RagasLLM
        except Exception:
            try:
                from ragas.llms import LangchainLLM as RagasLLM
            except Exception:
                try:
                    from ragas.llms import LangChainLLM as RagasLLM
                except Exception:
                    RagasLLM = None
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper as RagasEmb
        except Exception:
            try:
                from ragas.embeddings import LangchainEmbeddings as RagasEmb
            except Exception:
                try:
                    from ragas.embeddings import LangChainEmbeddings as RagasEmb
                except Exception:
                    RagasEmb = None
        return evaluate, RagasLLM, RagasEmb, faithfulness, answer_relevancy
    except Exception as e:
        logging.error("RAGAS 未安装或版本不兼容：%s", e)
        return None, None, None, None, None

def _curl_line(q: str) -> str:
    payload = json.dumps({"query": q}, ensure_ascii=False).replace('"', '\\"')
    return f'curl.exe -s -X POST {API_URL} -H "Content-Type: application/json; charset=utf-8" -d "{payload}"'

def _call_api(query: str, timeout: int = 30) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    data = {"query": query}
    r = requests.post(API_URL, headers=headers, data=json.dumps(data, ensure_ascii=False).encode("utf-8"), timeout=timeout)
    r.raise_for_status()
    return r.json()

def _get_contexts(query: str) -> List[str]:
    serialized, docs = retrieve_kb(query)
    ctxs = []
    try:
        for d in docs or []:
            c = getattr(d, "page_content", "")
            if isinstance(c, str) and c.strip():
                ctxs.append(c)
    except Exception:
        pass
    return ctxs

def _mask_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    try:
        keys = {"api_key", "token", "authorization", "secret", "password"}
        out = {}
        for k, v in (meta or {}).items():
            out[k] = "***" if k.lower() in keys else v
        return out
    except Exception:
        return {}

def _cosine(a: List[float], b: List[float]) -> float:
    try:
        import math
        if not a or not b:
            return 0.0
        s_ab = sum((x * y) for x, y in zip(a, b))
        s_a = math.sqrt(sum(x * x for x in a))
        s_b = math.sqrt(sum(y * y for y in b))
        if s_a == 0.0 or s_b == 0.0:
            return 0.0
        return float(s_ab / (s_a * s_b))
    except Exception:
        return 0.0

def _doc_id_from(doc: Any) -> str:
    try:
        meta = getattr(doc, "metadata", {}) or {}
        cid = meta.get("id") or meta.get("source") or meta.get("chunk_id")
        if cid:
            return str(cid)
        import hashlib
        text = getattr(doc, "page_content", "") or ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"

def _faiss_query_details(vs: Any, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if vs is None:
        return items
    try:
        res = []
        try:
            res = vs.similarity_search_with_score(query, k=k)
        except Exception:
            docs = vs.similarity_search(query, k=k)
            res = [(d, None) for d in docs]
        for doc, score in res:
            md = _mask_metadata(getattr(doc, "metadata", {}) or {})
            txt = getattr(doc, "page_content", "") or ""
            items.append({
                "doc_id": _doc_id_from(doc),
                "score": float(score) if isinstance(score, (int, float)) else None,
                "metadata": md,
                "text_len": len(txt),
                "snippet": txt[:200],
            })
        logging.info("FAISS 检索: query='%s' 命中=%d", query, len(items))
    except Exception as e:
        logging.error("FAISS 检索失败: %s | query=%s", e, query)
    return items

def _precision_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
    try:
        retrieved = set(retrieved_ids or [])
        relevant = set(relevant_ids or [])
        inter = retrieved & relevant
        precision = (len(inter) / len(retrieved)) if retrieved else 0.0
        recall = (len(inter) / len(relevant)) if relevant else 0.0
        return {"precision_at_k": precision, "recall_at_k": recall}
    except Exception:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

def _build_problem_description(f: float, r: float, precision: float, recall: float) -> str:
    parts = []
    if f < FAITH_THRESH:
        parts.append("忠实度偏低，可能未严格依据检索上下文作答")
    if r < RELEV_THRESH:
        parts.append("回答相关性不足，未充分贴合用户问题")
    if precision < 0.5:
        parts.append("检索精确度偏低，相关片段未充分命中")
    if recall < 0.5:
        parts.append("检索召回率偏低，可能遗漏部分相关信息")
    return "；".join(parts) if parts else "无明显问题"

def _evaluate_one(evaluate, RagasLLM, RagasEmb, faithfulness, answer_relevancy, llm, emb, q: str, ans: str, ctxs: List[str]) -> Dict[str, float]:
    ds = Dataset.from_dict({"question": [q], "answer": [ans], "contexts": [ctxs]})
    ragas_llm = RagasLLM(llm) if RagasLLM else llm
    ragas_emb = RagasEmb(emb) if RagasEmb else emb
    res = evaluate(ds, metrics=[faithfulness, answer_relevancy], llm=ragas_llm, embeddings=ragas_emb)
    if hasattr(res, "to_dict"):
        return res.to_dict()
    if isinstance(res, dict):
        return res
    if hasattr(res, "to_pandas"):
        df = res.to_pandas()
        row = df.iloc[0].to_dict()
        return {k: float(row.get(k, 0.0)) for k in ["faithfulness", "answer_relevancy"] if k in row}
    return {"faithfulness": 0.0, "answer_relevancy": 0.0}

def run():
    t0 = time.time()
    evaluate, RagasLLM, RagasEmb, faithfulness, answer_relevancy = _load_ragas()
    if not all([evaluate, faithfulness, answer_relevancy]):
        logging.error("无法进行评估，请安装兼容版本的 ragas 与 datasets")
        return
    timestamp = datetime.now().isoformat()
    llm = get_llm()
    emb = get_embeddings()
    vs = get_vector_store()
    if vs is None:
        logging.warning("FAISS 向量索引不可用，将跳过向量对比与检索指标")
    tests = [
        {"name": "课程相关", "query": "AI会不会替代程序员？", "ground_truth": "不会完全替代，但会极大提升效率。我们的课程重点是“AI for Developer”，教会程序员驾驭AI。"},
        {"name": "售前咨询", "query": "售前咨询：课程适合新人吗？", "ground_truth": "建议有一定项目经验后再学习。我们会提供前置自学资料包，帮助补足基础。"},
        {"name": "售后服务", "query": "售后服务：课后答疑支持有哪些？", "ground_truth": "提供专属微信群+GitHub Discussion区，讲师每周集中答疑两次，问题归档为知识库供检索。"},
    ]
    results = []
    pass_count = 0
    fail_count = 0
    for case in tests:
        q = case["query"]
        try:
            logging.info("请求: %s", _curl_line(q))
            resp = _call_api(q)
            ans = str(resp.get("answer", "") or "").strip()
            route = resp.get("route")
            ctxs = _get_contexts(q)
            metrics = _evaluate_one(evaluate, RagasLLM, RagasEmb, faithfulness, answer_relevancy, llm, emb, q, ans, ctxs)
            f = float(metrics.get("faithfulness", 0.0))
            r = float(metrics.get("answer_relevancy", 0.0))
            gt = case.get("ground_truth", "")
            faiss_q = _faiss_query_details(vs, q, TOP_K) if vs else []
            faiss_gt = _faiss_query_details(vs, gt, TOP_K) if vs else []
            retrieved_ids = [x["doc_id"] for x in faiss_q]
            relevant_ids = [x["doc_id"] for x in faiss_gt]
            pr = _precision_recall(retrieved_ids, relevant_ids)
            try:
                q_vec = emb.embed_query(q)
                gt_vec = emb.embed_query(gt)
                qgt_sim = _cosine(q_vec, gt_vec)
                doc_to_ground_truth = []
                for it in faiss_q[:min(len(faiss_q), TOP_K)]:
                    dv = emb.embed_query(it["snippet"])
                    s = _cosine(dv, gt_vec)
                    doc_to_ground_truth.append({"doc_id": it["doc_id"], "cosine": s, "above_threshold": s >= SIM_THRESH})
                vec_cmp = {"query_ground_truth_cosine": qgt_sim, "threshold": SIM_THRESH, "doc_to_ground_truth": doc_to_ground_truth}
            except Exception:
                vec_cmp = {"query_ground_truth_cosine": 0.0, "threshold": SIM_THRESH, "doc_to_ground_truth": []}
            problem = _build_problem_description(f, r, pr["precision_at_k"], pr["recall_at_k"])
            passed = (f >= FAITH_THRESH) and (r >= RELEV_THRESH)
            pass_count += 1 if passed else 0
            fail_count += 0 if passed else 1
            logging.info("用例=%s 路由=%s faithfulness=%.3f answer_relevancy=%.3f 结果=%s", case["name"], route, f, r, "通过" if passed else "失败")
            results.append({
                "case": case["name"],
                "query": q,
                "route": route,
                "answer_preview": ans[:120],
                "contexts_count": len(ctxs),
                "faiss_results": {"query": faiss_q, "ground_truth": faiss_gt},
                "vector_comparison": vec_cmp,
                "evaluation_metrics": {
                    "faithfulness": f,
                    "answer_relevancy": r,
                    "precision_at_k": pr["precision_at_k"],
                    "recall_at_k": pr["recall_at_k"],
                },
                "problem_description": problem,
                "passed": passed
            })
        except Exception as e:
            logging.error("用例失败: %s | 错误=%s", case["name"], e)
            results.append({
                "case": case["name"],
                "query": q,
                "route": None,
                "answer_preview": "",
                "contexts_count": 0,
                "metrics": {"faithfulness": 0.0, "answer_relevancy": 0.0},
                "passed": False,
                "error": str(e)
            })
            fail_count += 1
    summary = {
        "thresholds": {"faithfulness": FAITH_THRESH, "answer_relevancy": RELEV_THRESH},
        "pass": pass_count,
        "fail": fail_count,
        "total": len(tests),
        "avg_faithfulness": float(sum(x["evaluation_metrics"]["faithfulness"] for x in results) / len(results)) if results else 0.0,
        "avg_answer_relevancy": float(sum(x["evaluation_metrics"]["answer_relevancy"] for x in results) / len(results)) if results else 0.0,
        "avg_precision_at_k": float(sum(x["evaluation_metrics"].get("precision_at_k", 0.0) for x in results) / len(results)) if results else 0.0,
        "avg_recall_at_k": float(sum(x["evaluation_metrics"].get("recall_at_k", 0.0) for x in results) / len(results)) if results else 0.0,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    report = {"timestamp": timestamp, "summary": summary, "cases": results}
    required = ["problem_description", "faiss_results", "vector_comparison", "evaluation_metrics"]
    has_all = all(all(k in c for k in required) for c in results)
    report["validation"] = {"required_fields_present_all_cases": bool(has_all)}
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logging.info("评估报告生成: %s", str(REPORT_PATH))
    except Exception as e:
        logging.error("报告生成失败: %s", e)
    print("通过/失败统计:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    run()