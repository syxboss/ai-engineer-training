from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple, Any
import logging
import re
import time
import hashlib
import json
from datetime import datetime, timezone
from enum import Enum

app = FastAPI()

# 配置结构化日志
class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        
        # 配置日志格式
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_audit(self, audit_data: Dict[str, Any]):
        """记录审计日志"""
        # 添加时间戳
        audit_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # 格式化为JSON并记录
        json_log = json.dumps(audit_data, ensure_ascii=False, indent=2)
        self.logger.info(f"AUDIT_LOG: {json_log}")
    
    def log_security_alert(self, alert_data: Dict[str, Any]):
        """记录安全告警日志"""
        alert_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        alert_data["log_type"] = "security_alert"
        
        json_log = json.dumps(alert_data, ensure_ascii=False, indent=2)
        self.logger.critical(f"SECURITY_ALERT: {json_log}")
    
    def log_error(self, error_data: Dict[str, Any]):
        """记录错误日志"""
        error_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        error_data["log_type"] = "error"
        
        json_log = json.dumps(error_data, ensure_ascii=False, indent=2)
        self.logger.error(f"ERROR_LOG: {json_log}")

# 初始化结构化日志记录器
structured_logger = StructuredLogger("sql-gateway")

# 保持原有的基础日志记录器用于简单日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql-gateway-basic")

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class InputSanitizer:
    """输入过滤器 - 检测和清理恶意输入"""
    
    def __init__(self):
        # 危险关键词模式
        self.dangerous_patterns = [
            r'(?i)\b(drop|delete|truncate|alter)\s+table\b',
            r'(?i)\bunion\s+select\b',
            r'(?i)\bor\s+1\s*=\s*1\b',
            r'(?i)\band\s+1\s*=\s*1\b',
            r'(?i)--\s*',
            r'(?i)/\*.*?\*/',
            r'(?i)\bexec\s*\(',
            r'(?i)\beval\s*\(',
            r'(?i)\bscript\b',
            r'(?i)<script.*?>.*?</script>',
        ]
        
        # SQL注入模式
        self.injection_patterns = [
            r"'.*?'.*?'",  # 多个单引号
            r'";.*?--',     # 分号加注释
            r'\bor\b.*?\b1\s*=\s*1\b',  # OR 1=1
        ]
    
    def sanitize(self, input_text: str) -> Dict[str, Any]:
        """
        清理输入文本
        返回: {"is_clean": bool, "cleaned_input": str, "detected_threats": List[str]}
        """
        detected_threats = []
        
        # 检测危险模式
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_text):
                detected_threats.append(f"危险SQL关键词: {pattern}")
        
        # 检测注入模式
        for pattern in self.injection_patterns:
            if re.search(pattern, input_text):
                detected_threats.append(f"SQL注入模式: {pattern}")
        
        # 基本清理
        cleaned_input = input_text.strip()
        cleaned_input = re.sub(r'[<>"\']', '', cleaned_input)  # 移除危险字符
        
        return {
            "is_clean": len(detected_threats) == 0,
            "cleaned_input": cleaned_input,
            "detected_threats": detected_threats
        }

class SchemaRestrictor:
    """Schema访问限制器 - 基于用户角色控制表访问"""
    
    def __init__(self):
        # 角色权限配置
        self.role_permissions = {
            "admin": {"allowed_tables": ["*"], "forbidden_operations": []},
            "analyst": {
                "allowed_tables": ["orders", "customers", "products", "sales"],
                "forbidden_operations": ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
            },
            "viewer": {
                "allowed_tables": ["orders", "customers", "products"],
                "forbidden_operations": ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
            },
            "guest": {
                "allowed_tables": ["products"],
                "forbidden_operations": ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
            }
        }
    
    def is_allowed(self, sql: str, user_role: str) -> Tuple[bool, str]:
        """
        检查SQL是否被用户角色允许
        返回: (是否允许, 错误消息)
        """
        if user_role not in self.role_permissions:
            return False, f"未知用户角色: {user_role}"
        
        permissions = self.role_permissions[user_role]
        
        # 检查操作权限
        sql_upper = sql.upper()
        for forbidden_op in permissions["forbidden_operations"]:
            if forbidden_op in sql_upper:
                return False, f"角色 {user_role} 不允许执行 {forbidden_op} 操作"
        
        # 检查表访问权限
        if "*" not in permissions["allowed_tables"]:
            # 提取SQL中的表名（简化版本）
            table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)|UPDATE\s+(\w+)|INSERT\s+INTO\s+(\w+)'
            matches = re.findall(table_pattern, sql_upper)
            
            for match_group in matches:
                for table in match_group:
                    if table and table.lower() not in [t.lower() for t in permissions["allowed_tables"]]:
                        return False, f"角色 {user_role} 不允许访问表 {table}"
        
        return True, "访问权限检查通过"

class SQLTemplater:
    """SQL模板器 - 预定义安全SQL模板"""
    
    def __init__(self):
        self.templates = {
            "list_orders": {
                "pattern": r"(?i)(显示|查看|列出).*(订单|order)",
                "sql": "SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE customer_id = {customer_id} LIMIT {limit}",
                "params": ["customer_id", "limit"]
            },
            "customer_info": {
                "pattern": r"(?i)(显示|查看|查询).*(客户|customer).*(信息|详情)",
                "sql": "SELECT customer_id, customer_name, email, phone FROM customers WHERE customer_id = {customer_id}",
                "params": ["customer_id"]
            },
            "product_search": {
                "pattern": r"(?i)(搜索|查找|查询).*(产品|商品|product)",
                "sql": "SELECT product_id, product_name, price, category FROM products WHERE product_name LIKE '%{keyword}%' LIMIT {limit}",
                "params": ["keyword", "limit"]
            }
        }
    
    def match_template(self, question: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        匹配问题到模板
        返回: (模板ID, 参数字典)
        """
        for template_id, template in self.templates.items():
            if re.search(template["pattern"], question):
                # 简化参数提取（实际应用中需要更复杂的NLP处理）
                params = {}
                if "customer_id" in template["params"]:
                    params["customer_id"] = "1"  # 默认值
                if "limit" in template["params"]:
                    params["limit"] = "10"  # 默认值
                if "keyword" in template["params"]:
                    # 提取关键词
                    words = question.split()
                    params["keyword"] = words[-1] if words else "product"
                
                return template_id, params
        
        return None, {}
    
    def render_sql(self, template_id: str, params: Dict[str, Any]) -> str:
        """渲染SQL模板"""
        if template_id not in self.templates:
            raise ValueError(f"未知模板ID: {template_id}")
        
        template = self.templates[template_id]
        try:
            return template["sql"].format(**params)
        except KeyError as e:
            raise ValueError(f"模板参数缺失: {e}")

class SQLValidator:
    """SQL验证器 - 最终安全检查"""
    
    def __init__(self):
        # 高风险操作模式
        self.high_risk_patterns = [
            r'(?i)\bdrop\s+table\b',
            r'(?i)\bdelete\s+from\b.*\bwhere\s+1\s*=\s*1\b',
            r'(?i)\btruncate\s+table\b',
            r'(?i)\balter\s+table\b',
            r'(?i)\bcreate\s+user\b',
            r'(?i)\bgrant\s+all\b',
        ]
        
        # 中风险操作模式
        self.medium_risk_patterns = [
            r'(?i)\bdelete\s+from\b',
            r'(?i)\bupdate\s+.*\bset\b',
            r'(?i)\binsert\s+into\b',
            r'(?i)\bunion\s+select\b',
            r'(?i)\bselect\s+.*\bfrom\s+.*\bwhere\s+.*\bor\b',
        ]
    
    def validate(self, sql: str) -> Dict[str, Any]:
        """
        验证SQL安全性
        返回: {"risk_level": str, "issues": List[str], "score": int}
        """
        issues = []
        risk_score = 0
        
        # 检查高风险模式
        for pattern in self.high_risk_patterns:
            if re.search(pattern, sql):
                issues.append(f"高风险操作: {pattern}")
                risk_score += 10
        
        # 检查中风险模式
        for pattern in self.medium_risk_patterns:
            if re.search(pattern, sql):
                issues.append(f"中风险操作: {pattern}")
                risk_score += 5
        
        # 检查其他风险因素
        if len(sql) > 1000:
            issues.append("SQL语句过长")
            risk_score += 2
        
        if sql.count(';') > 1:
            issues.append("包含多条SQL语句")
            risk_score += 3
        
        # 确定风险等级
        if risk_score >= 10:
            risk_level = RiskLevel.HIGH.value
        elif risk_score >= 5:
            risk_level = RiskLevel.MEDIUM.value
        else:
            risk_level = RiskLevel.LOW.value
        
        return {
            "risk_level": risk_level,
            "issues": issues,
            "score": risk_score
        }

# 初始化各组件
sanitizer = InputSanitizer()
restrictor = SchemaRestrictor()
templater = SQLTemplater()
validator = SQLValidator()

class SQLRequest(BaseModel):
    question: str
    db_id: str
    user_role: str
    user_id: str

class SQLResponse(BaseModel):
    safe_sql: Optional[str] = None
    status: str
    message: str
    risk_level: str
    audit_id: str

@app.post("/generate-sql", response_model=SQLResponse)
async def generate_sql_endpoint(request: SQLRequest):
    audit_id = f"AUDIT-{int(time.time())}-{request.user_id[:4]}"
    start_time = time.time()
    
    # 初始化审计日志数据
    audit_log = {
        "audit_id": audit_id,
        "user_id": request.user_id,
        "user_role": request.user_role,
        "db_id": request.db_id,
        "input_question": request.question,
        "generated_sql": None,
        "validation_result": {
            "is_safe": False,
            "risk_level": "unknown",
            "issues": []
        },
        "status": "processing",
        "response_time_ms": 0,
        "processing_steps": []
    }
    
    try:
        # 1. 输入过滤
        audit_log["processing_steps"].append("input_sanitization")
        clean_result = sanitizer.sanitize(request.question)
        
        if not clean_result["is_clean"]:
            audit_log["status"] = "blocked"
            audit_log["validation_result"]["risk_level"] = "high"
            audit_log["validation_result"]["issues"] = clean_result["detected_threats"]
            audit_log["response_time_ms"] = int((time.time() - start_time) * 1000)
            
            # 记录结构化审计日志
            structured_logger.log_audit(audit_log)
            
            logger.warning(f"{audit_id} 输入过滤失败: {clean_result['detected_threats']}")
            return SQLResponse(
                status="blocked",
                message=f"输入包含敏感内容: {clean_result['detected_threats']}",
                risk_level="high",
                audit_id=audit_id
            )
        
        # 2. 优先尝试模板化
        audit_log["processing_steps"].append("template_matching")
        tmpl_id, params = templater.match_template(clean_result["cleaned_input"])
        if tmpl_id:
            audit_log["processing_steps"].append("template_rendering")
            sql = templater.render_sql(tmpl_id, params)
            audit_log["template_used"] = tmpl_id
        else:
            # 3. 调用 text2SQL 模型（省略调用细节）
            audit_log["processing_steps"].append("text2sql_generation")
            sql = "SELECT * FROM orders LIMIT 10"  # 模拟
        
        audit_log["generated_sql"] = sql
        
        # 4. Schema 限制检查
        audit_log["processing_steps"].append("schema_restriction_check")
        allowed, msg = restrictor.is_allowed(sql, request.user_role)
        if not allowed:
            audit_log["status"] = "blocked"
            audit_log["validation_result"]["risk_level"] = "high"
            audit_log["validation_result"]["issues"] = [msg]
            audit_log["response_time_ms"] = int((time.time() - start_time) * 1000)
            
            # 记录结构化审计日志
            structured_logger.log_audit(audit_log)
            
            logger.warning(f"{audit_id} Schema 检查失败: {msg}")
            return SQLResponse(
                status="blocked",
                message=msg,
                risk_level="high",
                audit_id=audit_id
            )
        
        # 5. 最终校验
        audit_log["processing_steps"].append("final_validation")
        validation = validator.validate(sql)
        
        audit_log["validation_result"]["risk_level"] = validation["risk_level"]
        audit_log["validation_result"]["issues"] = validation["issues"]
        audit_log["validation_result"]["risk_score"] = validation["score"]
        
        # 6. 风险决策
        if validation["risk_level"] == "high":
            audit_log["status"] = "blocked"
            audit_log["validation_result"]["is_safe"] = False
            audit_log["response_time_ms"] = int((time.time() - start_time) * 1000)
            
            # 记录结构化审计日志
            structured_logger.log_audit(audit_log)
            
            # 记录安全告警
            structured_logger.log_security_alert({
                "audit_id": audit_id,
                "user_id": request.user_id,
                "user_role": request.user_role,
                "threat_type": "high_risk_sql",
                "sql_statement": sql,
                "risk_factors": validation["issues"],
                "action_taken": "blocked"
            })
            
            logger.critical(f"{audit_id} 高风险 SQL 阻断: {sql}")
            send_alert(f"高风险SQL拦截: 用户{request.user_id} 尝试执行\n{sql}")
            
            return SQLResponse(
                status="blocked",
                message="检测到高风险操作，已自动阻断",
                risk_level="high",
                audit_id=audit_id
            )
            
        elif validation["risk_level"] == "medium":
            audit_log["status"] = "pending_review"
            audit_log["validation_result"]["is_safe"] = False
            audit_log["response_time_ms"] = int((time.time() - start_time) * 1000)
            
            # 记录结构化审计日志
            structured_logger.log_audit(audit_log)
            
            # 加入人工审核队列
            enqueue_for_review({
                "audit_id": audit_id,
                "sql": sql,
                "user": request.user_id,
                "reason": validation["issues"]
            })
            logger.info(f"{audit_id} 中风险 SQL 进入人工审核")
            
            return SQLResponse(
                status="pending_review",
                message="查询需人工审核，请稍候",
                risk_level="medium",
                audit_id=audit_id
            )
        else:
            audit_log["status"] = "approved"
            audit_log["validation_result"]["is_safe"] = True
            audit_log["response_time_ms"] = int((time.time() - start_time) * 1000)
            
            # 记录结构化审计日志
            structured_logger.log_audit(audit_log)
            
            logger.info(f"{audit_id} 低风险 SQL 自动放行")
            
            return SQLResponse(
                safe_sql=sql,
                status="approved",
                message="SQL 已生成",
                risk_level="low",
                audit_id=audit_id
            )
                
    except Exception as e:
        audit_log["status"] = "error"
        audit_log["error_message"] = str(e)
        audit_log["response_time_ms"] = int((time.time() - start_time) * 1000)
        
        # 记录结构化错误日志
        structured_logger.log_error({
            "audit_id": audit_id,
            "user_id": request.user_id,
            "error_type": "internal_error",
            "error_message": str(e),
            "processing_steps_completed": audit_log["processing_steps"]
        })
        
        # 记录结构化审计日志
        structured_logger.log_audit(audit_log)
        
        logger.error(f"{audit_id} 网关内部错误: {str(e)}")
        raise HTTPException(500, "服务内部错误")

def send_alert(message: str):
    """发送安全告警"""
    try:
        # 实现告警发送（邮件、钉钉、Slack等）
        logger.critical(f"安全告警: {message}")
        print(f"发送告警: {message}")
        
        # 这里可以集成实际的告警系统
        # 例如: send_email(), send_dingtalk(), send_slack()
        
    except Exception as e:
        logger.error(f"告警发送失败: {str(e)}")

def enqueue_for_review(item: dict):
    """加入人工审核队列"""
    try:
        # 加入 Redis 队列或数据库待审表
        logger.info(f"SQL审核入队: {item['audit_id']}")
        print(f"加入审核队列: {item}")
        
        # 这里可以集成实际的队列系统
        # 例如: redis_client.lpush("review_queue", json.dumps(item))
        # 或者: db.insert("pending_reviews", item)
        
        return True
    except Exception as e:
        logger.error(f"审核队列入队失败: {str(e)}")
        return False

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="服务内部错误，请稍后重试"
    )

@app.middleware("http")
async def security_middleware(request, call_next):
    """安全中间件 - 记录请求和响应"""
    start_time = time.time()
    
    # 记录请求
    logger.info(f"请求开始: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # 记录响应时间
        process_time = time.time() - start_time
        logger.info(f"请求完成: 耗时 {process_time:.3f}s")
        
        return response
    except Exception as e:
        logger.error(f"请求处理异常: {str(e)}")
        raise

@app.post("/approve-sql")
async def approve_sql_endpoint(request: ApprovalRequest):
    start_time = time.time()
    audit_id = f"APPROVAL-{int(time.time())}-{request.user_id[:4]}"
    
    try:
        # 记录审批操作
        approval_log = {
            "audit_id": audit_id,
            "user_id": request.user_id,
            "user_role": request.user_role,
            "action": "manual_approval",
            "original_audit_id": request.audit_id,
            "approval_decision": request.approved,
            "approval_reason": request.reason,
            "status": "completed",
            "response_time_ms": 0
        }
        
        if request.approved:
            logger.info(f"{audit_id} SQL 审批通过: {request.audit_id}")
            approval_log["status"] = "approved"
        else:
            logger.warning(f"{audit_id} SQL 审批拒绝: {request.audit_id}, 原因: {request.reason}")
            approval_log["status"] = "rejected"
        
        approval_log["response_time_ms"] = int((time.time() - start_time) * 1000)
        structured_logger.log_audit(approval_log)
        
        return {"status": "success", "message": "审批完成"}
        
    except Exception as e:
        error_log = {
            "audit_id": audit_id,
            "user_id": request.user_id,
            "error_type": "approval_error",
            "error_message": str(e),
            "response_time_ms": int((time.time() - start_time) * 1000)
        }
        structured_logger.log_error(error_log)
        
        logger.error(f"{audit_id} 审批操作错误: {str(e)}")
        raise HTTPException(500, "审批操作失败")

@app.get("/health")
async def health_check():
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "input_sanitizer": "ok",
            "schema_restrictor": "ok", 
            "sql_templater": "ok",
            "sql_validator": "ok"
        },
        "response_time_ms": int((time.time() - start_time) * 1000)
    }
    
    # 记录健康检查日志
    structured_logger.log_audit({
        "audit_id": f"HEALTH-{int(time.time())}",
        "user_id": "system",
        "user_role": "system",
        "action": "health_check",
        "status": "completed",
        "response_time_ms": health_status["response_time_ms"]
    })
    
    return health_status

@app.get("/")
async def root():
    start_time = time.time()
    
    response_time = int((time.time() - start_time) * 1000)
    
    # 记录根路径访问日志
    structured_logger.log_audit({
        "audit_id": f"ROOT-{int(time.time())}",
        "user_id": "anonymous",
        "user_role": "guest",
        "action": "root_access",
        "status": "completed",
        "response_time_ms": response_time
    })
    
    return {
        "service": "DB Security Gateway",
        "version": "1.0.0",
        "status": "running",
        "response_time_ms": response_time
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 SQL 安全网关服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")