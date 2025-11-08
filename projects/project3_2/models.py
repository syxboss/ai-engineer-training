from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class Entity:
    name: str
    type: str
    confidence: float = 1.0

@dataclass  
class Relationship:
    source: str
    target: str
    type: str
    confidence: float = 1.0

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    document: Document
    score: float
    source: str  # 'vector' or 'keyword' or 'graph'

@dataclass
class GraphResult:
    entities: List[str]
    relationships: List[Dict]
    confidence: float
    reasoning_path: List[str]