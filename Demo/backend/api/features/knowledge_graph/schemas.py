from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Node(BaseModel):
    id: str
    label: str
    type: str
    description: Optional[str] = None

class Edge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None
    weight: Optional[float] = None

class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class GraphQuery(BaseModel):
    query: str
