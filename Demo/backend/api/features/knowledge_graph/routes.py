from fastapi import APIRouter, Depends, HTTPException
from api.features.rag_manager import get_rag_instance
from models.database import User
from api.auth.jwt_handler import get_current_active_user
from .schemas import Graph, GraphQuery, Node, Edge
from PathRAG.PathRAG import PathRAG

# Initialize PathRAG
rag = PathRAG(working_dir="./data")

router = APIRouter(
    prefix="/knowledge-graph",
    tags=["Knowledge Graph"],
    dependencies=[Depends(get_current_active_user)]
)

@router.post("/query", response_model=Graph)
async def query_knowledge_graph(query: GraphQuery, current_user: User = Depends(get_current_active_user)):
    try:
        rag = get_rag_instance()
        # Get the knowledge graph from PathRAG
        knowledge_graph = rag.chunk_entity_relation_graph
        
        # Convert NetworkX graph to our schema
        nodes = []
        edges = []
        
        # Get all nodes and edges
        for node_id in knowledge_graph._graph.nodes():
            node_data = knowledge_graph._graph.nodes[node_id]
            nodes.append(
                Node(
                    id=node_id,
                    label=node_id,
                    type=node_data.get("entity_type", "Unknown"),
                    description=node_data.get("description", "")
                )
            )
        
        for source, target, data in knowledge_graph._graph.edges(data=True):
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    label=data.get("description", ""),
                    weight=data.get("weight", 1.0)
                )
            )
        
        return Graph(nodes=nodes, edges=edges)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying knowledge graph: {str(e)}")

@router.get("/", response_model=Graph)
async def get_knowledge_graph(current_user: User = Depends(get_current_active_user)):
    try:
        rag = get_rag_instance()
        # Get the knowledge graph from PathRAG
        knowledge_graph = rag.chunk_entity_relation_graph
        
        # Convert NetworkX graph to our schema
        nodes = []
        edges = []
        
        # Get all nodes and edges
        for node_id in knowledge_graph._graph.nodes():
            node_data = knowledge_graph._graph.nodes[node_id]
            nodes.append(
                Node(
                    id=node_id,
                    label=node_id,
                    type=node_data.get("entity_type", "Unknown"),
                    description=node_data.get("description", "")
                )
            )
        
        for source, target, data in knowledge_graph._graph.edges(data=True):
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    label=data.get("description", ""),
                    weight=data.get("weight", 1.0)
                )
            )
        
        return Graph(nodes=nodes, edges=edges)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting knowledge graph: {str(e)}")
