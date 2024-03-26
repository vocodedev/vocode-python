from enum import Enum
from typing import Optional
from .model import TypedModel

DEFAULT_EMBEDDINGS_MODEL = {"name": "text-embedding-ada-002", "dimension": 1536}


class VectorDBType(str, Enum):
    BASE = "vector_db_base"
    PINECONE = "vector_db_pinecone"
    PGVector = "vector_db_pgvector"


class VectorDBConfig(TypedModel, type=VectorDBType.BASE.value):
    embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL["name"]
    embedding_dimension = DEFAULT_EMBEDDINGS_MODEL["dimension"]


class PineconeConfig(VectorDBConfig, type=VectorDBType.PINECONE.value):
    index: str
    api_key: Optional[str] = None
    api_environment: Optional[str] = None
    top_k: int = 3


class PGVectorConfig(VectorDBConfig, type=VectorDBType.PGVector.value):
    top_k: int = 3
    password: str = ""
    user: str = "postgres"
    port: int = 5432
    host: str = ""
    database_name: str = "postgres"
