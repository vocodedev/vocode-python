import logging
from typing import Iterable, List, Optional, Tuple
import uuid
import vecs
import asyncio
from langchain.docstore.document import Document
from vocode import getenv
from vocode.streaming.models.vector_db import PGVectorConfig
from vocode.streaming.vector_db.base_vector_db import VectorDB

logger = logging.getLogger(__name__)


class PGVector(VectorDB):
    def __init__(self, config: PGVectorConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.password = getenv("PG_VECTOR_PASSWORD") or self.config.password
        self.host = getenv("PG_VECTOR_HOST") or self.config.host
        self.database_name = (
            getenv("PG_VECTOR_DATABASE_NAME") or self.config.database_name
        )
        self.user = getenv("PG_VECTOR_USER") or self.config.user
        self.port = getenv("PG_VECTOR_PORT") or self.config.port
        self.pg_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database_name}"
        self.dimension = config.embedding_dimension
        self.vecs = vecs.create_client(self.pg_url)
        self._text_key = "text"

    async def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespace: Optional namespace to add the texts to, by default it will be docs

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if namespace is None:
            namespace = "docs"
        # Embed and create the documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            embedding = await self.create_openai_embedding(text)
            metadata = metadatas[i] if metadatas else {}
            metadata[self._text_key] = text
            docs.append((ids[i], embedding, metadata))

        self.docs = self.vecs.get_or_create_collection(
            name=namespace, dimension=self.dimension
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.docs.upsert, docs)
        return ids

    async def similarity_search_with_score(
        self,
        query: str,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return PGVector documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        if namespace is None:
            namespace = "docs"
        query_obj = await self.create_openai_embedding(query)
        self.docs = self.vecs.get_or_create_collection(
            name=namespace, dimension=self.dimension
        )
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.docs.query(
                data=query_obj,
                filters=filter,
                limit=5,
                include_value=True,
                include_metadata=True,
            ),
        )
        docs = []
        for id, score, metadata in results:
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                docs.append((Document(page_content=text, metadata=metadata), score))
            else:
                logger.warning(
                    f"Found document with no `{self._text_key} key. Skipping"
                )
        return docs
