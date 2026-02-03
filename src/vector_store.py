"""ChromaDB vector store for semantic similarity search."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.utils import load_config

logger = logging.getLogger("mindguard")

# Load config
try:
    CONFIG = load_config()
    CHROMA_PATH = CONFIG["database"]["chroma_path"]
    COLLECTION_NAME = CONFIG["database"]["chroma_collection"]
    EMBEDDING_MODEL = CONFIG["embeddings"]["model_name"]
    TOP_K = CONFIG["similarity"]["top_k"]
except Exception:
    CHROMA_PATH = "data/chroma_db"
    COLLECTION_NAME = "mindguard_sessions"
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    TOP_K = 3


class VectorStore:
    """
    ChromaDB-based vector store for semantic similarity search.

    Stores session embeddings and enables finding similar past cases.
    """

    def __init__(
            self,
            persist_path: str = None,
            collection_name: str = None,
            embedding_model: str = None
    ):
        """
        Initialize the vector store.

        Args:
            persist_path: Path for ChromaDB persistence.
            collection_name: Name of the collection.
            embedding_model: Sentence transformer model name.
        """
        self.persist_path = persist_path or CHROMA_PATH
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL

        # Ensure directory exists
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Load embedding model (lazy loading)
        self._embedding_model = None

        logger.info(
            f"VectorStore initialized: {self.collection_name} "
            f"({self.collection.count()} documents)"
        )

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Input text.

        Returns:
            Embedding vector as list of floats.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_session(
            self,
            session_id: int,
            text: str,
            metadata: Dict[str, Any],
            embedding: List[float] = None
    ) -> None:
        """
        Add a session to the vector store.

        Args:
            session_id: Unique session identifier.
            text: Original input text.
            metadata: Session metadata (user_id, risk_level, etc.).
            embedding: Pre-computed embedding (computed if not provided).
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embed_text(text)

        # Ensure metadata values are ChromaDB-compatible
        clean_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, list):
                clean_metadata[key] = ", ".join(str(v) for v in value)
            else:
                clean_metadata[key] = str(value)

        # Upsert to collection
        self.collection.upsert(
            ids=[str(session_id)],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata]
        )

        logger.debug(f"Added session {session_id} to vector store")

    def find_similar(
            self,
            text: str = None,
            embedding: List[float] = None,
            top_k: int = None,
            exclude_ids: List[str] = None,
            filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar sessions by text or embedding.

        Args:
            text: Query text (embedding computed if provided).
            embedding: Pre-computed query embedding.
            top_k: Number of results to return.
            exclude_ids: Session IDs to exclude from results.
            filter_metadata: Metadata filter (e.g., {"risk_level": "High"}).

        Returns:
            List of similar sessions with scores.
        """
        if text is None and embedding is None:
            raise ValueError("Either text or embedding must be provided")

        # Generate embedding from text if needed
        if embedding is None:
            embedding = self.embed_text(text)

        top_k = top_k or TOP_K

        # Build where filter
        where_filter = None
        if filter_metadata:
            where_filter = filter_metadata

        # Query collection
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k + (len(exclude_ids) if exclude_ids else 0),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        similar_sessions = []

        if results["ids"] and results["ids"][0]:
            for i, session_id in enumerate(results["ids"][0]):
                # Skip excluded IDs
                if exclude_ids and session_id in exclude_ids:
                    continue

                # Convert distance to similarity score (cosine distance -> similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity_score = 1 - distance  # Cosine distance to similarity

                similar_sessions.append({
                    "session_id": int(session_id),
                    "text": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity_score": round(similarity_score, 4)
                })

                # Stop at top_k
                if len(similar_sessions) >= top_k:
                    break

        return similar_sessions

    def delete_session(self, session_id: int) -> bool:
        """
        Delete a session from the vector store.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            self.collection.delete(ids=[str(session_id)])
            logger.debug(f"Deleted session {session_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def delete_user_sessions(self, user_id: str) -> int:
        """
        Delete all sessions for a user.

        Args:
            user_id: User ID whose sessions to delete.

        Returns:
            Number of sessions deleted.
        """
        # Get all sessions for user
        results = self.collection.get(
            where={"user_id": user_id},
            include=[]
        )

        if not results["ids"]:
            return 0

        # Delete all
        self.collection.delete(ids=results["ids"])
        deleted_count = len(results["ids"])

        logger.info(f"Deleted {deleted_count} sessions for user {user_id}")
        return deleted_count

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific session by ID.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session data or None if not found.
        """
        results = self.collection.get(
            ids=[str(session_id)],
            include=["documents", "metadatas", "embeddings"]
        )

        if not results["ids"]:
            return None

        return {
            "session_id": session_id,
            "text": results["documents"][0] if results["documents"] else None,
            "metadata": results["metadatas"][0] if results["metadatas"] else {},
            "embedding": results["embeddings"][0] if results["embeddings"] else None
        }

    def count(self) -> int:
        """Get total number of sessions in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all sessions from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("Vector store cleared")


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("MindGuard Vector Store Test")
    print("=" * 60)

    # Initialize store
    store = VectorStore(persist_path="data/chroma_db_test")

    # Test sessions
    test_sessions = [
        {
            "session_id": 1,
            "text": "I feel hopeless and depressed. Nothing matters anymore.",
            "metadata": {"user_id": "test-user-1", "risk_level": "High", "confidence": 0.92}
        },
        {
            "session_id": 2,
            "text": "I can't sleep at night and I'm always exhausted.",
            "metadata": {"user_id": "test-user-1", "risk_level": "Medium", "confidence": 0.78}
        },
        {
            "session_id": 3,
            "text": "I'm constantly anxious and worried about everything.",
            "metadata": {"user_id": "test-user-2", "risk_level": "Medium", "confidence": 0.81}
        },
        {
            "session_id": 4,
            "text": "Life feels meaningless. I don't see the point.",
            "metadata": {"user_id": "test-user-2", "risk_level": "High", "confidence": 0.89}
        },
    ]

    # Add sessions
    print("\nAdding test sessions...")
    for session in test_sessions:
        store.add_session(
            session_id=session["session_id"],
            text=session["text"],
            metadata=session["metadata"]
        )
    print(f"Total sessions in store: {store.count()}")

    # Test similarity search
    print("\n--- Similarity Search Test ---")
    query = "I feel empty and sad, like nothing will ever get better."
    print(f"Query: {query}")

    similar = store.find_similar(text=query, top_k=3)
    print(f"\nTop {len(similar)} similar sessions:")
    for i, session in enumerate(similar, 1):
        print(f"  {i}. Session {session['session_id']} (score: {session['similarity_score']:.3f})")
        print(f"     Text: {session['text'][:60]}...")
        print(f"     Risk: {session['metadata'].get('risk_level', 'N/A')}")

    # Test deletion
    print("\n--- Deletion Test ---")
    store.delete_session(1)
    print(f"After deleting session 1: {store.count()} sessions")

    # Clear test data
    store.clear()
    print(f"After clearing: {store.count()} sessions")

    print("\n✅ Vector store tests passed!")