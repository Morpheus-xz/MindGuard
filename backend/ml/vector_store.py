"""ChromaDB vector store for semantic similarity search."""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


class VectorStore:
    """ChromaDB-based vector store for finding similar sessions."""
    
    def __init__(self, persist_dir: Optional[str] = None):
        if persist_dir is None:
            persist_dir = str(Path(__file__).parent.parent.parent / "data" / "chromadb")
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize ChromaDB and embedding model."""
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
            
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="mindguard_sessions",
                metadata={"description": "Mental health session embeddings"}
            )
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self._initialized = True
            print(f"✓ ChromaDB initialized ({self.collection.count()} documents)")
            return True
            
        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("Run: pip install chromadb sentence-transformers")
            return False
        except Exception as e:
            print(f"ChromaDB error: {e}")
            return False
    
    def add_session(self, session_id: str, text: str, user_id: str, 
                    risk_level: str, confidence: float, clinical_flags: List[str] = None) -> bool:
        """Add a session to the vector store."""
        if not self._initialized:
            return False
        
        try:
            embedding = self.embedding_model.encode(text).tolist()
            metadata = {
                "user_id": user_id,
                "risk_level": risk_level,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "flags": ",".join(clinical_flags) if clinical_flags else ""
            }
            
            self.collection.add(ids=[session_id], embeddings=[embedding], metadatas=[metadata], documents=[text])
            return True
        except Exception as e:
            print(f"Error adding session: {e}")
            return False
    
    def find_similar(self, text: str, n_results: int = 3, exclude_user_id: Optional[str] = None) -> List[Dict]:
        """Find similar past sessions."""
        if not self._initialized or self.collection.count() == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(text).tolist()
            fetch_n = n_results * 2 if exclude_user_id else n_results
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(fetch_n, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            similar_sessions = []
            if results and results['ids'] and results['ids'][0]:
                for i, session_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    if exclude_user_id and metadata.get('user_id') == exclude_user_id:
                        continue
                    
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = max(0, 1 - distance)
                    
                    doc = results['documents'][0][i]
                    similar_sessions.append({
                        "session_id": session_id,
                        "text_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                        "risk_level": metadata.get('risk_level', 'Unknown'),
                        "similarity_score": round(similarity, 3),
                        "flags": metadata.get('flags', '').split(',') if metadata.get('flags') else []
                    })
                    
                    if len(similar_sessions) >= n_results:
                        break
            
            return similar_sessions
        except Exception as e:
            print(f"Error finding similar: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        if not self._initialized:
            return {"initialized": False}
        return {"initialized": True, "total_sessions": self.collection.count(), "persist_dir": self.persist_dir}
    
    def delete_user_data(self, user_id: str) -> int:
        """Delete all sessions for a user (GDPR)."""
        if not self._initialized:
            return 0
        try:
            all_data = self.collection.get(include=["metadatas"], where={"user_id": user_id})
            if all_data and all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                return len(all_data['ids'])
            return 0
        except:
            return 0
