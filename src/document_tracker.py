# src/document_tracker.py
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

class DocumentTracker:
    """Track which documents have been ingested to avoid duplicates."""
    
    def __init__(self, db_path: str = "data/tracking.db"):
        self.db_path = Path(db_path).resolve()  # Ensure absolute path
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the tracking database with schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    ingested_at TIMESTAMP NOT NULL,
                    num_chunks INTEGER,
                    status TEXT DEFAULT 'completed'
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash 
                ON documents(file_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ingested_at 
                ON documents(ingested_at)
            """)
            
            conn.commit()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def is_document_ingested(self, file_path: Path) -> bool:
        """
        Check if document has already been ingested.
        Returns True if file exists and hash matches.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Get current file hash
            current_hash = self._calculate_file_hash(file_path)
            current_size = file_path.stat().st_size
            
            # Check database
            cursor.execute("""
                SELECT file_hash, file_size FROM documents 
                WHERE file_path = ?
            """, (str(file_path),))
            
            result = cursor.fetchone()
            
            if result is None:
                return False
            
            stored_hash, stored_size = result
            
            # Document exists and hasn't changed
            return (stored_hash == current_hash and stored_size == current_size)
    
    def mark_document_ingested(
        self, 
        file_path: Path, 
        num_chunks: int
    ):
        """Mark a document as successfully ingested."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (file_path, file_hash, file_size, ingested_at, num_chunks, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(file_path),
                file_hash,
                file_size,
                datetime.now().isoformat(),
                num_chunks,
                'completed'
            ))
            
            conn.commit()
    
    def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested documents."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    SUM(num_chunks) as total_chunks,
                    MIN(ingested_at) as first_ingestion,
                    MAX(ingested_at) as last_ingestion
                FROM documents
            """)
            
            result = cursor.fetchone()
            
            return {
                'total_documents': result[0] or 0,
                'total_chunks': result[1] or 0,
                'first_ingestion': result[2],
                'last_ingestion': result[3]
            }
    
    def list_ingested_documents(self) -> List[Dict]:
        """List all ingested documents with metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, ingested_at, num_chunks, status
                FROM documents
                ORDER BY ingested_at DESC
            """)
            
            results = cursor.fetchall()
            
            return [
                {
                    'file_path': row[0],
                    'ingested_at': row[1],
                    'num_chunks': row[2],
                    'status': row[3]
                }
                for row in results
            ]
    
    def remove_document(self, file_path: Path):
        """Remove a document from tracking (e.g., if file was deleted)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM documents WHERE file_path = ?", (str(file_path),))
            
            conn.commit()
    
    def verify_database_integrity(self) -> bool:
        """Verify database integrity and return True if OK."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                return result[0] == 'ok'
        except Exception:
            return False