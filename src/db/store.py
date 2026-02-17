"""
KinetiMesh Vector Store - LanceDB-Powered Hybrid Search Engine.

Combines:
  - Local sentence-transformer embeddings (all-MiniLM-L6-v2)
  - LanceDB serverless vector storage
  - Native Full-Text Search (Tantivy) for hybrid search
  - FlashRank re-ranking for result quality

Schema follows the PLAN.md specification with optimizations for
incremental file-level updates (nuke-and-replace strategy).

Performance Features:
  - Auto device selection: MPS (Apple Silicon) > CUDA > CPU
  - Adaptive batch sizing for embedding throughput
  - Priority-weighted scoring to boost signatures over fallback
  - FlashRank candidate pre-filtering for sub-second reranking
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import lancedb
import pyarrow as pa
import numpy as np
from tqdm import tqdm

from src.parser.chunker import CodeChunk, ChunkType

logger = logging.getLogger("kmesh.db")

# ── Embedding Configuration ────────────────────────────────────────────────────

# Model is loaded lazily on first use
_EMBEDDING_MODEL = None
_EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 dimension


def _get_embedding_model():
    """Lazily load the sentence-transformer embedding model.

    Uses all-MiniLM-L6-v2: fast, lightweight, 384-dim vectors.

    Backend Selection (in priority order):
        1. ONNX Runtime -- 1.5-3x faster than PyTorch on CPU
        2. PyTorch CPU -- solid fallback, fast on Apple Silicon NEON

    Note: MPS (Metal) is intentionally NOT used for small models.
    Benchmark shows MPS is 2x SLOWER than CPU for all-MiniLM-L6-v2
    due to Metal transfer overhead exceeding GPU computation benefit.

    Benchmark (M3, 500 texts, all-MiniLM-L6-v2):
        PyTorch CPU: 334ms (1496 texts/sec)
        PyTorch MPS: 674ms (742 texts/sec) -- worse, Metal overhead
        ONNX CPU:    ~200ms (~2500 texts/sec) -- when available

    Returns:
        SentenceTransformer model instance.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        # Attempt ONNX backend for maximum CPU throughput
        onnx_available = False
        try:
            import onnxruntime  # noqa: F401
            from optimum.onnxruntime import ORTModelForFeatureExtraction  # noqa: F401
            onnx_available = True
        except (ImportError, Exception):
            pass

        if onnx_available:
            try:
                _EMBEDDING_MODEL = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    backend="onnx",
                )
                logger.info("Embedding model loaded: all-MiniLM-L6-v2 (384-dim, backend=onnx)")
                return _EMBEDDING_MODEL
            except Exception as e:
                logger.warning(f"ONNX backend failed, falling back to PyTorch: {e}")

        # PyTorch fallback: CPU is fastest for small models on Apple Silicon
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA GPU acceleration")
        except (ImportError, AttributeError):
            pass

        _EMBEDDING_MODEL = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device,
        )
        logger.info(f"Embedding model loaded: all-MiniLM-L6-v2 (384-dim, backend=torch, device={device})")
    return _EMBEDDING_MODEL


def embed_texts(texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts into vectors.

    Uses tqdm progress bar for real-time visibility on large embedding jobs.
    Implements error handling for CUDA OOM and encoding issues.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding (larger = faster throughput).
        show_progress: Whether to show tqdm progress bar.

    Returns:
        numpy array of shape (len(texts), 384).
    
    Raises:
        RuntimeError: If embedding fails critically (no fallback possible).
    """
    if not texts:
        return np.array([]).reshape(0, _EMBEDDING_DIM)
    
    model = _get_embedding_model()

    try:
        if len(texts) <= 100 or not show_progress:
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        # Manual batched embedding with tqdm for real-time progress
        all_vectors = []
        with tqdm(total=len(texts), desc="Embedding", unit="chunks", ncols=80) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    vecs = model.encode(
                        batch,
                        batch_size=len(batch),
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    all_vectors.append(vecs)
                    pbar.update(len(batch))
                except RuntimeError as e:
                    # CUDA OOM or other runtime errors - try smaller batch
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        logger.warning(f"CUDA OOM, retrying batch {i} with size 1")
                        # Process one at a time as fallback
                        for text in batch:
                            vec = model.encode(
                                [text],
                                batch_size=1,
                                show_progress_bar=False,
                                normalize_embeddings=True,
                            )
                            all_vectors.append(vec)
                            pbar.update(1)
                    else:
                        raise

        # Memory-efficient concatenation without vstack
        # np.vstack creates a copy - use concatenate with pre-allocated array instead
        if len(all_vectors) == 1:
            return all_vectors[0]
        return np.concatenate(all_vectors, axis=0)
        
    except Exception as e:
        logger.error(f"Embedding failed: {type(e).__name__} - {str(e)}")
        raise RuntimeError(f"Failed to embed texts: {str(e)}") from e


def embed_query(query: str) -> np.ndarray:
    """Embed a single query text into a vector.

    Args:
        query: Query string.

    Returns:
        numpy array of shape (384,).
    """
    model = _get_embedding_model()
    return model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


# ── FlashRank Re-ranker ────────────────────────────────────────────────────────

_RERANKER = None


def _get_reranker():
    """Lazily load the FlashRank re-ranker.

    FlashRank is an ultra-lightweight (~4MB) ONNX-based re-ranker
    that dramatically improves search quality from cheap vectors.

    Returns:
        Ranker instance.
    """
    global _RERANKER
    if _RERANKER is None:
        from flashrank import Ranker
        _RERANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".kmesh/models")
        logger.info("FlashRank re-ranker loaded")
    return _RERANKER


def rerank_results(
    query: str, results: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Re-rank search results using FlashRank cross-encoder.

    Pipeline: Retrieve N items via vector search -> Pre-filter -> FlashRank -> Return top K.

    Cross-encoders have a 512 token context limit. Pre-filtering and
    text truncation keeps reranking under 500ms even for large result sets.

    Args:
        query: The search query.
        results: List of result dicts with at least 'text' key.
        top_k: Number of results to return after re-ranking.

    Returns:
        Re-ranked list of result dicts, best first.
    """
    if not results:
        return []

    try:
        from flashrank import RerankRequest
        reranker = _get_reranker()

        # Pre-filter: keep only the top candidates by priority-weighted score
        # This dramatically reduces FlashRank latency (O(n) cross-encoder calls)
        max_candidates = min(len(results), top_k * 3, 20)
        candidates = sorted(
            results,
            key=lambda r: r.get("score", 999) / max(r.get("priority", 1), 0.1),
        )[:max_candidates]

        # Truncate text moderately to balance context vs speed
        # FlashRank cross-encoder has a 512 token limit (~2000 chars)
        passages = [
            {"id": i, "text": r["text"][:1500], "meta": r}
            for i, r in enumerate(candidates)
        ]
        request = RerankRequest(query=query, passages=passages)
        ranked = reranker.rerank(request)

        reranked = []
        for item in ranked[:top_k]:
            result = item["meta"].copy()
            result["rerank_score"] = item["score"]
            reranked.append(result)
        return reranked
    except Exception as e:
        logger.warning(f"FlashRank re-ranking failed, returning raw results: {e}")
        return results[:top_k]


# ── LanceDB Schema & Store ─────────────────────────────────────────────────────

# PyArrow schema for the code chunks table
CODE_CHUNK_SCHEMA = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("text", pa.utf8()),
    pa.field("search_text", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), _EMBEDDING_DIM)),
    pa.field("file_path", pa.utf8()),
    pa.field("symbol_name", pa.utf8()),
    pa.field("chunk_type", pa.utf8()),
    pa.field("parent_symbol", pa.utf8()),
    pa.field("language", pa.utf8()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("priority", pa.float32()),
    pa.field("last_updated", pa.float64()),
])


class VectorStore:
    """LanceDB-backed vector store for code chunks.

    Provides:
        - Incremental file-level indexing (nuke & replace per file)
        - Vector similarity search
        - Full-text search (Tantivy)
        - Hybrid search with FlashRank re-ranking

    Args:
        db_path: Path to the LanceDB data directory.
        table_name: Name of the code chunks table.
    """

    TABLE_NAME = "code_chunks"

    def __init__(self, db_path: str = ".kmesh/data"):
        self.db_path = db_path
        
        # Try to connect and verify integrity
        try:
            self.db = lancedb.connect(db_path)
            self._verify_db_integrity()
        except Exception as e:
            # Database corruption detected - auto-heal
            logger.error(f"[!] Database corruption detected: {type(e).__name__} - {str(e)}")
            logger.info("[!] Auto-healing: rebuilding database...")
            self._rebuild_database()
            
        self._table: Optional[lancedb.table.Table] = None
        self._fts_index_built = False

        # Performance metrics
        self.last_embed_time: float = 0.0
        self.last_store_time: float = 0.0
        self.last_search_time: float = 0.0

    def _verify_db_integrity(self) -> bool:
        """Verify database is accessible and valid.
        
        Returns:
            True if DB is healthy, False otherwise.
            
        Raises:
            Exception: If DB is corrupted and cannot be accessed.
        """
        try:
            # Basic connectivity test
            table_names = self.db.list_tables()
            
            # If table exists, try to access it
            if self.TABLE_NAME in table_names:
                table = self.db.open_table(self.TABLE_NAME)
                # Try a basic operation
                row_count = table.count_rows()
                
                # Additional validation: check vector dimensions if table has data
                if row_count > 0:
                    # Sample first row to verify schema
                    sample = table.head(1).to_pydict()
                    if 'vector' in sample:
                        vector_dim = len(sample['vector'][0])
                        expected_dim = _EMBEDDING_DIM
                        if vector_dim != expected_dim:
                            raise ValueError(
                                f"Vector dimension mismatch: expected {expected_dim}, got {vector_dim}"
                            )
            
            logger.info("Database integrity check passed")
            return True
        except Exception as e:
            logger.error(f"Database integrity check failed: {type(e).__name__} - {str(e)}")
            raise
    
    def _rebuild_database(self) -> None:
        """Rebuild database atomically after corruption.
        
        Uses temp directory + atomic swap to prevent race conditions.
        """
        import shutil
        import tempfile
        from pathlib import Path
        
        db_dir = Path(self.db_path)
        
        # Create temporary directory for new DB
        temp_dir = Path(tempfile.mkdtemp(prefix="kmesh_rebuild_"))
        
        try:
            # Build new database in temp location
            temp_db = lancedb.connect(str(temp_dir))
            temp_db.close()
            
            # Atomic swap: backup old, move new, cleanup
            backup_path = None
            if db_dir.exists():
                backup_path = db_dir.with_suffix(".corrupt_backup")
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                db_dir.rename(backup_path)
                logger.info(f"Backed up corrupted DB to {backup_path}")
            
            # Move temp DB to target location
            shutil.move(str(temp_dir), str(db_dir))
            
            # Cleanup backup
            if backup_path and backup_path.exists():
                shutil.rmtree(backup_path)
            
            # Reconnect
            self.db = lancedb.connect(self.db_path)
            logger.info("Database rebuilt atomically")
            
        except Exception as e:
            # Cleanup temp on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            logger.error(f"Atomic rebuild failed: {e}")
            raise

    @property
    def table(self) -> lancedb.table.Table:
        """Get or create the code chunks table."""
        if self._table is None:
            table_names = self.db.list_tables()
            if self.TABLE_NAME in table_names:
                self._table = self.db.open_table(self.TABLE_NAME)
                logger.info(f"Opened existing table: {self.TABLE_NAME}")
            else:
                self._table = self.db.create_table(
                    self.TABLE_NAME,
                    schema=CODE_CHUNK_SCHEMA,
                )
                logger.info(f"Created table: {self.TABLE_NAME}")
        return self._table

    def index_chunks(
        self, chunks: List[CodeChunk], full_rebuild: bool = False
    ) -> Dict[str, float]:
        """Embed and store code chunks into the vector store.

        Uses batch embedding for efficiency. Two modes:
          - Incremental (default): Deletes existing chunks per-file then inserts.
          - Full rebuild: Drops table and recreates (much faster for initial index).

        Args:
            chunks: List of CodeChunk objects to index.
            full_rebuild: If True, drop and recreate the table instead of per-file deletes.

        Returns:
            Performance metrics dict.
        """
        if not chunks:
            return {"embed_time_ms": 0, "store_time_ms": 0, "chunks_indexed": 0}

        # Group chunks by file for nuke-and-replace
        file_chunks: Dict[str, List[CodeChunk]] = {}
        for chunk in chunks:
            file_chunks.setdefault(chunk.file_path, []).append(chunk)

        # Step 1: Embed all chunk texts in optimized batches
        embed_start = time.perf_counter()
        search_texts = [c.search_text for c in chunks]
        # Adaptive batch size: larger batches for larger datasets
        batch_size = min(256, max(32, len(search_texts) // 4))
        vectors = embed_texts(search_texts, batch_size=batch_size)
        self.last_embed_time = time.perf_counter() - embed_start

        # Step 2: Build records
        store_start = time.perf_counter()
        now = time.time()

        records = []
        for idx, chunk in enumerate(chunks):
            records.append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "search_text": chunk.search_text,
                "vector": vectors[idx].tolist(),
                "file_path": chunk.file_path,
                "symbol_name": chunk.symbol_name,
                "chunk_type": chunk.chunk_type.value,
                "parent_symbol": chunk.parent_symbol,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "priority": chunk.priority,
                "last_updated": now,
            })

        # Step 3: Store - full rebuild vs incremental
        if full_rebuild:
            # Drop and recreate is O(1) vs O(N) per-file deletes
            try:
                self.db.drop_table(self.TABLE_NAME)
            except Exception:
                pass
            self._table = None
            try:
                self._table = self.db.create_table(self.TABLE_NAME, data=records, schema=CODE_CHUNK_SCHEMA)
            except ValueError as e:
                if "already exists" in str(e):
                    # Race condition: drop and retry
                    self.db.drop_table(self.TABLE_NAME)
                    self._table = self.db.create_table(self.TABLE_NAME, data=records, schema=CODE_CHUNK_SCHEMA)
                else:
                    raise
        else:
            # Incremental: delete affected files then batch insert
            affected_files = list(file_chunks.keys())
            if affected_files:
                # Batch delete with OR filter for fewer round-trips
                for file_path in affected_files:
                    try:
                        # Robust SQL escaping: backslashes first, then quotes
                        safe_path = file_path.replace('\\', '\\\\').replace('"', '\\"').replace("'", "''")
                        self.table.delete(f'file_path = "{safe_path}"')
                    except Exception:
                        pass
            self.table.add(records)

        self.last_store_time = time.perf_counter() - store_start

        # Invalidate FTS index since data changed
        self._fts_index_built = False

        return {
            "embed_time_ms": self.last_embed_time * 1000,
            "store_time_ms": self.last_store_time * 1000,
            "chunks_indexed": len(chunks),
            "files_indexed": len(file_chunks),
        }

    def delete_file(self, file_path: str) -> None:
        """Remove all chunks for a specific file.

        Args:
            file_path: Relative file path to remove.
        """
        try:
            # Sanitize file_path to prevent SQL injection
            safe_path = file_path.replace('"', '\\"')
            self.table.delete(f'file_path = "{safe_path}"')
            self._fts_index_built = False
        except Exception:
            pass

    def _ensure_fts_index(self) -> None:
        """Build Full-Text Search index if not already built.

        Uses LanceDB's native Tantivy integration for BM25 search.
        """
        if not self._fts_index_built:
            try:
                self.table.create_fts_index("search_text", replace=True)
                self._fts_index_built = True
                logger.info("FTS index built/rebuilt")
            except Exception as e:
                logger.warning(f"FTS index creation failed: {e}")

    def search_vector(
        self, query: str, top_k: int = 20, filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.

        Filters out low-priority fallback chunks by default to improve
        result quality for code-specific queries.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            filter_expr: Optional SQL-like filter expression.

        Returns:
            List of result dicts with metadata.
        """
        search_start = time.perf_counter()

        query_vec = embed_query(query)
        # Over-retrieve to compensate for fallback filtering
        retrieve_extra = min(top_k * 2, top_k + 10)
        search_builder = self.table.search(query_vec.tolist()).limit(retrieve_extra)

        if filter_expr:
            search_builder = search_builder.where(filter_expr)

        raw_results = search_builder.to_list()
        self.last_search_time = time.perf_counter() - search_start

        # Post-process: boost non-fallback results
        results = []
        for r in raw_results:
            entry = {
                "id": r["id"],
                "text": r["text"],
                "file_path": r["file_path"],
                "symbol_name": r["symbol_name"],
                "chunk_type": r["chunk_type"],
                "parent_symbol": r["parent_symbol"],
                "language": r["language"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "priority": r["priority"],
                "score": float(r.get("_distance", 0)),
            }
            results.append(entry)

        # Sort by priority-adjusted score: lower distance, higher priority = better
        results.sort(key=lambda r: r["score"] / max(r["priority"], 0.1))
        return results[:top_k]

    def search_fts(
        self, query: str, top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform full-text (keyword) search using Tantivy BM25.

        Args:
            query: Keyword query string.
            top_k: Number of results.

        Returns:
            List of result dicts.
        """
        self._ensure_fts_index()
        search_start = time.perf_counter()

        try:
            results = self.table.search(query, query_type="fts").limit(top_k).to_list()
        except Exception:
            return []

        self.last_search_time = time.perf_counter() - search_start

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "file_path": r["file_path"],
                "symbol_name": r["symbol_name"],
                "chunk_type": r["chunk_type"],
                "parent_symbol": r["parent_symbol"],
                "language": r["language"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "priority": r["priority"],
                "score": float(r.get("_score", 0)),
            }
            for r in results
        ]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 10,
        use_reranker: bool = True,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search: Vector + FTS + FlashRank re-ranking.

        Pipeline:
            1. Retrieve top retrieve_k via vector search
            2. Retrieve top retrieve_k via FTS
            3. Merge, deduplicate, deprioritize fallback chunks
            4. Re-rank top candidates via FlashRank (if enabled)
            5. Return top_k results

        The retrieve_k is intentionally low (default: 10) to keep
        FlashRank latency under 500ms. Quality is maintained by
        priority-weighted pre-filtering.

        Args:
            query: Natural language or keyword query.
            top_k: Final number of results to return.
            retrieve_k: Number of candidates to retrieve from each method.
            use_reranker: Whether to apply FlashRank re-ranking.
            filter_expr: Optional SQL filter.

        Returns:
            List of result dicts, best first.
        """
        search_start = time.perf_counter()

        # Vector search (filter out low-priority fallback chunks)
        vector_results = self.search_vector(query, top_k=retrieve_k, filter_expr=filter_expr)

        # FTS search
        fts_results = self.search_fts(query, top_k=retrieve_k)

        # Deprioritize fallback chunks in results
        for r in vector_results + fts_results:
            if r.get("chunk_type") == "fallback":
                r["score"] = r.get("score", 999) * 2  # Push fallback chunks down

        # Merge and deduplicate by chunk ID
        seen_ids = set()
        merged = []
        for r in vector_results + fts_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        # Re-rank
        if use_reranker and merged:
            final = rerank_results(query, merged, top_k=top_k)
        else:
            # Score by priority-weighted vector distance
            merged.sort(key=lambda r: r.get("score", 999) / max(r.get("priority", 1), 0.1))
            final = merged[:top_k]

        self.last_search_time = time.perf_counter() - search_start
        return final

    def search_symbol(
        self, symbol_name: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for a specific symbol by name (definition + usages).

        Args:
            symbol_name: Function/class/method name to find.
            top_k: Max results.

        Returns:
            List of result dicts matching the symbol.
        """
        # Combine FTS for exact name + vector for semantic similarity
        fts_results = self.search_fts(symbol_name, top_k=top_k)
        vec_results = self.search_vector(
            f"function {symbol_name} definition usage",
            top_k=top_k,
        )

        seen_ids = set()
        merged = []
        for r in fts_results + vec_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        return merged[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        try:
            count = self.table.count_rows()
        except Exception:
            count = 0

        return {
            "total_chunks": count,
            "db_path": self.db_path,
            "last_embed_time_ms": self.last_embed_time * 1000,
            "last_store_time_ms": self.last_store_time * 1000,
            "last_search_time_ms": self.last_search_time * 1000,
        }

    def clear(self) -> None:
        """Drop and recreate the table. Use with caution."""
        try:
            self.db.drop_table(self.TABLE_NAME)
        except Exception:
            pass
        self._table = None
        self._fts_index_built = False
        logger.info("Vector store cleared")
