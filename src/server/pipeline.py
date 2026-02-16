"""
KinetiMesh Pipeline Orchestrator.

Coordinates the full indexing pipeline:
    Ingestor -> Parser -> Embedder -> VectorStore

Provides both initial full-index and incremental file-level updates
triggered by watchdog file system events. Includes real-time tqdm
progress bars for visibility during large indexing operations.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from src.ingestor.scanner import MeshIngestor, CodeFile, EXTENSION_TO_LANGUAGE
from src.parser.chunker import parse_file, CodeChunk
from src.db.store import VectorStore

logger = logging.getLogger("kmesh.pipeline")


class KinetiMeshPipeline:
    """Orchestrates the full KinetiMesh indexing pipeline.

    Manages the ingestor, parser, and vector store lifecycle.
    Supports both batch indexing and incremental single-file updates.

    Args:
        repo_path: Path to the repository to index.
        db_path: Path for LanceDB storage (default: .kmesh/data).
    """

    def __init__(self, repo_path: str, db_path: str = ".kmesh/data"):
        self.repo_path = str(Path(repo_path).resolve())
        self.ingestor = MeshIngestor(self.repo_path)
        self.store = VectorStore(db_path)
        self._observer: Optional[Observer] = None
        self._lock = threading.Lock()

        # Performance tracking
        self.last_full_index_time: float = 0.0
        self.last_incremental_time: float = 0.0
        self.total_indexed_files: int = 0
        self.total_indexed_chunks: int = 0

    def full_index(self, incremental: bool = True) -> Dict[str, Any]:
        """Perform a full repository index.

        Scans all files, parses them into chunks, embeds, and stores.
        When incremental=True, only processes files that changed since
        the last scan (hash-based detection).

        Args:
            incremental: Skip unchanged files if True.

        Returns:
            Dict with performance metrics.
        """
        start = time.perf_counter()
        metrics: Dict[str, Any] = {}

        with self._lock:
            # Step 1: Scan
            scan_start = time.perf_counter()
            files = list(self.ingestor.scan(incremental=incremental))
            metrics["scan_time_ms"] = (time.perf_counter() - scan_start) * 1000
            metrics["files_scanned"] = len(files)

            if not files:
                metrics["total_time_ms"] = (time.perf_counter() - start) * 1000
                metrics["status"] = "no_changes"
                return metrics

            # Step 2: Parse all files into chunks
            parse_start = time.perf_counter()
            all_chunks: List[CodeChunk] = []
            if len(files) > 20:
                for f in tqdm(files, desc="Parsing", unit="files", ncols=80):
                    chunks = parse_file(f.content, f.language, f.rel_path)
                    all_chunks.extend(chunks)
            else:
                for f in files:
                    chunks = parse_file(f.content, f.language, f.rel_path)
                    all_chunks.extend(chunks)
            metrics["parse_time_ms"] = (time.perf_counter() - parse_start) * 1000
            metrics["chunks_parsed"] = len(all_chunks)

            # Step 3: Handle deleted files
            deleted = self.ingestor.get_deleted_paths()
            for d in deleted:
                self.store.delete_file(d)
            metrics["files_deleted"] = len(deleted)

            # Step 4: Embed and store
            # Use full_rebuild mode when indexing many files (initial index)
            # to avoid expensive per-file deletes
            is_full_rebuild = not incremental or len(files) > 50
            if all_chunks:
                store_metrics = self.store.index_chunks(
                    all_chunks, full_rebuild=is_full_rebuild
                )
                metrics.update(store_metrics)

            self.total_indexed_files += len(files)
            self.total_indexed_chunks += len(all_chunks)

        elapsed = time.perf_counter() - start
        self.last_full_index_time = elapsed
        metrics["total_time_ms"] = elapsed * 1000
        metrics["status"] = "indexed"

        logger.info(
            f"Full index: {len(files)} files, {len(all_chunks)} chunks "
            f"in {elapsed*1000:.1f}ms"
        )
        return metrics

    def index_single_file(self, file_path: str) -> Dict[str, Any]:
        """Index a single file (triggered by file watcher).

        This is the hot path for live updates. Must be fast.

        Args:
            file_path: Absolute path to the modified file.

        Returns:
            Dict with performance metrics.
        """
        start = time.perf_counter()
        metrics: Dict[str, Any] = {"file": file_path}

        with self._lock:
            code_file = self.ingestor.scan_single_file(file_path)
            if code_file is None:
                metrics["status"] = "unchanged"
                metrics["total_time_ms"] = (time.perf_counter() - start) * 1000
                return metrics

            # Parse
            chunks = parse_file(code_file.content, code_file.language, code_file.rel_path)
            metrics["chunks"] = len(chunks)

            # Store (nuke-and-replace handled internally)
            if chunks:
                store_metrics = self.store.index_chunks(chunks)
                metrics.update(store_metrics)

            self.total_indexed_files += 1
            self.total_indexed_chunks += len(chunks)

        elapsed = time.perf_counter() - start
        self.last_incremental_time = elapsed
        metrics["total_time_ms"] = elapsed * 1000
        metrics["status"] = "indexed"

        logger.info(
            f"File indexed: {code_file.rel_path} -> {len(chunks)} chunks "
            f"in {elapsed*1000:.1f}ms"
        )
        return metrics

    def handle_file_delete(self, file_path: str) -> None:
        """Handle a file deletion event.

        Args:
            file_path: Absolute path to the deleted file.
        """
        try:
            rel_path = str(Path(file_path).relative_to(self.repo_path))
            with self._lock:
                self.store.delete_file(rel_path)
                if rel_path in self.ingestor.file_state:
                    del self.ingestor.file_state[rel_path]
            logger.info(f"File deleted from index: {rel_path}")
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to handle delete for {file_path}: {e}")

    def search(
        self, query: str, top_k: int = 5, use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search across the indexed codebase.

        Args:
            query: Natural language or keyword query.
            top_k: Number of results.
            use_reranker: Use FlashRank re-ranking.

        Returns:
            List of result dicts.
        """
        return self.store.search_hybrid(
            query, top_k=top_k, use_reranker=use_reranker
        )

    def search_symbol(self, symbol_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for a specific symbol.

        Args:
            symbol_name: Name of function/class/method.
            top_k: Max results.

        Returns:
            List of result dicts.
        """
        return self.store.search_symbol(symbol_name, top_k=top_k)

    def get_file_skeleton(self, file_path: str) -> str:
        """Generate a file skeleton (signatures only).

        Args:
            file_path: Relative path to the file.

        Returns:
            String skeleton of the file.
        """
        from src.parser.chunker import get_file_skeleton

        abs_path = Path(self.repo_path) / file_path
        if not abs_path.is_file():
            return f"File not found: {file_path}"

        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
            suffix = abs_path.suffix.lower()
            language = EXTENSION_TO_LANGUAGE.get(suffix, suffix.lstrip(".") or "text")
            return get_file_skeleton(content, language, file_path)
        except OSError as e:
            return f"Error reading {file_path}: {e}"

    def start_watcher(self) -> None:
        """Start the file system watcher for live incremental indexing.

        Uses watchdog to monitor the repository for file changes.
        Runs in a background thread.
        """
        if self._observer is not None:
            return

        handler = _FileChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, self.repo_path, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info(f"File watcher started for: {self.repo_path}")

    def stop_watcher(self) -> None:
        """Stop the file system watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("File watcher stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        store_stats = self.store.get_stats()
        ingestor_stats = self.ingestor.get_stats()
        return {
            "repo_path": self.repo_path,
            "total_indexed_files": self.total_indexed_files,
            "total_indexed_chunks": self.total_indexed_chunks,
            "tracked_files": ingestor_stats["tracked_files"],
            "total_stored_chunks": store_stats["total_chunks"],
            "last_full_index_ms": self.last_full_index_time * 1000,
            "last_incremental_ms": self.last_incremental_time * 1000,
            "watcher_active": self._observer is not None,
        }


class _FileChangeHandler(FileSystemEventHandler):
    """Watchdog event handler for file system changes.

    Debounces rapid events and triggers single-file re-indexing.
    """

    # Directories to ignore in event handling
    _IGNORE_DIRS = {".git", "__pycache__", ".kmesh", "node_modules", ".venv", "venv"}

    def __init__(self, pipeline: KinetiMeshPipeline):
        super().__init__()
        self.pipeline = pipeline
        self._debounce: Dict[str, float] = {}
        self._debounce_interval = 0.5  # seconds

    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        parts = Path(path).parts
        return any(p in self._IGNORE_DIRS for p in parts)

    def _is_debounced(self, path: str) -> bool:
        """Check if this event was triggered too recently."""
        now = time.time()
        last = self._debounce.get(path, 0)
        if now - last < self._debounce_interval:
            return True
        self._debounce[path] = now
        return False

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if self._is_debounced(event.src_path):
            return

        try:
            self.pipeline.index_single_file(event.src_path)
        except Exception as e:
            logger.error(f"Error indexing modified file {event.src_path}: {e}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if self._is_debounced(event.src_path):
            return

        try:
            self.pipeline.index_single_file(event.src_path)
        except Exception as e:
            logger.error(f"Error indexing created file {event.src_path}: {e}")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory or self._should_ignore(event.src_path):
            return

        try:
            self.pipeline.handle_file_delete(event.src_path)
        except Exception as e:
            logger.error(f"Error handling deleted file {event.src_path}: {e}")
