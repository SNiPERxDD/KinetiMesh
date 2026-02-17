"""
KinetiMesh Internal Wrapper Module.

Provides synchronous Python function access to KinetiMesh tools without
requiring command-line MCP server initialization. This enables direct
internal usage by AI agents and other Python code.

Usage:
    from src.mcp_wrapper import init_kinetimesh, search_code_internal

    # Initialize once
    init_kinetimesh("/path/to/repo")

    # Use tools
    results = search_code_internal("authentication middleware", top_k=5)
    structure = get_file_structure_internal("src/auth/middleware.py")
    related = find_related_internal("authenticate_user", top_k=10)
    stats = get_index_stats_internal()
    diagnostics = doctor_internal()
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from src.server.pipeline import KinetiMeshPipeline

logger = logging.getLogger("kmesh.wrapper")

# Global state for singleton pipeline
_global_pipeline: Optional[KinetiMeshPipeline] = None
_global_lock = threading.RLock()


def init_kinetimesh(repo_path: str, db_path: str = ".kmesh/data") -> Dict[str, Any]:
    """Initialize the KinetiMesh pipeline for internal use.

    This function initializes the global pipeline instance that will be used
    by all internal tool functions. It performs initial indexing and starts
    the file watcher.

    Args:
        repo_path: Path to the repository root (can be relative or absolute).
        db_path: Path for LanceDB storage (default: .kmesh/data).
                 Relative paths are resolved relative to repo_path.

    Returns:
        Dict with initialization metrics:
            - status: "ready" if successful
            - repo_path: Absolute path to repository
            - init_time_ms: Initialization time in milliseconds
            - files_scanned: Number of files scanned
            - chunks_parsed: Number of code chunks parsed
            - total_time_ms: Total indexing time

    Raises:
        RuntimeError: If pipeline initialization fails.
        OSError: If repository path doesn't exist or isn't accessible.

    Example:
        >>> metrics = init_kinetimesh(".")
        >>> print(f"Indexed {metrics['files_scanned']} files")
    """
    global _global_pipeline

    with _global_lock:
        # Resolve absolute path
        repo_path_obj = Path(repo_path).resolve()
        if not repo_path_obj.exists():
            raise OSError(f"Repository path does not exist: {repo_path}")
        if not repo_path_obj.is_dir():
            raise OSError(f"Repository path is not a directory: {repo_path}")

        repo_path_str = str(repo_path_obj)

        # Handle db_path resolution
        db_path_obj = Path(db_path)
        if not db_path_obj.is_absolute():
            db_path_obj = repo_path_obj / db_path

        try:
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Pivot to home if repo is read-only
            db_path_obj = Path.home() / ".kmesh" / "data"
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Repo read-only. Pivoting database to: {db_path_obj}")

        db_path_str = str(db_path_obj)

        # Initialize pipeline
        try:
            _global_pipeline = KinetiMeshPipeline(repo_path_str, db_path_str)

            # Full initial index
            index_metrics = _global_pipeline.full_index(incremental=True)

            # Start file watcher
            _global_pipeline.start_watcher()

            logger.info(f"KinetiMesh initialized for: {repo_path_str}")

            return {
                "status": "ready",
                "repo_path": repo_path_str,
                **index_metrics,
            }
        except Exception as e:
            logger.error(f"Failed to initialize KinetiMesh: {e}", exc_info=True)
            _global_pipeline = None
            raise RuntimeError(f"Pipeline initialization failed: {e}") from e


def _get_pipeline() -> KinetiMeshPipeline:
    """Get the global pipeline instance.

    Returns:
        Active KinetiMeshPipeline instance.

    Raises:
        RuntimeError: If pipeline is not initialized.
    """
    with _global_lock:
        if _global_pipeline is None:
            raise RuntimeError(
                "KinetiMesh not initialized. Call init_kinetimesh() first."
            )
        return _global_pipeline


def search_code_internal(query: str, top_k: int = 5) -> str:
    """Search the codebase using hybrid vector + keyword search.

    Performs semantic search across all indexed source files, combining
    vector similarity (sentence-transformers) with BM25 keyword matching.

    Args:
        query: Natural language query describing the code you're looking for.
               Examples: "authentication middleware", "database connection pool",
               "error handling in API routes"
        top_k: Number of results to return (default: 5, max: 20).

    Returns:
        Formatted string with search results including file paths, symbol names,
        code snippets, and relevance scores.

    Raises:
        RuntimeError: If pipeline is not initialized.

    Example:
        >>> results = search_code_internal("vector database search")
        >>> print(results)
    """
    import time

    pipeline = _get_pipeline()
    top_k = min(max(top_k, 1), 20)

    start = time.perf_counter()
    results = pipeline.search(query, top_k=top_k, use_reranker=False)
    elapsed = time.perf_counter() - start

    if not results:
        return f"No results found for: '{query}'"

    output_lines = [f"Found {len(results)} results for '{query}' ({elapsed*1000:.0f}ms):\n"]

    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0))
        output_lines.append(f"--- Result {i} ---")
        output_lines.append(f"File: {r['file_path']}  (L{r['start_line']}-L{r['end_line']})")
        output_lines.append(f"Symbol: {r['symbol_name']}  [{r['chunk_type']}]")
        output_lines.append(f"Score: {score:.4f}")

        # Truncate very long code snippets
        text = r["text"]
        if len(text) > 1500:
            text = text[:1500] + "\n... (truncated)"
        output_lines.append(f"```{r.get('language', '')}")
        output_lines.append(text)
        output_lines.append("```")
        output_lines.append("")

    return "\n".join(output_lines)


def get_file_structure_internal(path: str) -> str:
    """Get the structural skeleton of a file (symbols only, no implementation).

    Returns function signatures, class definitions, and docstrings without
    the full implementation bodies. Helps understand a file's API surface.

    Args:
        path: Relative path to the file from the repository root.
              Example: "src/auth/middleware.py"

    Returns:
        String containing file skeleton with imports, class definitions,
        function signatures, and docstrings.

    Raises:
        RuntimeError: If pipeline is not initialized.

    Example:
        >>> skeleton = get_file_structure_internal("src/server/pipeline.py")
        >>> print(skeleton)
    """
    pipeline = _get_pipeline()
    return pipeline.get_file_skeleton(path)


def find_related_internal(symbol_name: str, top_k: int = 10) -> str:
    """Find where a symbol is defined and used across the codebase.

    Searches for function, class, or method definitions and their usages.
    Useful for understanding call graphs and dependencies.

    Args:
        symbol_name: Name of the function, class, or method to search for.
                     Example: "authenticate", "UserModel", "handle_request"
        top_k: Maximum number of results (default: 10, max: 20).

    Returns:
        Formatted string with locations where the symbol is defined or
        referenced, grouped by definition vs usage.

    Raises:
        RuntimeError: If pipeline is not initialized.

    Example:
        >>> related = find_related_internal("KinetiMeshPipeline")
        >>> print(related)
    """
    pipeline = _get_pipeline()
    top_k = min(max(top_k, 1), 20)

    results = pipeline.search_symbol(symbol_name, top_k=top_k)

    if not results:
        return f"No references found for symbol: '{symbol_name}'"

    # Separate definitions from usages
    definitions = [r for r in results if r["chunk_type"] in ("signature", "class_definition")]
    usages = [r for r in results if r["chunk_type"] not in ("signature", "class_definition")]

    output_lines = [f"References for '{symbol_name}':\n"]

    if definitions:
        output_lines.append("DEFINITIONS:")
        for r in definitions:
            output_lines.append(
                f"  {r['file_path']}:L{r['start_line']} "
                f"[{r['chunk_type']}] {r['symbol_name']}"
            )
            # Show signature text
            text = r["text"]
            if len(text) > 500:
                text = text[:500] + "..."
            output_lines.append(f"    {text}")
            output_lines.append("")

    if usages:
        output_lines.append("USAGES / RELATED:")
        for r in usages:
            output_lines.append(
                f"  {r['file_path']}:L{r['start_line']} "
                f"[{r['chunk_type']}] {r['symbol_name']}"
            )

    return "\n".join(output_lines)


def get_index_stats_internal() -> str:
    """Get statistics about the current index state.

    Returns information about indexed files, chunks, performance metrics,
    watcher status, and failed files.

    Returns:
        Formatted string with index statistics.

    Raises:
        RuntimeError: If pipeline is not initialized.

    Example:
        >>> stats = get_index_stats_internal()
        >>> print(stats)
    """
    pipeline = _get_pipeline()
    stats = pipeline.get_stats()

    lines = ["KinetiMesh Index Stats:"]

    # Core metrics
    lines.append(f"  repo_path: {stats.get('repo_path', 'N/A')}")
    lines.append(f"  total_indexed_files: {stats.get('total_indexed_files', 0)}")
    lines.append(f"  total_indexed_chunks: {stats.get('total_indexed_chunks', 0)}")
    lines.append(f"  total_stored_chunks: {stats.get('total_stored_chunks', 0)}")
    lines.append(f"  tracked_files: {stats.get('tracked_files', 0)}")
    lines.append(f"  watcher_active: {stats.get('watcher_active', False)}")

    # Performance metrics
    lines.append(f"  last_full_index_ms: {stats.get('last_full_index_ms', 0):.2f}")
    lines.append(f"  last_incremental_ms: {stats.get('last_incremental_ms', 0):.2f}")

    # Failed files
    failed_count = stats.get('failed_files_count', 0)
    lines.append(f"  failed_files_count: {failed_count}")

    if failed_count > 0:
        lines.append("\nRecent Failed Files:")
        recent_failures = stats.get('recent_failures', [])
        for fail in recent_failures[:10]:  # Show max 10
            lines.append(f"  - {fail.get('file', 'unknown')}: {fail.get('error', 'unknown error')[:100]}")
        if failed_count > 10:
            lines.append(f"  ... and {failed_count - 10} more")

    return "\n".join(lines)


def doctor_internal() -> str:
    """Run comprehensive system diagnostics.

    Checks:
    - Database integrity
    - Write permissions
    - Memory usage estimates
    - Failed/poison files
    - Index health

    Returns:
        Formatted diagnostic report with health status and recommendations.

    Raises:
        RuntimeError: If pipeline is not initialized.

    Example:
        >>> diagnostics = doctor_internal()
        >>> print(diagnostics)
    """
    import os
    from pathlib import Path

    pipeline = _get_pipeline()

    lines = ["="*60]
    lines.append("KinetiMesh System Diagnostics")
    lines.append("="*60)

    # 1. Database Health
    lines.append("\n[1] Database Health:")
    try:
        stats = pipeline.store.get_stats()
        db_path = stats.get('db_path', 'unknown')
        lines.append(f"  ✓ Database path: {db_path}")
        lines.append(f"  ✓ Total chunks: {stats.get('total_chunks', 0)}")
        lines.append("  ✓ Database accessible")
    except Exception as e:
        lines.append(f"  ✗ Database error: {str(e)}")

    # 2. Write Permissions
    lines.append("\n[2] Write Permissions:")
    try:
        kmesh_dir = Path(".kmesh")
        if kmesh_dir.exists():
            test_file = kmesh_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            lines.append("  ✓ .kmesh directory writable")
        else:
            lines.append("  ⚠ .kmesh directory does not exist")
    except Exception as e:
        lines.append(f"  ✗ Write permission error: {str(e)}")

    # 3. Memory Usage Estimate
    lines.append("\n[3] Resource Usage:")
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        lines.append(f"  Memory usage: {mem_mb:.1f} MB")
    except ImportError:
        lines.append("  ⚠ psutil not available for memory checks")
    except Exception as e:
        lines.append(f"  ⚠ Could not check memory: {str(e)}")

    # 4. Failed Files
    lines.append("\n[4] Failed Files:")
    pipeline_stats = pipeline.get_stats()
    failed_count = pipeline_stats.get('failed_files_count', 0)
    lines.append(f"  Total failed files: {failed_count}")

    if failed_count > 0:
        lines.append("\n  Recent failures:")
        recent = pipeline_stats.get('recent_failures', [])
        for fail in recent[:5]:  # Show max 5 in doctor
            stage = fail.get('stage', 'unknown')
            file = fail.get('file', 'unknown')
            error = fail.get('error', 'unknown')[:80]
            lines.append(f"    [{stage}] {file}: {error}")

        if failed_count > 5:
            lines.append(f"    ... and {failed_count - 5} more (see get_index_stats for full list)")
    else:
        lines.append("  ✓ No failed files")

    # 5. Index Health
    lines.append("\n[5] Index Health:")
    watcher_active = pipeline_stats.get('watcher_active', False)
    lines.append(f"  File watcher: {'✓ Active' if watcher_active else '✗ Inactive'}")

    tracked = pipeline_stats.get('tracked_files', 0)
    indexed = pipeline_stats.get('total_indexed_files', 0)
    lines.append(f"  Tracked files: {tracked}")
    lines.append(f"  Indexed files: {indexed}")

    # 6. Recommendations
    lines.append("\n[6] Recommendations:")
    if failed_count > 10:
        lines.append("  ⚠ High number of failed files detected")
        lines.append("    Consider checking file encodings or size limits")
    if not watcher_active:
        lines.append("  ⚠ File watcher is inactive")
        lines.append("    This is normal for internal wrapper usage")
    if failed_count == 0 and indexed > 0:
        lines.append("  ✓ All systems healthy")

    lines.append("\n" + "="*60)
    return "\n".join(lines)


def shutdown_kinetimesh() -> None:
    """Shutdown the KinetiMesh pipeline and cleanup resources.

    Stops the file watcher and releases the global pipeline instance.
    Should be called when the pipeline is no longer needed.

    Example:
        >>> shutdown_kinetimesh()
    """
    global _global_pipeline

    with _global_lock:
        if _global_pipeline is not None:
            try:
                _global_pipeline.stop_watcher()
                logger.info("KinetiMesh pipeline shutdown complete")
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")
            finally:
                _global_pipeline = None
