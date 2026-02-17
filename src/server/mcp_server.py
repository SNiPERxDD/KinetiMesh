"""
KinetiMesh MCP Server - The Agentic Interface.

Exposes three tools to LLM coding agents via the Model Context Protocol:
    1. search_code(query) - Hybrid vector + FTS search with re-ranking
    2. get_file_structure(path) - File skeleton (symbols only, no implementation)
    3. find_related(symbol_name) - Find where a symbol is defined and used

Also starts the file watcher for live incremental indexing.

Usage:
    kmesh [--repo /path/to/repo]

Protocol:
    Communicates via stdio (stdin/stdout) as per MCP specification.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.server.pipeline import KinetiMeshPipeline

# Global logger instance (configured in _setup_logging)
logger = logging.getLogger("kmesh.server")

# ── Global Pipeline Instance ───────────────────────────────────────────────────

_pipeline: Optional[KinetiMeshPipeline] = None


def _setup_logging(repo_path: str):
    """Setup logging to write to the repository's .kmesh directory, with fallback to home."""
    try:
        # Try primary location
        kmesh_dir = Path(repo_path) / ".kmesh"
        if repo_path == "/" or not os.access(repo_path, os.W_OK):
            # Pivot to home if repo is root or read-only
            kmesh_dir = Path.home() / ".kmesh"
        
        kmesh_dir.mkdir(parents=True, exist_ok=True)
        log_file = kmesh_dir / "server.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a"),
                logging.StreamHandler(sys.stderr),
            ],
            force=True  # Ensure we reconfigure if already called
        )
        logger.info(f"Logging initialized at: {log_file}")
    except Exception as e:
        # Ultimate fallback to stderr if home is also read-only
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True
        )
        logger.error(f"Failed to initialize file logging (using stderr): {e}")



def _get_pipeline() -> KinetiMeshPipeline:
    """Get or initialize the global pipeline instance.

    Returns:
        Active KinetiMeshPipeline.

    Raises:
        RuntimeError: If pipeline is not initialized.
    """
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")
    return _pipeline


def init_pipeline(repo_path: str, db_path: str = ".kmesh/data") -> dict:
    """Initialize the KinetiMesh pipeline for a repository.

    Performs initial full index and starts the file watcher.

    Args:
        repo_path: Path to the repository root.
        db_path: Path for LanceDB storage.

    Returns:
        Dict with initialization metrics.
    """
    global _pipeline

    # Ensure storage directory exists (with fallback to home if read-only)
    db_path_obj = Path(db_path)
    if not db_path_obj.is_absolute():
        db_path_obj = Path(repo_path) / db_path
        
    try:
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Pivot DB path to home if repo is read-only
        db_path_obj = Path.home() / ".kmesh" / "data"
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Repo read-only. Pivoting database to: {db_path_obj}")
    
    # Update db_path to the one we actually found/created
    db_path = str(db_path_obj)
    
    start = time.perf_counter()

    _pipeline = KinetiMeshPipeline(repo_path, db_path)

    # Full initial index
    index_metrics = _pipeline.full_index(incremental=True)

    # Start file watcher
    _pipeline.start_watcher()

    elapsed = time.perf_counter() - start
    logger.info(f"Pipeline initialized in {elapsed*1000:.1f}ms")

    return {
        "status": "ready",
        "repo_path": repo_path,
        "init_time_ms": elapsed * 1000,
        **index_metrics,
    }


# ── MCP Server Setup ──────────────────────────────────────────────────────────

mcp = FastMCP("KinetiMesh")


@mcp.tool()
def search_code(query: str, top_k: int = 5) -> str:
    """Search the codebase using hybrid vector + keyword search.

    Performs semantic search across all indexed source files, combining
    vector similarity (sentence-transformers) with BM25 keyword matching.
    Priority-weighted scoring automatically boosts function signatures
    and class definitions over raw code blocks.

    Search latency: ~50-100ms (instant - no waiting needed).
    FlashRank re-ranking is disabled by default for speed.
    Benchmarked: 100% Top-3 accuracy without reranking vs 83% with.

    Args:
        query: Natural language query describing the code you're looking for.
               Examples: "authentication middleware", "database connection pool",
               "error handling in API routes"
        top_k: Number of results to return (default: 5, max: 20).

    Returns:
        Formatted search results with file paths, symbol names, code snippets,
        and relevance scores. Parse the output to extract file paths and line numbers.
    """
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


@mcp.tool()
def get_file_structure(path: str) -> str:
    """Get the structural skeleton of a file (symbols only, no implementation).

    Returns function signatures, class definitions, and docstrings without
    the full implementation bodies. Helps understand a file's API surface
    without burning tokens on reading the entire source.

    Args:
        path: Relative path to the file from the repository root.
              Example: "src/auth/middleware.py"

    Returns:
        File skeleton showing imports, class definitions, function signatures,
        and docstrings.
    """
    pipeline = _get_pipeline()
    return pipeline.get_file_skeleton(path)


@mcp.tool()
def find_related(symbol_name: str, top_k: int = 10) -> str:
    """Find where a symbol is defined and used across the codebase.

    Searches for function, class, or method definitions and their usages.
    Useful for understanding call graphs and dependencies.

    Args:
        symbol_name: Name of the function, class, or method to search for.
                     Example: "authenticate", "UserModel", "handle_request"
        top_k: Maximum number of results (default: 10).

    Returns:
        List of locations where the symbol is defined or referenced,
        grouped by definition vs usage.
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


@mcp.tool()
def get_index_stats() -> str:
    """Get statistics about the current index state.

    Returns information about indexed files, chunks, performance metrics,
    watcher status, and failed files.

    Returns:
        Formatted statistics about the KinetiMesh index.
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


@mcp.tool()
def doctor() -> str:
    """Run comprehensive system diagnostics.
    
    Checks:
    - Database integrity
    - Write permissions
    - Memory usage estimates
    - Failed/poison files
    - Index health
    
    Returns:
        Diagnostic report with health status and recommendations.
    """
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
    import os
    from pathlib import Path
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
        import os
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
        lines.append("    Run start_watcher() to enable live updates")
    if failed_count == 0 and watcher_active:
        lines.append("  ✓ All systems healthy")
    
    lines.append("\n" + "="*60)
    return "\n".join(lines)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for the KinetiMesh MCP server.

    Usage:
        kmesh start [--repo /path/to/repo]
        python -m src.server.mcp_server [--repo /path/to/repo]
    """
    parser = argparse.ArgumentParser(
        description="KinetiMesh MCP Server - Instant-sync semantic code search"
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the repository to index (default: current directory)",
    )
    parser.add_argument(
        "--db-path",
        default=".kmesh/data",
        help="Path for LanceDB storage (default: .kmesh/data)",
    )

    args = parser.parse_args()

    # Resolve repo path
    repo_path = str(Path(args.repo).resolve())
    
    # Safeguard: Do not index root or system dirs
    if repo_path in ("/", "/usr", "/bin", "/sbin", "/Applications"):
        # Fallback to current working directory if it's not root
        cwd = os.getcwd()
        if cwd != "/":
            repo_path = cwd
        else:
            print(f"Refusing to index root directory: {repo_path}", file=sys.stderr)
            sys.exit(1)
            
    # Initialize logging first
    _setup_logging(repo_path)
    
    logger.info(f"Starting KinetiMesh MCP Server for: {repo_path}")

    # Initialize pipeline
    try:
        init_metrics = init_pipeline(repo_path, args.db_path)
        logger.info(f"Init metrics: {json.dumps(init_metrics, default=str)}")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Run MCP server (stdio transport)
    logger.info("MCP server starting on stdio transport...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
    