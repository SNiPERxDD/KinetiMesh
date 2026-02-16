"""
Comprehensive integration tests for the KinetiMesh pipeline.

Tests each module independently and then the full pipeline end-to-end,
with performance benchmarks at each step.
"""

import sys
import os
import time
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestor.scanner import MeshIngestor, CodeFile
from src.parser.chunker import parse_file, get_file_skeleton, ChunkType
from src.db.store import VectorStore, embed_texts, embed_query
from src.server.pipeline import KinetiMeshPipeline


PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, condition, detail=""):
    """Register a test result."""
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status} {name}" + (f" ({detail})" if detail else ""))
    return condition


def run_all():
    """Execute all integration tests."""
    print("=" * 70)
    print("KinetiMesh Integration Test Suite")
    print("=" * 70)

    # Clean test environment
    test_db = ".kmesh_test"
    shutil.rmtree(test_db, ignore_errors=True)

    # ── Test 1: Ingestor ────────────────────────────────────────
    print("\n[MODULE] Ingestor Tests")

    ingestor = MeshIngestor(".")
    t0 = time.perf_counter()
    files = list(ingestor.scan(incremental=False))
    scan_time = (time.perf_counter() - t0) * 1000

    check("Ingestor finds files", len(files) > 0, f"{len(files)} files")
    check("Scan time < 100ms", scan_time < 100, f"{scan_time:.1f}ms")

    # Verify CodeFile fields
    if files:
        f = files[0]
        check("CodeFile has rel_path", bool(f.rel_path))
        check("CodeFile has content", len(f.content) > 0)
        check("CodeFile has hash", len(f.file_hash) == 64)
        check("CodeFile has language", bool(f.language))
        check("CodeFile has line_count", f.line_count > 0)

    # Incremental scan (no changes)
    t1 = time.perf_counter()
    files2 = list(ingestor.scan(incremental=True))
    inc_time = (time.perf_counter() - t1) * 1000

    check("Incremental detects 0 changes", len(files2) == 0)
    check("Incremental time < 20ms", inc_time < 20, f"{inc_time:.1f}ms")

    # Single file scan
    result = ingestor.scan_single_file("repo_dumper.py")
    check("Single file scan (cached)", result is None, "correctly returns None for unchanged")

    # Stats
    stats = ingestor.get_stats()
    check("Stats returns tracked_files", stats["tracked_files"] > 0)

    # ── Test 2: Parser ──────────────────────────────────────────
    print("\n[MODULE] Parser Tests")

    # Python parsing
    py_code = '''
"""Module docstring."""
import os
from pathlib import Path

class MyClass:
    """A test class."""
    
    def method_one(self, arg1: str, arg2: int = 0) -> bool:
        """Do something."""
        return True
    
    def method_two(self):
        pass

def standalone_func(x, y):
    """Standalone function."""
    return x + y
'''

    t2 = time.perf_counter()
    py_chunks = parse_file(py_code, "python", "test.py")
    parse_time = (time.perf_counter() - t2) * 1000

    check("Python parser produces chunks", len(py_chunks) > 0, f"{len(py_chunks)} chunks")
    check("Parse time < 50ms", parse_time < 50, f"{parse_time:.1f}ms")

    # Check chunk types
    types = {c.chunk_type for c in py_chunks}
    check("Has MODULE_DOC chunks", ChunkType.MODULE_DOC in types)
    check("Has SIGNATURE chunks", ChunkType.SIGNATURE in types)
    check("Has IMPLEMENTATION chunks", ChunkType.IMPLEMENTATION in types)
    check("Has CLASS_DEF chunks", ChunkType.CLASS_DEF in types)
    check("Has IMPORT chunks", ChunkType.IMPORT in types)

    # Verify signature extraction
    sigs = [c for c in py_chunks if c.chunk_type == ChunkType.SIGNATURE]
    sig_names = {c.symbol_name for c in sigs}
    check("Finds method_one", "MyClass.method_one" in sig_names)
    check("Finds method_two", "MyClass.method_two" in sig_names)
    check("Finds standalone_func", "standalone_func" in sig_names)

    # Skeleton
    skel = get_file_skeleton(py_code, "python", "test.py")
    check("Skeleton contains class", "MyClass" in skel)
    check("Skeleton contains method", "method_one" in skel)

    # Fallback for unsupported language
    yaml_content = "key: value\nlist:\n  - item1\n  - item2"
    yaml_chunks = parse_file(yaml_content, "yaml", "config.yaml")
    check("YAML falls back to line chunks", len(yaml_chunks) > 0)
    check("Fallback type is FALLBACK", yaml_chunks[0].chunk_type == ChunkType.FALLBACK)

    # JavaScript parsing
    js_code = '''
function greet(name) {
    return "Hello " + name;
}

class Animal {
    constructor(name) {
        this.name = name;
    }
    
    speak() {
        return this.name + " makes a noise.";
    }
}

const arrowFn = (x) => x * 2;
'''
    js_chunks = parse_file(js_code, "javascript", "test.js")
    check("JS parser produces chunks", len(js_chunks) > 0, f"{len(js_chunks)} chunks")

    # ── Test 3: Embedding ───────────────────────────────────────
    print("\n[MODULE] Embedding Tests")

    texts = ["function to authenticate users", "database connection pool", "error handler"]
    t3 = time.perf_counter()
    vecs = embed_texts(texts)
    embed_time = (time.perf_counter() - t3) * 1000

    check("Embedding produces correct shape", vecs.shape == (3, 384), f"shape={vecs.shape}")
    check("Embeddings are normalized", abs(float((vecs[0] ** 2).sum()) - 1.0) < 0.01)

    # Warm embedding: re-run to measure post-warmup speed
    t3b = time.perf_counter()
    vecs2 = embed_texts(texts)
    warm_time = (time.perf_counter() - t3b) * 1000
    check("Warm embed time < 200ms", warm_time < 200, f"{warm_time:.1f}ms")

    # Vector quality: cosine similarity sanity check
    import numpy as np
    sim_01 = float(np.dot(vecs[0], vecs[1]))  # auth vs db - somewhat related
    sim_02 = float(np.dot(vecs[0], vecs[2]))  # auth vs error - somewhat related
    sim_same = float(np.dot(vecs[0], vecs2[0]))  # same text - should be ~1.0
    check("Same-text cosine ~1.0", sim_same > 0.99, f"sim={sim_same:.4f}")
    check("Different-text cosine < 1.0", sim_01 < 0.95, f"auth-vs-db={sim_01:.4f}")

    # Query embedding
    qvec = embed_query("test query")
    check("Query embedding shape", qvec.shape == (384,))

    # ── Test 4: Vector Store ────────────────────────────────────
    print("\n[MODULE] Vector Store Tests")

    store = VectorStore(f"{test_db}/data")
    check("Store created", store is not None)

    # Index the Python chunks
    t4 = time.perf_counter()
    metrics = store.index_chunks(py_chunks)
    store_time = (time.perf_counter() - t4) * 1000

    check("Chunks stored", metrics["chunks_indexed"] > 0, f"{metrics['chunks_indexed']}")
    check("Store time reasonable", store_time < 30000, f"{store_time:.1f}ms")

    # Vector search
    results_v = store.search_vector("method to do something", top_k=3)
    check("Vector search returns results", len(results_v) > 0)
    if results_v:
        check("Vector result has file_path", "file_path" in results_v[0])
        check("Vector result has symbol_name", "symbol_name" in results_v[0])

    # FTS search
    results_fts = store.search_fts("method_one", top_k=3)
    check("FTS search returns results", len(results_fts) > 0)

    # Hybrid search
    results_hybrid = store.search_hybrid("class definition", top_k=3, use_reranker=False)
    check("Hybrid search returns results", len(results_hybrid) > 0)

    # Delete
    store.delete_file("test.py")
    results_after_delete = store.search_vector("method_one", top_k=3)
    check("Delete removes file chunks", len(results_after_delete) == 0)

    # Stats
    store_stats = store.get_stats()
    check("Stats has total_chunks", "total_chunks" in store_stats)

    # ── Test 5: Pipeline ────────────────────────────────────────
    print("\n[MODULE] Pipeline Integration Tests")

    pipeline = KinetiMeshPipeline(".", f"{test_db}/pipeline_data")
    t5 = time.perf_counter()
    idx_metrics = pipeline.full_index(incremental=False)
    pipeline_time = (time.perf_counter() - t5) * 1000

    check("Pipeline indexes files", idx_metrics["files_scanned"] > 0)
    check("Pipeline produces chunks", idx_metrics["chunks_parsed"] > 0)
    check("Pipeline status is indexed", idx_metrics["status"] == "indexed")

    # Search through pipeline
    search_results = pipeline.search("file scanning", top_k=3, use_reranker=False)
    check("Pipeline search works", len(search_results) > 0)

    # Symbol search
    sym_results = pipeline.search_symbol("MeshIngestor", top_k=5)
    check("Symbol search finds MeshIngestor", len(sym_results) > 0)

    # File skeleton
    skel = pipeline.get_file_skeleton("repo_dumper.py")
    check("File skeleton works", "RepoScanner" in skel)

    # Stats
    p_stats = pipeline.get_stats()
    check("Pipeline stats available", p_stats["total_indexed_chunks"] > 0)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if failed > 0:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  {FAIL} {name} ({detail})")

    print("=" * 70)

    # Cleanup
    shutil.rmtree(test_db, ignore_errors=True)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
