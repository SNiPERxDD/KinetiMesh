"""
KinetiMesh Performance Benchmark Suite.

Measures throughput and latency for each pipeline stage:
    1. Ingestor scan speed (files/sec)
    2. Parser throughput (chunks/sec)
    3. Embedding throughput (texts/sec)
    4. Search latency (ms/query)
    5. Full pipeline throughput (files/sec end-to-end)

Uses the synthetic E2E repo for deterministic, network-free benchmarks.
Run: pytest -m benchmark tests/bench_speed.py -v -s

Marker: @pytest.mark.benchmark — skipped by default.
"""

import textwrap
import time
import statistics
from pathlib import Path
from typing import Dict, List

import pytest

from src.ingestor.scanner import MeshIngestor
from src.parser.chunker import parse_file, CodeChunk
from src.db.store import embed_texts, embed_query, VectorStore
from src.server.pipeline import KinetiMeshPipeline


# ── Synthetic Repo (deterministic, no network) ─────────────────────────────────

def _build_bench_repo(tmp_path: Path, n_files: int = 50) -> Path:
    """Generate a synthetic repo with n_files Python files.

    Each file contains a class with 3 methods, producing ~10 chunks/file.

    Args:
        tmp_path: Base temporary directory.
        n_files: Number of Python files to generate.

    Returns:
        Path to the generated repo root.
    """
    repo = tmp_path / "bench_repo"
    repo.mkdir()

    for i in range(n_files):
        pkg = repo / f"pkg_{i // 10}"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").touch()

        content = textwrap.dedent(f'''\
            """Module {i} - auto-generated for benchmarking."""

            import os
            from typing import Optional, List


            class Service{i}:
                """Service class {i} with CRUD operations."""

                def __init__(self, name: str, config: Optional[dict] = None):
                    """Initialize Service{i}."""
                    self.name = name
                    self.config = config or {{}}

                def create(self, item: dict) -> dict:
                    """Create a new item in service {i}."""
                    item["id"] = hash(str(item))
                    return item

                def read(self, item_id: int) -> Optional[dict]:
                    """Read an item by ID from service {i}."""
                    return {{"id": item_id, "service": self.name}}

                def update(self, item_id: int, data: dict) -> dict:
                    """Update an existing item in service {i}."""
                    return {{"id": item_id, **data}}

                def delete(self, item_id: int) -> bool:
                    """Delete an item from service {i}."""
                    return True


            def helper_{i}(x: int, y: int) -> int:
                """Helper function {i}: compute x + y * {i}."""
                return x + y * {i}
        ''')
        (pkg / f"service_{i}.py").write_text(content, encoding="utf-8")

    return repo


def _timed(fn, *args, **kwargs):
    """Run a function and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ── Benchmarks ──────────────────────────────────────────────────────────────────

@pytest.mark.benchmark
class TestScanBenchmark:
    """Benchmark the file scanner."""

    def test_scan_throughput(self, tmp_path: Path):
        """Measure files/second for initial scan."""
        repo = _build_bench_repo(tmp_path, n_files=50)
        ingestor = MeshIngestor(str(repo))

        files, elapsed = _timed(lambda: list(ingestor.scan(incremental=False)))
        fps = len(files) / elapsed if elapsed > 0 else float("inf")

        print(f"\n[BENCH] Scan: {len(files)} files in {elapsed*1000:.1f}ms "
              f"({fps:.0f} files/sec)")
        assert len(files) >= 50, f"Expected >=50 files, got {len(files)}"
        assert elapsed < 5.0, f"Scan took {elapsed:.1f}s, should be <5s"

    def test_incremental_scan_speed(self, tmp_path: Path):
        """Incremental scan with no changes should be near-instant."""
        repo = _build_bench_repo(tmp_path, n_files=50)
        ingestor = MeshIngestor(str(repo))
        list(ingestor.scan(incremental=False))  # populate state

        _, elapsed = _timed(lambda: list(ingestor.scan(incremental=True)))
        print(f"\n[BENCH] Incremental scan (no changes): {elapsed*1000:.1f}ms")
        assert elapsed < 1.0, f"No-change scan took {elapsed:.1f}s"


@pytest.mark.benchmark
class TestParserBenchmark:
    """Benchmark the AST parser."""

    def test_parse_throughput(self, tmp_path: Path):
        """Measure chunks/second for parsing."""
        repo = _build_bench_repo(tmp_path, n_files=50)
        ingestor = MeshIngestor(str(repo))
        files = list(ingestor.scan(incremental=False))

        all_chunks = []

        def parse_all():
            for f in files:
                chunks = parse_file(f.content, f.language, f.rel_path)
                all_chunks.extend(chunks)

        _, elapsed = _timed(parse_all)
        cps = len(all_chunks) / elapsed if elapsed > 0 else float("inf")

        print(f"\n[BENCH] Parse: {len(all_chunks)} chunks from {len(files)} files "
              f"in {elapsed*1000:.1f}ms ({cps:.0f} chunks/sec)")
        assert len(all_chunks) > 200, f"Expected >200 chunks, got {len(all_chunks)}"
        assert elapsed < 5.0, f"Parsing took {elapsed:.1f}s"


@pytest.mark.benchmark
class TestEmbeddingBenchmark:
    """Benchmark the embedding model."""

    def test_embedding_throughput(self):
        """Measure texts/second for embedding (warm model)."""
        # Warm up model (exclude load time from benchmark)
        embed_texts(["warmup"], show_progress=False)

        texts = [f"def function_{i}(x): return x + {i}" for i in range(500)]
        vecs, elapsed = _timed(embed_texts, texts, show_progress=False)
        tps = len(texts) / elapsed if elapsed > 0 else float("inf")

        print(f"\n[BENCH] Embed: {len(texts)} texts in {elapsed*1000:.1f}ms "
              f"({tps:.0f} texts/sec)")
        assert vecs.shape == (500, 384)
        # M3 baseline: ~1500 texts/sec. Allow 100+ for slow CI.
        assert tps > 100, f"Embedding throughput {tps:.0f} texts/sec too low"

    def test_query_embedding_latency(self):
        """Single query embedding should be <50ms."""
        times = []
        for _ in range(10):
            _, elapsed = _timed(embed_query, "search for authentication handler")
            times.append(elapsed * 1000)

        avg = statistics.mean(times)
        print(f"\n[BENCH] Query embed: avg {avg:.1f}ms (n=10)")
        assert avg < 50, f"Query embedding avg {avg:.1f}ms, limit 50ms"


@pytest.mark.benchmark
class TestSearchBenchmark:
    """Benchmark search latency on indexed data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        """Build and index a 50-file repo for search benchmarks."""
        repo = _build_bench_repo(tmp_path, n_files=50)
        db_path = str(tmp_path / "bench_db")
        self.pipeline = KinetiMeshPipeline(str(repo), db_path=db_path)
        self.pipeline.full_index(incremental=False)

    def test_vector_search_latency(self):
        """Vector search should complete in <100ms."""
        queries = [
            "create new item", "delete by ID", "CRUD operations",
            "helper function compute", "service initialization config",
        ]
        times = []
        for q in queries:
            _, elapsed = _timed(
                self.pipeline.store.search_vector, q, top_k=5,
            )
            times.append(elapsed * 1000)

        avg = statistics.mean(times)
        print(f"\n[BENCH] Vector search: avg {avg:.1f}ms (n={len(queries)})")
        assert avg < 100, f"Avg vector search {avg:.1f}ms exceeds 100ms"

    def test_hybrid_search_latency(self):
        """Hybrid search (no reranker) should complete in <200ms."""
        queries = [
            "create item", "read by ID", "update data",
            "helper compute", "service config",
        ]
        times = []
        for q in queries:
            _, elapsed = _timed(
                self.pipeline.search, q, top_k=5, use_reranker=False,
            )
            times.append(elapsed * 1000)

        avg = statistics.mean(times)
        print(f"\n[BENCH] Hybrid search (no reranker): avg {avg:.1f}ms (n={len(queries)})")
        assert avg < 200, f"Avg hybrid search {avg:.1f}ms exceeds 200ms"

    def test_reranker_latency(self):
        """Hybrid search WITH reranker should complete in <1000ms."""
        _, elapsed = _timed(
            self.pipeline.search, "create new item", top_k=5, use_reranker=True,
        )
        print(f"\n[BENCH] Hybrid search (with reranker): {elapsed*1000:.1f}ms")
        assert elapsed < 1.0, f"Reranker search took {elapsed:.1f}s, limit 1.0s"


@pytest.mark.benchmark
class TestFullPipelineBenchmark:
    """Benchmark end-to-end pipeline throughput."""

    def test_full_pipeline_throughput(self, tmp_path: Path):
        """Measure end-to-end files/sec (scan → parse → embed → store)."""
        repo = _build_bench_repo(tmp_path, n_files=50)
        db_path = str(tmp_path / "bench_db")

        pipeline = KinetiMeshPipeline(str(repo), db_path=db_path)
        metrics, elapsed = _timed(pipeline.full_index, incremental=False)

        fps = metrics["files_scanned"] / elapsed if elapsed > 0 else float("inf")
        print(f"\n[BENCH] Full pipeline: {metrics['files_scanned']} files, "
              f"{metrics['chunks_parsed']} chunks in {elapsed:.2f}s ({fps:.1f} files/sec)")

        assert metrics["files_scanned"] >= 50
        assert metrics["chunks_parsed"] > 200
        assert elapsed < 30, f"Full pipeline took {elapsed:.1f}s, limit 30s"
