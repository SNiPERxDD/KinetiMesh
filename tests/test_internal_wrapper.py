"""
Integration test for KinetiMesh internal wrapper module.

Tests all internal functions to verify they work correctly without
MCP server dependency. Validates initialization, search, file structure,
symbol tracking, statistics, and diagnostics.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_wrapper import (
    init_kinetimesh,
    search_code_internal,
    get_file_structure_internal,
    find_related_internal,
    get_index_stats_internal,
    doctor_internal,
    shutdown_kinetimesh,
)


def test_initialization():
    """Test pipeline initialization."""
    print("=" * 60)
    print("TEST 1: Initialization")
    print("=" * 60)
    
    repo_path = "."
    metrics = init_kinetimesh(repo_path)
    
    assert metrics["status"] == "indexed" or metrics["status"] == "no_changes", f"Unexpected status: {metrics['status']}"
    assert "repo_path" in metrics
    assert "total_time_ms" in metrics
    
    print(f"✓ Status: {metrics['status']}")
    print(f"✓ Files scanned: {metrics.get('files_scanned', 0)}")
    print(f"✓ Chunks parsed: {metrics.get('chunks_parsed', 0)}")
    print(f"✓ Init time: {metrics.get('total_time_ms', 0):.1f}ms")
    print()


def test_search_code():
    """Test code search functionality."""
    print("=" * 60)
    print("TEST 2: Search Code")
    print("=" * 60)
    
    results = search_code_internal("vector search", top_k=3)
    
    assert isinstance(results, str)
    assert len(results) > 0, "Search returned empty results"
    assert "Found" in results or "No results" in results
    
    print("✓ Search completed successfully")
    print(f"✓ Results length: {len(results)} chars")
    print(f"Preview:\n{results[:400]}...")
    print()


def test_file_structure():
    """Test file skeleton generation."""
    print("=" * 60)
    print("TEST 3: Get File Structure")
    print("=" * 60)
    
    skeleton = get_file_structure_internal("src/server/pipeline.py")
    
    assert isinstance(skeleton, str)
    assert len(skeleton) > 0, "Skeleton is empty"
    assert "KinetiMeshPipeline" in skeleton, "Expected class not found in skeleton"
    
    print("✓ File structure retrieved")
    print(f"✓ Skeleton length: {len(skeleton)} chars")
    print(f"Preview:\n{skeleton[:400]}...")
    print()


def test_find_related():
    """Test symbol search functionality."""
    print("=" * 60)
    print("TEST 4: Find Related Symbols")
    print("=" * 60)
    
    related = find_related_internal("KinetiMeshPipeline", top_k=5)
    
    assert isinstance(related, str)
    assert len(related) > 0, "No related symbols found"
    assert "KinetiMeshPipeline" in related
    
    print("✓ Symbol references found")
    print(f"✓ Results length: {len(related)} chars")
    print(f"Preview:\n{related[:400]}...")
    print()


def test_index_stats():
    """Test index statistics retrieval."""
    print("=" * 60)
    print("TEST 5: Get Index Stats")
    print("=" * 60)
    
    stats = get_index_stats_internal()
    
    assert isinstance(stats, str)
    assert len(stats) > 0
    assert "repo_path" in stats
    assert "total_indexed_files" in stats
    assert "total_indexed_chunks" in stats
    
    print("✓ Stats retrieved successfully")
    print(stats)
    print()


def test_doctor():
    """Test system diagnostics."""
    print("=" * 60)
    print("TEST 6: System Diagnostics (Doctor)")
    print("=" * 60)
    
    diagnostics = doctor_internal()
    
    assert isinstance(diagnostics, str)
    assert len(diagnostics) > 0
    assert "Database Health" in diagnostics
    assert "Write Permissions" in diagnostics
    assert "Index Health" in diagnostics
    
    print("✓ Diagnostics completed")
    print(diagnostics)
    print()


def test_concurrent_access():
    """Test thread safety with concurrent searches."""
    print("=" * 60)
    print("TEST 7: Concurrent Access (Thread Safety)")
    print("=" * 60)
    
    from concurrent.futures import ThreadPoolExecutor
    
    queries = [
        "embedding model",
        "vector store",
        "search hybrid",
        "index chunks",
    ]
    
    def search_task(query):
        return search_code_internal(query, top_k=2)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(search_task, queries))
    
    assert len(results) == 4
    for i, result in enumerate(results):
        assert isinstance(result, str)
        assert len(result) > 0, f"Query {i} returned empty result"
    
    print("✓ Concurrent access successful")
    print(f"✓ Completed {len(queries)} concurrent searches")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KinetiMesh Internal Wrapper Integration Test")
    print("=" * 60)
    print()
    
    start_time = time.perf_counter()
    
    try:
        # Note: test_initialization already calls init_kinetimesh
        test_initialization()
        test_search_code()
        test_file_structure()
        test_find_related()
        test_index_stats()
        test_doctor()
        test_concurrent_access()
        
        elapsed = time.perf_counter() - start_time
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print(f"Total time: {elapsed:.2f}s")
        print()
        
        # Cleanup
        shutdown_kinetimesh()
        print("✓ Shutdown complete")
        
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
