# KinetiMesh Internal Usage Guide

## Overview

The `src.mcp_wrapper` module provides direct Python access to all KinetiMesh tools without requiring MCP server initialization or command-line usage. This enables AI agents and other Python code to use KinetiMesh functionality internally.

## Installation

KinetiMesh must be installed in editable mode for internal usage:

```bash
cd /path/to/KinetiMesh
pip install -e .
```

## Quick Start

```python
from src.mcp_wrapper import (
    init_kinetimesh,
    search_code_internal,
    get_file_structure_internal,
    find_related_internal,
    get_index_stats_internal,
    doctor_internal,
    shutdown_kinetimesh
)

# Initialize once per session
metrics = init_kinetimesh("/path/to/repository")
print(f"Indexed {metrics['files_scanned']} files")

# Search for code
results = search_code_internal("authentication middleware", top_k=5)
print(results)

# Get file structure
skeleton = get_file_structure_internal("src/auth/middleware.py")
print(skeleton)

# Find symbol references
references = find_related_internal("authenticate_user", top_k=10)
print(references)

# Get index statistics
stats = get_index_stats_internal()
print(stats)

# Run diagnostics
diagnostics = doctor_internal()
print(diagnostics)

# Cleanup (optional)
shutdown_kinetimesh()
```

## API Reference

### `init_kinetimesh(repo_path: str, db_path: str = ".kmesh/data") -> Dict[str, Any]`

Initialize the KinetiMesh pipeline for a repository.

**Parameters:**
- `repo_path`: Path to repository root (absolute or relative)
- `db_path`: LanceDB storage path (default: `.kmesh/data`)

**Returns:**
Dictionary with initialization metrics:
- `status`: "ready" if successful
- `repo_path`: Absolute repository path
- `files_scanned`: Number of files indexed
- `chunks_parsed`: Number of code chunks created
- `total_time_ms`: Indexing duration

**Raises:**
- `RuntimeError`: Pipeline initialization failed
- `OSError`: Repository path invalid or inaccessible

**Example:**
```python
metrics = init_kinetimesh(".")
if metrics['status'] == 'ready':
    print(f"Ready! Indexed {metrics['chunks_parsed']} chunks")
```

### `search_code_internal(query: str, top_k: int = 5) -> str`

Hybrid semantic + keyword search across indexed code.

**Parameters:**
- `query`: Natural language search query
- `top_k`: Number of results (1-20, default: 5)

**Returns:**
Formatted string with search results including:
- File paths and line numbers
- Symbol names and types
- Code snippets
- Relevance scores

**Raises:**
- `RuntimeError`: Pipeline not initialized

**Example:**
```python
results = search_code_internal("database connection pool", top_k=3)
# Returns formatted results with file paths and code snippets
```

### `get_file_structure_internal(path: str) -> str`

Get file skeleton (signatures only, no implementation).

**Parameters:**
- `path`: File path relative to repository root

**Returns:**
Formatted string with:
- Import statements
- Class definitions with docstrings
- Function signatures with docstrings
- No implementation bodies

**Raises:**
- `RuntimeError`: Pipeline not initialized

**Example:**
```python
skeleton = get_file_structure_internal("src/server/pipeline.py")
# Returns file structure without implementation details
```

### `find_related_internal(symbol_name: str, top_k: int = 10) -> str`

Find symbol definitions and usages across codebase.

**Parameters:**
- `symbol_name`: Function, class, or method name
- `top_k`: Maximum results (1-20, default: 10)

**Returns:**
Formatted string with:
- DEFINITIONS section: Where symbol is defined
- USAGES section: Where symbol is referenced

**Raises:**
- `RuntimeError`: Pipeline not initialized

**Example:**
```python
references = find_related_internal("KinetiMeshPipeline")
# Returns all definitions and usages of the class
```

### `get_index_stats_internal() -> str`

Get current index statistics and health metrics.

**Returns:**
Formatted string with:
- Repository path
- File and chunk counts
- Performance metrics
- Watcher status
- Failed files (if any)

**Raises:**
- `RuntimeError`: Pipeline not initialized

**Example:**
```python
stats = get_index_stats_internal()
# Returns comprehensive index statistics
```

### `doctor_internal() -> str`

Run comprehensive system diagnostics.

**Returns:**
Formatted diagnostic report with:
- Database health check
- Write permissions status
- Memory usage
- Failed files analysis
- Index health metrics
- Recommendations

**Raises:**
- `RuntimeError`: Pipeline not initialized

**Example:**
```python
diagnostics = doctor_internal()
# Returns detailed health report
```

### `shutdown_kinetimesh() -> None`

Cleanup and shutdown the pipeline.

Stops file watcher and releases resources. Optional - pipeline will be garbage collected automatically.

**Example:**
```python
shutdown_kinetimesh()
```

## Thread Safety

All functions are thread-safe and use internal locking. Multiple threads can call wrapper functions concurrently.

```python
from concurrent.futures import ThreadPoolExecutor

def search_task(query):
    return search_code_internal(query, top_k=3)

with ThreadPoolExecutor(max_workers=4) as executor:
    queries = ["auth", "database", "api", "error"]
    results = executor.map(search_task, queries)
```

## Error Handling

```python
from src.mcp_wrapper import init_kinetimesh, search_code_internal

try:
    init_kinetimesh("/path/to/repo")
except OSError as e:
    print(f"Repository not found: {e}")
except RuntimeError as e:
    print(f"Initialization failed: {e}")

try:
    results = search_code_internal("query")
except RuntimeError:
    print("Pipeline not initialized - call init_kinetimesh() first")
```

## Performance Considerations

### Initialization
- First initialization performs full repository scan (270ms for 2858 files on M3)
- Subsequent calls to `init_kinetimesh()` with same repo reuse existing pipeline
- File watcher runs in background thread for live updates

### Search Operations
- Vector search: ~10ms average
- Keyword search: ~2.5ms average
- Hybrid search: ~13ms average

### Memory Usage
- Embedding model: ~100MB
- Vector database: Depends on repository size
- Use `doctor_internal()` to check memory usage

## Example: Agent Usage Pattern

```python
from src.mcp_wrapper import (
    init_kinetimesh,
    search_code_internal,
    get_file_structure_internal,
    find_related_internal
)

# Initialize for current project
init_kinetimesh(".")

# Agent workflow: Fix authentication bug
# Step 1: Find authentication code
auth_results = search_code_internal("authentication API middleware")

# Step 2: Get file structure for relevant file
structure = get_file_structure_internal("src/auth.py")

# Step 3: Find all usages of auth function
usages = find_related_internal("authenticate_user")

# Agent now has complete context to fix the bug
```

## Comparison: CLI vs Internal

### CLI Usage (Traditional MCP)
```json
{
  "mcpServers": {
    "kinetimesh": {
      "command": "kmesh",
      "args": ["--repo", "/path/to/repo"]
    }
  }
}
```
- Spawns separate stdio process
- Communication via JSON-RPC
- Cannot be called directly from Python

### Internal Usage (This Module)
```python
from src.mcp_wrapper import init_kinetimesh, search_code_internal

init_kinetimesh("/path/to/repo")
results = search_code_internal("query")
```
- Direct Python function calls
- No process spawning
- Same functionality, zero overhead

## Limitations

- Single global pipeline instance (one repository per process)
- Call `shutdown_kinetimesh()` before initializing different repository
- File watcher uses background thread (daemon, automatically cleaned up)

## Troubleshooting

### "Pipeline not initialized" Error
```python
# Call init_kinetimesh() first
init_kinetimesh(".")
```

### "Repository path does not exist" Error
```python
# Use absolute path or valid relative path
init_kinetimesh("/absolute/path/to/repo")
```

### Import Errors
```python
# Install in editable mode
# cd /path/to/KinetiMesh
# pip install -e .
```

### Check System Health
```python
diagnostics = doctor_internal()
print(diagnostics)
```
