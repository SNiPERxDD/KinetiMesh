# KinetiMesh

**Living Mesh of Code Context - Instant-Sync Semantic Search for Agentic Coding Tools**

KinetiMesh is a local-first, event-driven MCP server that treats your repository as a constantly evolving mesh of interconnected code relationships. It indexes with sub-second latency and feeds semantically relevant context to LLM coding agents (Claude Code, Cursor, Windsurf) via the Model Context Protocol.

```
Stop waiting for your RAG to index.
KinetiMesh indexes your code in sub-milliseconds and feeds it to Claude via MCP.
Zero config. Local-first. Always in sync.
```

---

## Architecture

```
[ Disk / File System ]
       ||
       || (Watchdog File Watcher)
       \/
[ 1. KINETIC INGESTOR ]   -- os.scandir + SHA-256 incremental detection
       ||
[ 2. STRUCTURAL PARSER ]  -- Tree-sitter AST (Python, JS, TS)
       ||
[ 3. EMBEDDING ENGINE ]   -- all-MiniLM-L6-v2 (384-dim, local CPU)
       ||
[ 4. LANCE DB ]           -- Serverless vector + Tantivy FTS
       ||
[ 5. MCP SERVER ]    <===> [ CLAUDE CODE / CURSOR ]
```

---

## Installation

```bash
pip install kinetimesh
```

Or install from source:

```bash
git clone https://github.com/SNiPERxDD/KinetiMesh.git
cd KinetiMesh
pip install -e .
```

### Optional: ONNX Runtime (1.5-3x faster embeddings)

```bash
pip install kinetimesh[fast]
```

> **Note:** ONNX backend requires `optimum` with `transformers` v5 support. Currently blocked by an upstream version conflict. PyTorch CPU is the default backend and is already fast (~1500 texts/sec on Apple Silicon M3).

---

## Quick Start

```bash
# Start MCP server for current directory
kmesh start

# Start for a specific repository
kmesh start --repo /path/to/your/repo
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kinetimesh": {
      "command": "kmesh",
      "args": ["start", "--repo", "/path/to/your/repo"]
    }
  }
}
```

### Cursor Configuration

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "kinetimesh": {
      "command": "kmesh",
      "args": ["start"]
    }
  }
}
```

---

## How Agents Use KinetiMesh

### Tool Discovery (Automatic via MCP)

When Claude Desktop, Cursor, or Windsurf connects to the KinetiMesh MCP server, the agent automatically discovers available tools through the Model Context Protocol's `tools/list` method. Each tool includes:

- **Name and description**: What the tool does
- **Input schema**: Parameters, types, and constraints (JSON Schema)
- **Examples**: Sample queries agents can learn from

**The agent sees this automatically:**

```json
{
  "name": "search_code",
  "description": "Semantic + keyword hybrid search across the indexed codebase. Returns file paths, symbol names, code snippets, and relevance scores. Use this to find implementations, locate code patterns, or answer 'where is X' questions.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Natural language or keyword search query"},
      "top_k": {"type": "integer", "default": 5, "description": "Number of results to return"}
    },
    "required": ["query"]
  }
}
```

### Agent Workflow Example

**User:** "Fix the authentication bug in the API"

**Agent (internal reasoning with KinetiMesh):**

1. `search_code("authentication API middleware")` → Finds `auth.py:42` (OAuth2 flow)
2. `get_file_structure("src/auth.py")` → Gets function signatures without burning 800 lines of tokens
3. `find_related("authenticate_user")` → Locates all 12 call sites
4. Agent reads the relevant code, identifies the bug, proposes fix
5. User saves file → Watchdog triggers re-index in <200ms → Agent's next query sees updated code

**No user back-and-forth. No "can you show me X?" The agent navigates autonomously.**

### Supported Tools

**Claude Desktop**: Full MCP support (tools, sampling, prompts, resources)  
**Cursor**: MCP tools support via `.cursor/mcp.json`  
**Windsurf**: MCP tools support via extensions  
**Cline (VSCode)**: MCP tools support via settings  
**Zed**: Experimental MCP support  

### Installation for Different Tools

#### Claude Desktop
Location: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "kinetimesh": {
      "command": "kmesh",
      "args": ["start", "--repo", "/absolute/path/to/your/repo"]
    }
  }
}
```

Restart Claude Desktop after editing the config.

#### Cursor
Location: `.cursor/mcp.json` in your workspace root

```json
{
  "mcpServers": {
    "kinetimesh": {
      "command": "kmesh",
      "args": ["start"]
    }
  }
}
```

Cursor automatically uses the workspace directory as the repo path.

#### Windsurf
Add via Settings → MCP Servers → Add Server:
- **Command**: `kmesh`
- **Args**: `["start", "--repo", "/path/to/repo"]`

#### Cline (VSCode Extension)
Add to VSCode settings (`settings.json`):

```json
{
  "cline.mcpServers": {
    "kinetimesh": {
      "command": "kmesh",
      "args": ["start"]
    }
  }
}
```

---

## MCP Tools

KinetiMesh exposes four tools to LLM coding agents:

### `search_code(query, top_k=5)`
Hybrid vector + keyword search across the indexed codebase. Returns file paths, symbol names, code snippets, and relevance scores.

```
"find the authentication middleware"
"database connection pool setup"
"error handling in API routes"
```

### `get_file_structure(path)`
Returns a structural skeleton of a file (function signatures, class definitions, docstrings) without the full implementation. Saves tokens.

### `find_related(symbol_name, top_k=10)`
Finds where a symbol is defined and used across the codebase. Useful for understanding call graphs and dependencies.

### `get_index_stats()`
Returns statistics about the index: file count, chunk count, performance metrics, watcher status.

---

## Performance

Benchmarked on Apple Silicon M3, FastAPI repository (2858 files, 16480 chunks, 360K lines):

| Operation | Latency |
|---|---|
| Full scan (2858 files) | 270ms |
| Full parse (16480 chunks) | 312ms |
| Vector search | 10ms avg |
| Full-text search (warm) | 2.5ms avg |
| Hybrid search | 13ms avg |
| Incremental scan (no changes) | 1.5ms |

### Search Quality (12 known-answer queries on KinetiMesh codebase)

| Mode | Top-3 Precision | Latency |
|---|---|---|
| Vector Only | 100% | 10ms |
| FTS Only | 58% | 4ms |
| **Hybrid (default)** | **100%** | **13ms** |
| Hybrid + FlashRank | 83% | 359ms |

Hybrid without reranking is the default in the MCP server. FlashRank adds latency without improving quality for code-specific queries because priority-weighted scoring already boosts function signatures.

### GPU Note

MPS (Metal Performance Shaders) on Apple Silicon is **2x slower** than CPU for the all-MiniLM-L6-v2 model (22M params). The Metal transfer overhead exceeds computation benefit for small models. CPU with ARM NEON SIMD is the optimal backend.

---

## Supported Languages

### AST Parsing (Tree-sitter)
Python, JavaScript, TypeScript, TSX

### Extension Detection (30+ languages)
Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, Scala, Lua, R, Bash, YAML, TOML, JSON, XML, HTML, CSS, SCSS, Markdown, SQL, Dockerfile, HCL, Protobuf, GraphQL, Vue, Svelte

Unsupported languages fall back to line-based chunking.

---

## How It Works

### Chunking Strategy

KinetiMesh uses AST-aware semantic chunking, not naive text splitting:

- **Signature** (priority 2.0): `def func(args) -> RetType:` + docstring. The agent's primary search target.
- **Implementation** (priority 1.0): Full function body. For when the agent needs the actual code.
- **Class Definition** (priority 1.8): Class name, bases, docstring.
- **Import** (priority 0.3): Import blocks. Low priority, rarely searched.
- **Module Docstring** (priority 0.5): File-level documentation.
- **Fallback** (priority 0.5): Line-based chunks for non-AST languages.

### Incremental Indexing

On startup, KinetiMesh scans the repository and builds an in-memory `{path: SHA-256}` state map. When a file changes (detected by watchdog), only that file is re-indexed using nuke-and-replace:

1. Detect change via SHA-256 hash comparison
2. Parse the changed file into AST chunks
3. Delete old vectors for that file
4. Embed and insert new chunks

This achieves sub-200ms Time-to-Consistency (TtC) for typical source files.

---

## Configuration

KinetiMesh uses sensible defaults. No configuration file is needed.

### Data Storage

All data is stored in `.kmesh/` in the working directory:

```
.kmesh/
  data/       -- LanceDB vector database
  models/     -- FlashRank model cache
  server.log  -- Server log file
```

### Ignored Paths

The following directories are automatically skipped: `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`, `.kmesh`, and others. Binary files, lock files, and files > 2MB are also skipped.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run integration tests (49 tests)
python tests/test_integration.py

# Run quality evaluation
python tests/test_quality.py

# Run stress test (requires FastAPI clone at /tmp/kmesh_test_fastapi)
git clone --depth 1 https://github.com/fastapi/fastapi.git /tmp/kmesh_test_fastapi
python tests/test_stress.py
```

---

## License

MIT
