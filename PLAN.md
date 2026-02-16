**Title: Project "TurboContext": The Instant-Sync Semantic Bridge for Agents**
**Date:** Feb 16, 2026
**Architect:** Senior Systems Engineer (You)
**Goal:** Achieve "Superior Agentic Coding" by eliminating context latency.

---

### **1. The 2026 Philosophy: "Context is Time"**
In 2026, the bottleneck isn't the LLM's intelligence; it's the **I/O bandwidth of context.**
*   **Current State:** Agents (Claude Code, Cursor) are "reactive." They search *after* you ask. This is slow (5-30s latency).
*   **TurboContext Vision:** The agent is "proactive." The moment you save a file, the semantic index updates. The agent already knows the code structure before it even generates the next token.

**The Metric:** **Time-to-Consistency (TtC) < 200ms.**
When a file changes on disk, the Vector Index must reflect that change in under 200 milliseconds.

---

### **2. The High-Level Architecture**

We are building a **Local-First, Event-Driven MCP Server**.

```text
[ Disk / File System ]
       ||
       || (File Watcher - Watchdog/Notify)
       \/
[ 1. TURBO INGESTOR ] <--- (Your Existing Tool)
       ||  (Sub-ms read + SHA-256 Hashing)
       ||
[ 2. STRUCTURAL PARSER ] <--- (Tree-sitter via Rust Bindings)
       ||  (Output: Function Signatures, Classes, Docstrings)
       ||
[ 3. EMBEDDING ENGINE ] <--- (Local ONNX Runtime / Nomic-Embed)
       ||  (Text -> Vector[768])
       ||
[ 4. LANCE DB ] <--- (Serverless Vector Store)
       ||
       || (Query Interface)
       ||
[ 5. MCP SERVER ] <===> [ CLAUDE CODE / CURSOR ]
```

---

### **3. The Implementation Blueprint (Phase by Phase)**

#### **Phase 1: The "Smart" Ingestor (Enhancing your Tool)**
Your current tool dumps *everything*. For vectors, we need *incremental* dumps.

*   **Logic:**
    1.  **Startup:** Scan repo (using your sub-ms logic). Build an in-memory `Map<FilePath, FileHash>`.
    2.  **Watcher:** Listen for file save events.
    3.  **Diffing:** When file changes, compare Hash. If changed, trigger the **Pipeline** for *only that file*.
*   **The Trap:** Don't re-index `.gitignore` files. Your tool already handles this well.
*   **Output:** Stream of modified file contents.

#### **Phase 2: Structural Chunking (The Context Winner)**
Raw text chunks are garbage for coding. We need **AST (Abstract Syntax Tree)** chunks.

*   **Tech:** `tree-sitter` (Python bindings are fine, Rust is better).
*   **Strategy:**
    *   **Chunk A:** The "Signature" (Function name + Args + Return Type + Docstring). *High Priority.*
    *   **Chunk B:** The "Body" (Implementation). *Medium Priority.*
*   **Why?** When the agent asks "How do I call `login`?", vector search should hit **Chunk A** instantly.

#### **Phase 3: The Local Embedding Engine (The Speed Demon)**
We cannot wait for OpenAI API calls for every file save. It must be local.

*   **Model:** `nomic-embed-text-v1.5` (Quantized) or `all-MiniLM-L6-v2`.
*   **Runtime:** **ONNX Runtime** (CPU optimized) or **CoreML** (Mac Silicon).
*   **Performance:** ~10ms per chunk on M3 Max.
*   **Fallback:** Allow config to use OpenAI/Cohere for users who want "Premium Accuracy" over speed.

#### **Phase 4: Storage (LanceDB)**
*   **Why LanceDB?** It's written in Rust, serverless (just files on disk), and supports hybrid search (Keyword + Vector) out of the box.
*   **Schema:**
    ```python
    class CodeChunk(LanceModel):
        id: str             # file_path + chunk_index
        text: str           # The code snippet
        vector: Vector(768) # The embedding
        metadata: struct    # { type: "function", name: "auth", file: "src/auth.py" }
        last_updated: float # Timestamp
    ```

#### **Phase 5: The MCP Server (The Interface)**
This is what Claude sees. We expose three specific tools:

1.  **`search_code(query: str)`**:
    *   Performs Hybrid Search (Vector + BM25 Keyword) in LanceDB.
    *   Returns: Top 5 code snippets with file paths.
2.  **`get_file_structure(path: str)`**:
    *   Uses Tree-sitter to return a "Skeleton" of a file (symbols only, no implementation).
    *   *Why?* Helps Claude understand a file without burning 5,000 tokens reading it.
3.  **`find_related(symbol_name: str)`**:
    *   Vectors search specifically for where a symbol is *defined* vs *used*.

---

### **4. The "Secret Sauce" (How we win)**

To make this "Flawless," we implement **Optimistic Caching**.

1.  **The "Ghost" Index:**
    When the user is typing, we don't index. We wait for the `save` event.
    *However*, if the user hasn't saved in 10 seconds, we take a snapshot from the editor (if using a VS Code extension hook) or just wait for disk save.

2.  **The Priority Queue:**
    If the user saves 50 files (git checkout), we prioritize indexing:
    1.  File currently open (if detectable).
    2.  `README` / Config files.
    3.  Source code.
    4.  Tests (Last).

3.  **Semantic Reranking (Rust Sidecar):**
    When Claude searches, we don't just return raw vectors. We use a tiny **Cross-Encoder** (Rust-based) to re-rank the top 20 results to the top 5. This makes the tool feel "smarter."

---

### **5. The Deployment Strategy (OSS Launch)**

**Name:** `turbo-mcp` (or `flash-context`)

**Repository Structure:**
```text
turbo-mcp/
├── src/
│   ├── ingester/      # Your Turbo Dumper Logic
│   ├── parser/        # Tree-sitter Logic
│   ├── db/            # LanceDB Wrapper
│   └── server.py      # FastMCP Entrypoint
├── binaries/          # Pre-compiled Tree-sitter parsers
├── pyproject.toml
├── .gitignore
└── README.md
```

**The README Hook:**
> "Stop waiting for your RAG to index. Turbo-MCP indexes your code in sub-milliseconds and feeds it to Claude via MCP. Zero config. Local-first."

**Installation:**
`pip install turbo-mcp`
`turbo-mcp install` (downloads models/parsers)
`turbo-mcp start`

---

### **6. Step-by-Step Execution Plan (Next 24 Hours)**

1.  **Refactor Dumper:** Separate the "Scanning" logic from the "XML Writing" logic. Make it yield objects (`FileObject`).
2.  **MVP Storage:** Install `lancedb` and `sentence-transformers`. Write a script that takes your dumper output and saves it to a DB.
3.  **MVP Server:** Use `mcp` library to wrap the LanceDB query in a `search` tool.
4.  **Test:** Connect Claude Desktop to your local MCP server. Ask it a question about a file you just saved.

**Senior Dev Note:** *Do not over-engineer the Graph RAG part yet.* EdgeQuake is doing graphs. You are doing **SPEED**. Stick to flat vectors + file structure first. Speed is your feature.

---
---

GPT is 100% correct. It accurately identified the **"Valley of Death"** between a cool script and a production engine.

However, as a Senior Dev, I’m going to give you the **"Cheat Codes"** to bypass the hardest parts identified in that feedback. You don't need to solve "Google-scale" problems (like AST diffing) to beat current tools. You just need to be smarter than the brute-force tools.

Here is the **Revised Execution Blueprint** to navigate those risks.

---

### 🛑 The "Cheat Codes" for the Hard Parts

#### **Risk 1: Incremental Parsing is Hard (AST Diffs)**
*   **The GPT Concern:** "If you re-parse the whole file every time, you lose TtC (Time-to-Consistency)."
*   **The Cheat Code:** **File-Level Atomicity.**
    *   Do *not* try to implement Tree-sitter's complex `edit()` method to diff the AST in memory yet. That is a nightmare of state management.
    *   **Reality:** Tree-sitter (Rust) parses a 2,000-line file in **< 5ms**.
    *   **Strategy:** If a file hash changes, **nuke that file's vectors and re-index the whole file.**
    *   *Why?* It’s simpler, robust, and for 99% of source files (which are < 500kb), it is sub-100ms. Don't premature optimize for 10MB source files.

#### **Risk 2: Re-ranking is Hard/Slow**
*   **The GPT Concern:** "Cheap vectors imply hallucinations."
*   **The Cheat Code:** **FlashRank.**
    *   Do not set up a heavy Cross-Encoder server.
    *   Use [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank). It is an ultra-lightweight (~4MB) re-ranking library based on ONNX. It runs on CPU, is incredibly fast, and creates "SOTA-ish" results from cheap vectors.
    *   **Pipeline:** Retrieve 50 items via LanceDB -> FlashRank -> Return top 5 to Claude.

#### **Risk 3: Hybrid Search Tuning**
*   **The Cheat Code:** **LanceDB Native FTS.**
    *   Don't build a separate BM25 engine. LanceDB has native Full-Text Search integration (via Tantivy). Use it. It’s one line of code: `.search(query, query_type="hybrid")`.

---

### 🚀 Phase 1: Refactoring the Dumper (Immediate Action)

We cannot build the Vector Engine until your Dumper stops speaking "XML" and starts speaking "Objects."

**Current State:** `RepoScanner` -> Writes XML string.
**Target State:** `RepoScanner` -> Yields `CodeFile` objects.

Here is the code structure you need to implement **TODAY**.

#### `src/ingestor.py`

```python
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Generator, List, Optional

# 1. The Atomic Unit of your system
@dataclass
class CodeFile:
    rel_path: str
    abs_path: str
    content: str
    file_hash: str
    language: str  # 'python', 'rust', 'go' (detected from extension)
    
    @property
    def is_binary(self) -> bool:
        # Move your binary detection logic here
        return b'\0' in self.content.encode('utf-8')[:1024]

class TurboIngestor:
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        # Keep a memory map of {path: hash} to detect changes instantly
        self.file_state: dict[str, str] = {} 

    def _calculate_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def scan(self) -> Generator[CodeFile, None, None]:
        """
        The Generator Pattern. 
        Instead of dumping a huge XML, we yield files one by one.
        This allows the Vector Engine to process File A while Dumper finds File B.
        """
        # ... Insert your high-performance scandir logic here ...
        # But instead of writing XML, do this:
        
        for file_path in found_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                f_hash = self._calculate_hash(content)
                
                # OPTIMIZATION: Skip if unchanged (The "Ghost" Index logic)
                rel_path = str(file_path.relative_to(self.root))
                if self.file_state.get(rel_path) == f_hash:
                    continue 

                self.file_state[rel_path] = f_hash
                
                yield CodeFile(
                    rel_path=rel_path,
                    abs_path=str(file_path),
                    content=content,
                    file_hash=f_hash,
                    language=file_path.suffix.lstrip('.')
                )
            except Exception:
                continue
```

---

### 📦 Phase 2: The "Vector Worker" (The Bridge)

This is the script that sits between your Dumper and LanceDB.

#### `src/indexer.py`

```python
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# 1. Setup Local Embeddings (Nomic or All-MiniLM)
db = lancedb.connect(".turbo_mcp/data")
func = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

# 2. Define the Schema (As discussed)
class CodeChunk(LanceModel):
    id: str             # path + "::" + function_name
    text: str           # The actual code
    vector: Vector(func.ndims()) = func.VectorField()
    path: str
    start_line: int
    last_updated: float

table = db.create_table("codebase", schema=CodeChunk, exist_ok=True)

# 3. The Indexing Loop
def sync_repo(ingestor):
    # This loop runs every time file watcher triggers
    for file_obj in ingestor.scan():
        print(f"⚡ Indexing: {file_obj.rel_path}")
        
        # A. PARSE (Tree-sitter logic goes here)
        # chunks = parse_code_to_chunks(file_obj)
        
        # B. EMBED & STORE
        # table.add(chunks)
        
        # C. CLEANUP (Remove old chunks for this file)
        # table.delete(f"path = '{file_obj.rel_path}'") 
```

---

### ✅ Senior Dev Directive: Your Next 3 Steps

Don't try to build the whole MCP server yet. Build the **Data Pipeline** first.

1.  **Refactor:** Modify your current dumper to match the `TurboIngestor` class above. It must yield objects, not XML.
2.  **Install:** `pip install lancedb sentence-transformers`.
3.  **Prototype:** Write a script that runs `TurboIngestor` -> `LanceDB`. Time it. If it handles your repo in under 5 seconds (initial load), we win.

**Do you want the Tree-sitter logic next (how to split that `CodeFile` into chunks)?** That is where the magic happens.