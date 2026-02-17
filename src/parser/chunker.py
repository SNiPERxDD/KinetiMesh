"""
KinetiMesh Structural Parser - AST-Based Semantic Chunking.

Uses Tree-sitter to parse source files into semantic chunks:
  - Chunk Type A: Signatures (function name + args + return type + docstring) [HIGH PRIORITY]
  - Chunk Type B: Full body (implementation) [MEDIUM PRIORITY]

This produces chunks that are far superior to naive text splitting
for vector search relevance in coding contexts.

Supported Languages: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++.
Fallback: Line-based chunking for unsupported languages.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple

import tree_sitter
from tree_sitter import Language, Parser

# ── Language Registry ───────────────────────────────────────────────────────────
# Tree-sitter language modules (pip-installed)
_LANGUAGE_CACHE: Dict[str, Language] = {}


def _get_language(lang_name: str) -> Optional[Language]:
    """Load and cache a tree-sitter Language object.

    Args:
        lang_name: Language identifier (e.g. 'python', 'javascript').

    Returns:
        Language object or None if unsupported.
    """
    if lang_name in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[lang_name]

    try:
        if lang_name == "python":
            import tree_sitter_python
            lang = Language(tree_sitter_python.language())
        elif lang_name == "javascript":
            import tree_sitter_javascript
            lang = Language(tree_sitter_javascript.language())
        elif lang_name == "typescript":
            import tree_sitter_typescript
            lang = Language(tree_sitter_typescript.language_typescript())
        elif lang_name == "tsx":
            import tree_sitter_typescript
            lang = Language(tree_sitter_typescript.language_tsx())
        else:
            return None

        _LANGUAGE_CACHE[lang_name] = lang
        return lang
    except (ImportError, AttributeError, Exception):
        return None


# ── Chunk Data Model ────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    """Classification of code chunk types for priority ranking."""
    SIGNATURE = "signature"
    IMPLEMENTATION = "implementation"
    CLASS_DEF = "class_definition"
    IMPORT = "import"
    MODULE_DOC = "module_docstring"
    FALLBACK = "fallback"


@dataclass(slots=True)
class CodeChunk:
    """A semantically meaningful piece of code extracted via AST parsing.

    Attributes:
        text: The raw code text of the chunk.
        chunk_type: Classification (signature, implementation, etc.).
        symbol_name: Name of the function/class/method if applicable.
        start_line: 1-based start line in the original file.
        end_line: 1-based end line in the original file.
        parent_symbol: Parent class/module name for methods.
        language: Source language.
        file_path: Relative file path.
        priority: Search priority weight (higher = more important).
    """
    text: str
    chunk_type: ChunkType
    symbol_name: str
    start_line: int
    end_line: int
    parent_symbol: str = ""
    language: str = ""
    file_path: str = ""
    priority: float = 1.0

    @property
    def chunk_id(self) -> str:
        """Generate a unique ID for this chunk: path::symbol::type."""
        parts = [self.file_path, self.symbol_name or f"L{self.start_line}", self.chunk_type.value]
        return "::".join(parts)

    @property
    def search_text(self) -> str:
        """Text optimized for embedding/search. Prepends metadata context."""
        prefix_parts = []
        if self.language:
            prefix_parts.append(f"Language: {self.language}")
        if self.parent_symbol:
            prefix_parts.append(f"Class: {self.parent_symbol}")
        if self.symbol_name:
            prefix_parts.append(f"Symbol: {self.symbol_name}")
        prefix_parts.append(f"Type: {self.chunk_type.value}")

        prefix = " | ".join(prefix_parts)
        return f"{prefix}\n{self.text}"


# ── Node Query Helpers ──────────────────────────────────────────────────────────

def _node_text(node: tree_sitter.Node, source_bytes: bytes) -> str:
    """Extract text from a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _find_child_by_type(node: tree_sitter.Node, type_name: str) -> Optional[tree_sitter.Node]:
    """Find first direct child of a specific type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _find_children_by_type(node: tree_sitter.Node, type_name: str) -> List[tree_sitter.Node]:
    """Find all direct children of a specific type."""
    return [child for child in node.children if child.type == type_name]


def _extract_docstring(node: tree_sitter.Node, source_bytes: bytes, lang: str) -> str:
    """Extract docstring from a function/class body node.

    Args:
        node: The function_definition or class_definition node.
        source_bytes: Raw source file bytes.
        lang: Language identifier.

    Returns:
        Docstring text or empty string.
    """
    if lang == "python":
        body = _find_child_by_type(node, "block")
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == "string":
                    return _node_text(expr, source_bytes).strip("\"'").strip()
    elif lang in ("javascript", "typescript", "tsx"):
        # JSDoc comments preceding the node
        if node.prev_sibling and node.prev_sibling.type == "comment":
            comment = _node_text(node.prev_sibling, source_bytes)
            if comment.startswith("/**"):
                return comment
    return ""


# ── Python-Specific Parsing ────────────────────────────────────────────────────

def _parse_python_signature(node: tree_sitter.Node, source_bytes: bytes) -> str:
    """Extract Python function signature (def line + type hints).

    Returns the 'def func_name(args) -> RetType:' line.
    """
    name_node = _find_child_by_type(node, "identifier")
    params_node = _find_child_by_type(node, "parameters")
    return_node = _find_child_by_type(node, "type")

    name = _node_text(name_node, source_bytes) if name_node else "unknown"
    params = _node_text(params_node, source_bytes) if params_node else "()"
    ret = f" -> {_node_text(return_node, source_bytes)}" if return_node else ""

    return f"def {name}{params}{ret}:"


def _parse_python(root_node: tree_sitter.Node, source_bytes: bytes, file_path: str) -> List[CodeChunk]:
    """Parse a Python source file into semantic chunks.

    Strategy:
        - Module-level docstring -> MODULE_DOC chunk
        - Import blocks -> IMPORT chunk
        - Function definitions -> SIGNATURE + IMPLEMENTATION chunks
        - Class definitions -> CLASS_DEF + method SIGNATURE chunks
    """
    chunks: List[CodeChunk] = []

    # Module docstring
    if root_node.children:
        first = root_node.children[0]
        if first.type == "expression_statement" and first.children:
            expr = first.children[0]
            if expr.type == "string":
                chunks.append(CodeChunk(
                    text=_node_text(expr, source_bytes),
                    chunk_type=ChunkType.MODULE_DOC,
                    symbol_name="__module__",
                    start_line=expr.start_point[0] + 1,
                    end_line=expr.end_point[0] + 1,
                    language="python",
                    file_path=file_path,
                    priority=0.5,
                ))

    # Collect imports as a single chunk
    import_lines = []
    import_start = None
    import_end = None

    for child in root_node.children:
        if child.type in ("import_statement", "import_from_statement"):
            text = _node_text(child, source_bytes)
            import_lines.append(text)
            if import_start is None:
                import_start = child.start_point[0] + 1
            import_end = child.end_point[0] + 1

    if import_lines:
        chunks.append(CodeChunk(
            text="\n".join(import_lines),
            chunk_type=ChunkType.IMPORT,
            symbol_name="__imports__",
            start_line=import_start or 1,
            end_line=import_end or 1,
            language="python",
            file_path=file_path,
            priority=0.3,
        ))

    # Functions and classes
    _extract_python_definitions(root_node, source_bytes, file_path, chunks, parent="")

    return chunks


def _extract_python_definitions(
    node: tree_sitter.Node,
    source_bytes: bytes,
    file_path: str,
    chunks: List[CodeChunk],
    parent: str = "",
) -> None:
    """Recursively extract function and class definitions from Python AST."""
    for child in node.children:
        if child.type == "function_definition" or (
            child.type == "decorated_definition"
            and _find_child_by_type(child, "function_definition")
        ):
            func_node = child
            if child.type == "decorated_definition":
                func_node = _find_child_by_type(child, "function_definition")

            name_node = _find_child_by_type(func_node, "identifier")
            name = _node_text(name_node, source_bytes) if name_node else "unknown"
            full_name = f"{parent}.{name}" if parent else name

            sig = _parse_python_signature(func_node, source_bytes)
            docstring = _extract_docstring(func_node, source_bytes, "python")

            sig_text = sig
            if docstring:
                sig_text = f"{sig}\n    \"\"\"{docstring}\"\"\""

            # Signature chunk (HIGH PRIORITY)
            chunks.append(CodeChunk(
                text=sig_text,
                chunk_type=ChunkType.SIGNATURE,
                symbol_name=full_name,
                start_line=child.start_point[0] + 1,
                end_line=child.start_point[0] + 1,
                parent_symbol=parent,
                language="python",
                file_path=file_path,
                priority=2.0,
            ))

            # Full implementation chunk (MEDIUM PRIORITY)
            full_text = _node_text(child, source_bytes)
            chunks.append(CodeChunk(
                text=full_text,
                chunk_type=ChunkType.IMPLEMENTATION,
                symbol_name=full_name,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                parent_symbol=parent,
                language="python",
                file_path=file_path,
                priority=1.0,
            ))

        elif child.type == "class_definition" or (
            child.type == "decorated_definition"
            and _find_child_by_type(child, "class_definition")
        ):
            cls_node = child
            if child.type == "decorated_definition":
                cls_node = _find_child_by_type(child, "class_definition")
                if not cls_node:
                    continue

            name_node = _find_child_by_type(cls_node, "identifier")
            cls_name = _node_text(name_node, source_bytes) if name_node else "UnknownClass"
            full_cls = f"{parent}.{cls_name}" if parent else cls_name

            # Class definition chunk
            docstring = _extract_docstring(cls_node, source_bytes, "python")
            cls_header = f"class {cls_name}:"
            if docstring:
                cls_header = f"class {cls_name}:\n    \"\"\"{docstring}\"\"\""

            # Superclasses
            arg_list = _find_child_by_type(cls_node, "argument_list")
            if arg_list:
                bases = _node_text(arg_list, source_bytes)
                cls_header = cls_header.replace(f"class {cls_name}:", f"class {cls_name}{bases}:")

            chunks.append(CodeChunk(
                text=cls_header,
                chunk_type=ChunkType.CLASS_DEF,
                symbol_name=full_cls,
                start_line=child.start_point[0] + 1,
                end_line=child.start_point[0] + 1,
                language="python",
                file_path=file_path,
                priority=1.8,
            ))

            # Recurse into class body for methods
            body = _find_child_by_type(cls_node, "block")
            if body:
                _extract_python_definitions(body, source_bytes, file_path, chunks, parent=full_cls)


# ── JavaScript/TypeScript Parsing ───────────────────────────────────────────────

def _parse_js_ts(root_node: tree_sitter.Node, source_bytes: bytes, file_path: str, lang: str) -> List[CodeChunk]:
    """Parse JavaScript/TypeScript source into semantic chunks."""
    chunks: List[CodeChunk] = []
    _extract_js_definitions(root_node, source_bytes, file_path, chunks, lang, parent="")
    return chunks


def _extract_js_definitions(
    node: tree_sitter.Node,
    source_bytes: bytes,
    file_path: str,
    chunks: List[CodeChunk],
    lang: str,
    parent: str = "",
) -> None:
    """Recursively extract function and class definitions from JS/TS AST."""
    for child in node.children:
        if child.type in (
            "function_declaration",
            "method_definition",
            "arrow_function",
            "generator_function_declaration",
        ):
            name_node = _find_child_by_type(child, "identifier") or _find_child_by_type(child, "property_identifier")
            name = _node_text(name_node, source_bytes) if name_node else "anonymous"
            full_name = f"{parent}.{name}" if parent else name

            # Signature
            params_node = _find_child_by_type(child, "formal_parameters")
            params = _node_text(params_node, source_bytes) if params_node else "()"
            sig_text = f"function {name}{params}"

            chunks.append(CodeChunk(
                text=sig_text,
                chunk_type=ChunkType.SIGNATURE,
                symbol_name=full_name,
                start_line=child.start_point[0] + 1,
                end_line=child.start_point[0] + 1,
                parent_symbol=parent,
                language=lang,
                file_path=file_path,
                priority=2.0,
            ))

            # Full implementation
            chunks.append(CodeChunk(
                text=_node_text(child, source_bytes),
                chunk_type=ChunkType.IMPLEMENTATION,
                symbol_name=full_name,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                parent_symbol=parent,
                language=lang,
                file_path=file_path,
                priority=1.0,
            ))

        elif child.type == "class_declaration":
            name_node = _find_child_by_type(child, "identifier") or _find_child_by_type(child, "type_identifier")
            cls_name = _node_text(name_node, source_bytes) if name_node else "UnknownClass"
            full_cls = f"{parent}.{cls_name}" if parent else cls_name

            chunks.append(CodeChunk(
                text=_node_text(child, source_bytes).split("{")[0].strip() + " {",
                chunk_type=ChunkType.CLASS_DEF,
                symbol_name=full_cls,
                start_line=child.start_point[0] + 1,
                end_line=child.start_point[0] + 1,
                language=lang,
                file_path=file_path,
                priority=1.8,
            ))

            body = _find_child_by_type(child, "class_body")
            if body:
                _extract_js_definitions(body, source_bytes, file_path, chunks, lang, parent=full_cls)

        elif child.type in ("lexical_declaration", "variable_declaration"):
            # Handle: const fn = () => {} or const fn = function() {}
            for decl in _find_children_by_type(child, "variable_declarator"):
                name_node = _find_child_by_type(decl, "identifier")
                value_node = decl.children[-1] if decl.children else None
                if name_node and value_node and value_node.type in ("arrow_function", "function"):
                    name = _node_text(name_node, source_bytes)
                    full_name = f"{parent}.{name}" if parent else name

                    chunks.append(CodeChunk(
                        text=_node_text(child, source_bytes),
                        chunk_type=ChunkType.IMPLEMENTATION,
                        symbol_name=full_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        parent_symbol=parent,
                        language=lang,
                        file_path=file_path,
                        priority=1.5,
                    ))

        elif child.type == "export_statement":
            # Recurse into exports
            _extract_js_definitions(child, source_bytes, file_path, chunks, lang, parent=parent)

        # Recurse into program-level blocks
        if child.type in ("program", "statement_block"):
            _extract_js_definitions(child, source_bytes, file_path, chunks, lang, parent=parent)


# ── Fallback Line-Based Chunking ────────────────────────────────────────────────

def _fallback_chunk(content: str, file_path: str, language: str, max_lines: int = 50) -> List[CodeChunk]:
    """Fallback chunking for unsupported languages. Splits by line groups.

    Args:
        content: Full file content.
        file_path: Relative file path.
        language: Language identifier.
        max_lines: Maximum lines per chunk.

    Returns:
        List of fallback CodeChunk objects.
    """
    lines = content.splitlines()
    chunks: List[CodeChunk] = []

    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i + max_lines]
        text = "\n".join(chunk_lines)
        if text.strip():
            chunks.append(CodeChunk(
                text=text,
                chunk_type=ChunkType.FALLBACK,
                symbol_name=f"block_{i // max_lines}",
                start_line=i + 1,
                end_line=min(i + max_lines, len(lines)),
                language=language,
                file_path=file_path,
                priority=0.5,
            ))

    return chunks


# ── Public API ──────────────────────────────────────────────────────────────────

# Supported languages with tree-sitter
TREESITTER_LANGUAGES = {"python", "javascript", "typescript", "tsx"}


def parse_file(content: str, language: str, file_path: str) -> List[CodeChunk]:
    """Parse a source file into semantic code chunks.

    Uses Tree-sitter for supported languages, falls back to line-based
    chunking for others. Implements "Iron Stomach" pattern - handles
    all errors gracefully without crashing.

    Args:
        content: Full file content as string.
        language: Language identifier (e.g. 'python', 'javascript').
        file_path: Relative file path for chunk metadata.

    Returns:
        List of CodeChunk objects representing semantic code units.
        Returns empty list only if content is empty or all parsing fails.
    
    Raises:
        Never raises - all errors are caught and logged.
    """
    try:
        if not content.strip():
            return []

        ts_lang = language
        # Normalize language name
        if language == "jsx":
            ts_lang = "javascript"
        elif language == "tsx":
            ts_lang = "tsx"

        lang_obj = _get_language(ts_lang)
        if lang_obj is None:
            return _fallback_chunk(content, file_path, language)

        parser = Parser(lang_obj)
        
        # Handle encoding issues
        try:
            source_bytes = content.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            import logging
            logger = logging.getLogger("kmesh.parser")
            logger.warning(f"Encoding error in {file_path}: {str(e)}")
            return _fallback_chunk(content, file_path, language)

        try:
            tree = parser.parse(source_bytes)
        except (Exception, RecursionError) as e:
            import logging
            logger = logging.getLogger("kmesh.parser")
            logger.warning(f"Tree-sitter parse failed for {file_path}: {str(e)}")
            return _fallback_chunk(content, file_path, language)

        root = tree.root_node

        if ts_lang == "python":
            chunks = _parse_python(root, source_bytes, file_path)
        elif ts_lang in ("javascript", "typescript", "tsx"):
            chunks = _parse_js_ts(root, source_bytes, file_path, language)
        else:
            chunks = _fallback_chunk(content, file_path, language)

        # If AST parsing produced no chunks (e.g. a config file), use fallback
        if not chunks and content.strip():
            chunks = _fallback_chunk(content, file_path, language)

        return chunks
        
    except (MemoryError, RecursionError) as e:
        # Critical errors - file too large or deeply nested
        import logging
        logger = logging.getLogger("kmesh.parser")
        logger.error(f"Critical parser error for {file_path}: {type(e).__name__} - {str(e)}")
        return []
    except Exception as e:
        # Catch-all for any unexpected errors
        import logging
        logger = logging.getLogger("kmesh.parser")
        logger.error(f"Unexpected parser error for {file_path}: {type(e).__name__} - {str(e)}")
        # Try fallback as last resort
        try:
            return _fallback_chunk(content, file_path, language)
        except Exception:
            return []


def get_file_skeleton(content: str, language: str, file_path: str) -> str:
    """Generate a skeleton view of a file (symbols only, no implementation).

    This is the 'get_file_structure' tool output - helps agents understand
    a file without burning tokens on full source.

    Args:
        content: Full file content.
        language: Language identifier.
        file_path: Relative file path.

    Returns:
        String with the file skeleton (signatures and class definitions).
    """
    chunks = parse_file(content, language, file_path)

    skeleton_types = {ChunkType.SIGNATURE, ChunkType.CLASS_DEF, ChunkType.IMPORT, ChunkType.MODULE_DOC}
    skeleton_chunks = [c for c in chunks if c.chunk_type in skeleton_types]

    if not skeleton_chunks:
        return f"# {file_path}\n# (No structural elements detected)"

    lines = [f"# Skeleton: {file_path}"]
    for chunk in sorted(skeleton_chunks, key=lambda c: c.start_line):
        lines.append(f"# L{chunk.start_line}: [{chunk.chunk_type.value}] {chunk.symbol_name}")
        lines.append(chunk.text)
        lines.append("")

    return "\n".join(lines)
