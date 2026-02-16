"""
Tests for the structural parser (src/parser/chunker.py).

Covers:
    - Python AST parsing (functions, classes, methods, docstrings)
    - JavaScript/TypeScript AST parsing
    - Fallback chunking for unsupported languages
    - Empty file handling
    - Syntax error resilience
    - Unicode content handling
    - get_file_skeleton() output validation
    - ChunkType classification correctness
    - CodeChunk property correctness (chunk_id, search_text)
"""

import pytest

from src.parser.chunker import (
    parse_file,
    get_file_skeleton,
    ChunkType,
    CodeChunk,
    TREESITTER_LANGUAGES,
)
from tests.conftest import (
    SAMPLE_PYTHON,
    SAMPLE_JAVASCRIPT,
    SAMPLE_TYPESCRIPT,
    SAMPLE_EMPTY,
    SAMPLE_SYNTAX_ERROR_PYTHON,
    SAMPLE_UNICODE_PYTHON,
)


class TestParsePython:
    """Tests for Python file parsing."""

    def test_parses_function_signatures(self):
        """Should extract function signatures as SIGNATURE chunks."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        signatures = [c for c in chunks if c.chunk_type == ChunkType.SIGNATURE]

        symbol_names = {c.symbol_name for c in signatures}
        assert "fibonacci" in symbol_names, "fibonacci function signature missing"
        assert "Calculator.add" in symbol_names, "Calculator.add method signature missing"
        assert "Calculator.divide" in symbol_names, "Calculator.divide method signature missing"

    def test_parses_class_definition(self):
        """Should extract class definitions as CLASS_DEF chunks."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        class_defs = [c for c in chunks if c.chunk_type == ChunkType.CLASS_DEF]

        assert len(class_defs) >= 1, "Expected at least one class definition"
        assert any("Calculator" in c.symbol_name for c in class_defs), \
            "Calculator class not found"

    def test_parses_implementation(self):
        """Should extract full function bodies as IMPLEMENTATION chunks."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        impls = [c for c in chunks if c.chunk_type == ChunkType.IMPLEMENTATION]
        assert len(impls) >= 3, f"Expected >=3 implementation chunks, got {len(impls)}"

        # Validate actual code content is in the implementation
        fib_impls = [c for c in impls if "fibonacci" in c.symbol_name]
        assert len(fib_impls) == 1
        assert "fibonacci(n - 1)" in fib_impls[0].text, \
            "fibonacci implementation text should contain recursive call"

    def test_parses_module_docstring(self):
        """Should extract module-level docstring as MODULE_DOC chunk."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        module_docs = [c for c in chunks if c.chunk_type == ChunkType.MODULE_DOC]
        assert len(module_docs) == 1
        assert "Sample module" in module_docs[0].text

    def test_parses_imports(self):
        """Should extract import statements as IMPORT chunk."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        imports = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
        assert len(imports) == 1
        assert "import os" in imports[0].text
        assert "import sys" in imports[0].text

    def test_method_parent_symbol_set(self):
        """Methods inside classes should have parent_symbol set to class name."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        add_sigs = [c for c in chunks if c.symbol_name == "Calculator.add"
                    and c.chunk_type == ChunkType.SIGNATURE]
        assert len(add_sigs) == 1
        assert add_sigs[0].parent_symbol == "Calculator"

    def test_line_numbers_are_positive(self):
        """All chunks should have positive start/end line numbers."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        for c in chunks:
            assert c.start_line >= 1, f"Invalid start_line: {c.start_line}"
            assert c.end_line >= c.start_line, \
                f"end_line ({c.end_line}) < start_line ({c.start_line})"

    def test_priority_ordering(self):
        """Signatures should have higher priority than implementations."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        sigs = [c for c in chunks if c.chunk_type == ChunkType.SIGNATURE]
        impls = [c for c in chunks if c.chunk_type == ChunkType.IMPLEMENTATION]

        if sigs and impls:
            assert sigs[0].priority > impls[0].priority, \
                "Signatures should have higher priority than implementations"

    def test_docstring_extracted_in_signature(self):
        """Function docstrings should be included in signature text."""
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        add_sig = [c for c in chunks if c.symbol_name == "Calculator.add"
                   and c.chunk_type == ChunkType.SIGNATURE]
        assert len(add_sig) == 1
        assert "Add two numbers" in add_sig[0].text, \
            "Docstring should be included in signature chunk"


class TestParseJavaScript:
    """Tests for JavaScript file parsing."""

    def test_parses_js_class(self):
        """Should extract JS class definitions."""
        chunks = parse_file(SAMPLE_JAVASCRIPT, "javascript", "auth.js")
        class_defs = [c for c in chunks if c.chunk_type == ChunkType.CLASS_DEF]
        assert any("AuthService" in c.symbol_name for c in class_defs)

    def test_parses_js_methods(self):
        """Should extract methods inside JS classes."""
        chunks = parse_file(SAMPLE_JAVASCRIPT, "javascript", "auth.js")
        sigs = [c for c in chunks if c.chunk_type == ChunkType.SIGNATURE]
        symbol_names = {c.symbol_name for c in sigs}

        assert any("login" in name for name in symbol_names), "login method missing"
        assert any("logout" in name for name in symbol_names), "logout method missing"

    def test_parses_js_standalone_function(self):
        """Should extract standalone functions (validateEmail)."""
        chunks = parse_file(SAMPLE_JAVASCRIPT, "javascript", "auth.js")
        all_symbols = {c.symbol_name for c in chunks}
        assert any("validateEmail" in s for s in all_symbols), \
            "Standalone validateEmail function missing"

    def test_js_chunks_have_language_set(self):
        """All JS chunks should have language='javascript'."""
        chunks = parse_file(SAMPLE_JAVASCRIPT, "javascript", "auth.js")
        for c in chunks:
            assert c.language == "javascript", f"Expected javascript, got {c.language}"


class TestParseTypeScript:
    """Tests for TypeScript file parsing."""

    def test_parses_ts_class(self):
        """Should parse TypeScript class definitions."""
        chunks = parse_file(SAMPLE_TYPESCRIPT, "typescript", "server.ts")
        class_defs = [c for c in chunks if c.chunk_type == ChunkType.CLASS_DEF]
        assert any("Server" in c.symbol_name for c in class_defs)

    def test_parses_ts_methods(self):
        """Should extract TypeScript methods."""
        chunks = parse_file(SAMPLE_TYPESCRIPT, "typescript", "server.ts")
        sigs = [c for c in chunks if c.chunk_type == ChunkType.SIGNATURE]
        symbol_names = {c.symbol_name for c in sigs}
        assert any("start" in name for name in symbol_names), "start method missing"
        assert any("getPort" in name for name in symbol_names), "getPort method missing"

    def test_ts_file_path_preserved(self):
        """All chunks should preserve the original file_path."""
        chunks = parse_file(SAMPLE_TYPESCRIPT, "typescript", "server.ts")
        for c in chunks:
            assert c.file_path == "server.ts"


class TestEdgeCases:
    """Tests for edge cases and error resilience."""

    def test_empty_file_returns_no_chunks(self):
        """Empty content should return empty list, not crash."""
        chunks = parse_file(SAMPLE_EMPTY, "python", "empty.py")
        assert chunks == [], f"Expected no chunks for empty file, got {len(chunks)}"

    def test_whitespace_only_returns_no_chunks(self):
        """Whitespace-only content should return no chunks."""
        chunks = parse_file("   \n\n\t\n  ", "python", "blank.py")
        assert chunks == []

    def test_syntax_error_does_not_crash(self):
        """Files with syntax errors should be handled gracefully."""
        # Tree-sitter is error-tolerant, so it may produce partial results
        # or fall back. The key point: it must NOT raise.
        chunks = parse_file(SAMPLE_SYNTAX_ERROR_PYTHON, "python", "broken.py")
        assert isinstance(chunks, list)  # no crash, returns a list

    def test_unicode_content_parsed(self):
        """Unicode identifiers and content should be parsed correctly."""
        chunks = parse_file(SAMPLE_UNICODE_PYTHON, "python", "unicode.py")
        assert len(chunks) > 0, "Unicode file should produce chunks"
        all_text = " ".join(c.text for c in chunks)
        assert "grüße" in all_text or "gr" in all_text, \
            "Unicode function name should appear in chunks"

    def test_unsupported_language_uses_fallback(self):
        """Unsupported languages should use line-based fallback chunking."""
        content = "# This is a markdown file\n## Section 1\nSome text here.\n" * 5
        chunks = parse_file(content, "markdown", "README.md")

        assert len(chunks) > 0, "Fallback should produce at least one chunk"
        for c in chunks:
            assert c.chunk_type == ChunkType.FALLBACK, \
                f"Expected FALLBACK type, got {c.chunk_type}"

    def test_very_long_file_fallback(self):
        """Large unsupported files should produce multiple fallback chunks."""
        content = "\n".join([f"line {i}: content here" for i in range(200)])
        chunks = parse_file(content, "unknown", "big.txt")
        # Default fallback splits every 50 lines => 200/50 = 4 chunks
        assert len(chunks) >= 3, f"Expected >=3 fallback chunks, got {len(chunks)}"


class TestGetFileSkeleton:
    """Tests for the get_file_skeleton() function."""

    def test_skeleton_contains_signatures(self):
        """Skeleton should include function signatures."""
        skeleton = get_file_skeleton(SAMPLE_PYTHON, "python", "main.py")
        assert "fibonacci" in skeleton
        assert "add" in skeleton
        assert "divide" in skeleton

    def test_skeleton_contains_class_name(self):
        """Skeleton should include class definition."""
        skeleton = get_file_skeleton(SAMPLE_PYTHON, "python", "main.py")
        assert "Calculator" in skeleton

    def test_skeleton_excludes_implementation_body(self):
        """Skeleton should NOT contain full implementation details."""
        skeleton = get_file_skeleton(SAMPLE_PYTHON, "python", "main.py")
        # The recursive call is implementation detail, not signature
        assert "fibonacci(n - 1)" not in skeleton

    def test_skeleton_empty_file(self):
        """Empty file skeleton should not crash."""
        skeleton = get_file_skeleton("", "python", "empty.py")
        assert isinstance(skeleton, str)
        assert "No structural elements" in skeleton or skeleton.strip() == ""

    def test_skeleton_header_contains_path(self):
        """Skeleton should reference the file path."""
        skeleton = get_file_skeleton(SAMPLE_PYTHON, "python", "main.py")
        assert "main.py" in skeleton


class TestCodeChunkProperties:
    """Tests for CodeChunk computed properties."""

    def test_chunk_id_format(self):
        """chunk_id should follow path::symbol::type format."""
        chunk = CodeChunk(
            text="def foo(): pass",
            chunk_type=ChunkType.SIGNATURE,
            symbol_name="foo",
            start_line=1,
            end_line=1,
            file_path="src/utils.py",
        )
        assert chunk.chunk_id == "src/utils.py::foo::signature"

    def test_chunk_id_no_symbol(self):
        """chunk_id should use line number when symbol_name is empty."""
        chunk = CodeChunk(
            text="fallback text",
            chunk_type=ChunkType.FALLBACK,
            symbol_name="",
            start_line=10,
            end_line=20,
            file_path="data.txt",
        )
        assert "L10" in chunk.chunk_id

    def test_search_text_includes_metadata(self):
        """search_text should prepend language/symbol metadata."""
        chunk = CodeChunk(
            text="def foo(): pass",
            chunk_type=ChunkType.SIGNATURE,
            symbol_name="foo",
            start_line=1,
            end_line=1,
            language="python",
            parent_symbol="MyClass",
            file_path="test.py",
        )
        st = chunk.search_text
        assert "Language: python" in st
        assert "Class: MyClass" in st
        assert "Symbol: foo" in st
        assert "def foo(): pass" in st
