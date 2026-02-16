"""
Shared pytest fixtures for the KinetiMesh test suite.

Provides isolated temporary directories, sample code files, and
pre-configured pipeline instances for deterministic testing.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import pytest


# ── Sample Code Fixtures ────────────────────────────────────────────────────────

SAMPLE_PYTHON = '''\
"""Sample module for testing."""

import os
import sys


class Calculator:
    """A simple calculator class."""

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

    def divide(self, a: float, b: float) -> float:
        """Divide a by b with error handling."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return round(a / b, self.precision)


def fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''

SAMPLE_JAVASCRIPT = '''\
/**
 * User authentication module.
 */

class AuthService {
    constructor(secretKey) {
        this.secretKey = secretKey;
        this.sessions = new Map();
    }

    login(username, password) {
        if (!username || !password) {
            throw new Error("Missing credentials");
        }
        const token = this._generateToken(username);
        this.sessions.set(token, { username, loginTime: Date.now() });
        return token;
    }

    logout(token) {
        return this.sessions.delete(token);
    }

    _generateToken(username) {
        return `${username}_${Date.now()}_${Math.random().toString(36)}`;
    }
}

function validateEmail(email) {
    const re = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return re.test(email);
}

module.exports = { AuthService, validateEmail };
'''

SAMPLE_TYPESCRIPT = '''\
interface Config {
    host: string;
    port: number;
    debug?: boolean;
}

export class Server {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    start(): void {
        console.log(`Listening on ${this.config.host}:${this.config.port}`);
    }

    getPort(): number {
        return this.config.port;
    }
}
'''

SAMPLE_EMPTY = ""

SAMPLE_BINARY_CONTENT = "Binary\x00content\x00with\x00nulls"

SAMPLE_SYNTAX_ERROR_PYTHON = '''\
def broken_function(
    # Missing closing paren and colon
    x, y
    return x + y
'''

SAMPLE_UNICODE_PYTHON = '''\
"""Unicode handling test: 日本語コメント."""

def grüße(name: str) -> str:
    """Return a greeting with special characters: café, naïve, résumé."""
    return f"Héllo, {name}! 🎉"

EMOJI_MAP = {"smile": "😊", "rocket": "🚀"}
'''


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal temporary repository with sample source files.

    Structure:
        tmp_repo/
            main.py          (Python with class + functions)
            src/
                auth.js       (JavaScript with class)
                server.ts     (TypeScript)
            docs/
                notes.md      (Markdown - unsupported for AST)
            empty.py          (Empty file)
    """
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Python file
    (repo / "main.py").write_text(SAMPLE_PYTHON, encoding="utf-8")

    # JavaScript in src/
    src_dir = repo / "src"
    src_dir.mkdir()
    (src_dir / "auth.js").write_text(SAMPLE_JAVASCRIPT, encoding="utf-8")

    # TypeScript
    (src_dir / "server.ts").write_text(SAMPLE_TYPESCRIPT, encoding="utf-8")

    # Markdown doc
    docs_dir = repo / "docs"
    docs_dir.mkdir()
    (docs_dir / "notes.md").write_text("# Notes\n\nSome documentation.", encoding="utf-8")

    # Empty file
    (repo / "empty.py").write_text(SAMPLE_EMPTY, encoding="utf-8")

    return repo


@pytest.fixture
def tmp_repo_extended(tmp_repo: Path) -> Path:
    """Extended repo with edge-case files (unicode, syntax errors, binary-like).

    Adds to tmp_repo:
        unicode.py        (Unicode identifiers & emoji)
        broken.py         (Syntax error)
        data.bin          (Binary-like content - should be skipped by ingestor)
    """
    (tmp_repo / "unicode.py").write_text(SAMPLE_UNICODE_PYTHON, encoding="utf-8")
    (tmp_repo / "broken.py").write_text(SAMPLE_SYNTAX_ERROR_PYTHON, encoding="utf-8")
    (tmp_repo / "data.bin").write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    return tmp_repo


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Return a temporary path for LanceDB storage."""
    db_dir = tmp_path / "test_db"
    db_dir.mkdir()
    return str(db_dir)


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Return an empty directory."""
    d = tmp_path / "empty_dir"
    d.mkdir()
    return d


@pytest.fixture
def nested_repo(tmp_path: Path) -> Path:
    """Create a deeply nested repo to test directory traversal.

    Structure:
        nested_repo/
            a/b/c/d/deep.py
            .git/            (should be pruned)
            node_modules/    (should be pruned)
            __pycache__/     (should be pruned)
            valid.py
    """
    repo = tmp_path / "nested_repo"
    repo.mkdir()

    # Deep nesting
    deep = repo / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("def deep_func(): pass\n", encoding="utf-8")

    # Prunable directories with files (should NOT be scanned)
    for pruned in [".git", "node_modules", "__pycache__"]:
        d = repo / pruned
        d.mkdir()
        (d / "should_skip.py").write_text("# should not appear\n", encoding="utf-8")

    # Valid top-level file
    (repo / "valid.py").write_text("def top_func(): return 42\n", encoding="utf-8")

    return repo
