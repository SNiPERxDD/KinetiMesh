"""
KinetiMesh Ingestor - High-Performance Incremental File Scanner.

Refactored from repo_dumper.py to yield CodeFile objects instead of XML.
Uses os.scandir for sub-millisecond directory traversal, SHA-256 hashing
for change detection, and generator patterns for streaming processing.

Key Design:
    - File-Level Atomicity: If hash changes, re-index entire file.
    - Generator Pattern: Yields files one-by-one for pipeline processing.
    - In-Memory State Map: Tracks {path: hash} for instant diff detection.
"""

import os
import hashlib
import subprocess
import fnmatch
import time
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Dict, List, Set, Optional, Tuple


# ── Language Detection Map ──────────────────────────────────────────────────────
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".md": "markdown",
    ".sql": "sql",
    ".dockerfile": "dockerfile",
    ".tf": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".vue": "vue",
    ".svelte": "svelte",
}

# ── Pruning Configuration ───────────────────────────────────────────────────────
PRUNE_DIRS: Set[str] = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".kiro", ".build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt", "dist", "build", "coverage",
    "venv", ".venv", "env", "site-packages",
    "Pods", "DerivedData", ".gradle", "target",
    ".kmesh", "deleted", "archive",
    ".app", "migrations", "dumps",
    ".eggs", ".tox", "htmlcov",
}

PRUNE_FILENAMES: Set[str] = {
    ".DS_Store", "Thumbs.db",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Gemfile.lock", "composer.lock", "cargo.lock",
    ".env", ".env.local", ".env.development", ".env.production",
    "credentials.json", "service_account.json", "secrets.yaml",
    "id_rsa", "id_dsa", "id_ed25519",
}

SKIP_EXTS: Set[str] = {
    ".zip", ".gz", ".bz2", ".xz", ".tar", ".rar", ".7z",
    ".mp4", ".mp3", ".mov", ".wav", ".avi", ".wmv", ".webm",
    ".so", ".dll", ".dylib", ".node", ".wasm", ".class", ".o", ".a",
    ".exe", ".bin", ".pack", ".db", ".sqlite", ".pkl", ".onnx",
    ".plist", ".eot", ".ttf", ".woff", ".woff2",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".min.js", ".min.css", ".map",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".lock", ".pem", ".key", ".icns",
}

MAX_FILE_BYTES: int = 2_000_000  # 2MB per-file limit


@dataclass(frozen=True, slots=True)
class CodeFile:
    """Atomic unit of the KinetiMesh pipeline.

    Represents a single source file with its content and metadata,
    ready for downstream parsing and embedding.
    """

    rel_path: str
    abs_path: str
    content: str
    file_hash: str
    language: str
    size_bytes: int
    line_count: int

    @property
    def is_binary(self) -> bool:
        """Detect binary content by checking for null bytes in first 1KB."""
        try:
            return b"\x00" in self.content.encode("utf-8")[:1024]
        except (UnicodeDecodeError, AttributeError):
            return True


class MeshIngestor:
    """High-performance incremental repository scanner.

    Uses os.scandir for rapid traversal and maintains an in-memory
    hash map for instant change detection. Files are yielded as
    CodeFile objects for streaming pipeline consumption.

    Args:
        root_path: Absolute path to the repository root.
        extra_prune_dirs: Additional directories to skip.
        extra_prune_files: Additional filenames to skip.
    """

    def __init__(
        self,
        root_path: str,
        extra_prune_dirs: Optional[Set[str]] = None,
        extra_prune_files: Optional[Set[str]] = None,
    ):
        self.root = Path(root_path).resolve()
        self.prune_dirs = PRUNE_DIRS | (extra_prune_dirs or set())
        self.prune_filenames = PRUNE_FILENAMES | (extra_prune_files or set())
        self.skip_exts = SKIP_EXTS

        # In-memory state map for incremental detection
        self.file_state: Dict[str, str] = {}

        # Git-ignored files cache
        self._git_ignored: Optional[Set[str]] = None

        # Performance metrics
        self.last_scan_time: float = 0.0
        self.last_file_count: int = 0
        self.last_changed_count: int = 0

    @property
    def git_ignored(self) -> Set[str]:
        """Lazily load and cache gitignored file paths."""
        if self._git_ignored is None:
            self._git_ignored = self._get_git_ignored()
        return self._git_ignored

    def _get_git_ignored(self) -> Set[str]:
        """Return set of gitignored file paths relative to repo root."""
        ignored: Set[str] = set()
        
        # Check if this is a git repository first to avoid subprocess hang
        if not (self.root / ".git").exists():
            return ignored
        
        try:
            out = subprocess.check_output(
                [
                    "git", "-C", str(self.root), "ls-files",
                    "--others", "--ignored", "--exclude-standard", "-z",
                ],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            for p in out.decode("utf-8", errors="ignore").split("\0"):
                if p:
                    ignored.add(p)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            gitignore = self.root / ".gitignore"
            if gitignore.exists():
                try:
                    patterns = []
                    with open(gitignore, "r", encoding="utf-8", errors="ignore") as f:
                        patterns = [
                            line.strip()
                            for line in f
                            if line.strip() and not line.startswith("#")
                        ]
                    for dirpath, _, filenames in os.walk(self.root):
                        for fname in filenames:
                            rel = os.path.relpath(
                                os.path.join(dirpath, fname), self.root
                            )
                            for pat in patterns:
                                if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(
                                    fname, pat
                                ):
                                    ignored.add(rel)
                                    break
                except OSError:
                    pass
        return ignored

    @staticmethod
    def _calculate_hash(content: str) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _detect_language(file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        # Handle special cases like Dockerfile
        if file_path.name.lower() == "dockerfile":
            return "dockerfile"
        if file_path.name.lower() == "makefile":
            return "makefile"
        return EXTENSION_TO_LANGUAGE.get(suffix, suffix.lstrip(".") or "text")

    def _should_skip_dir(self, dirname: str) -> bool:
        """Check if a directory should be pruned."""
        return dirname in self.prune_dirs or dirname.startswith(".")

    def _should_skip_file(self, entry: os.DirEntry, rel_path: str) -> bool:
        """Check if file should be skipped based on name, extension, size, gitignore."""
        if entry.name in self.prune_filenames:
            return True

        ext = Path(entry.name).suffix.lower()
        if ext in self.skip_exts:
            return True

        try:
            if entry.stat().st_size > MAX_FILE_BYTES:
                return True
        except OSError:
            return True

        if rel_path in self.git_ignored:
            return True

        return False

    def _recursive_collect(
        self, current_dir: Path, candidates: List[Tuple[Path, str]]
    ) -> None:
        """Iteratively collect file candidates using stack-based traversal.
        
        Prevents RecursionError on deep directory structures.

        Args:
            current_dir: Directory to scan.
            candidates: Accumulator list for (abs_path, rel_path) tuples.
        """
        # Use explicit stack instead of recursion to prevent stack overflow
        dirs_to_scan = [current_dir]
        
        while dirs_to_scan:
            scan_dir = dirs_to_scan.pop()
            try:
                with os.scandir(scan_dir) as scanner:
                    for entry in scanner:
                        if entry.is_dir(follow_symlinks=False):
                            if not self._should_skip_dir(entry.name):
                                dirs_to_scan.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            fpath = Path(entry.path)
                            rel_path = str(fpath.relative_to(self.root))
                            if not self._should_skip_file(entry, rel_path):
                                candidates.append((fpath, rel_path))
            except (PermissionError, FileNotFoundError, OSError):
                pass

    def scan(self, incremental: bool = True) -> Generator[CodeFile, None, None]:
        """Scan the repository and yield changed CodeFile objects.

        Uses generator pattern to enable streaming pipeline processing.
        When incremental=True, only yields files whose hash has changed
        since the last scan.

        Args:
            incremental: If True, skip unchanged files. If False, yield all.

        Yields:
            CodeFile objects for each (changed) source file.
        """
        start_time = time.perf_counter()
        candidates: List[Tuple[Path, str]] = []

        # Step 1: Rapid recursive collection via scandir
        self._recursive_collect(self.root, candidates)

        # Track files that still exist for state cleanup
        seen_paths: Set[str] = set()
        changed_count = 0

        # Step 2: Read content and yield CodeFile objects
        for fpath, rel_path in candidates:
            seen_paths.add(rel_path)
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                f_hash = self._calculate_hash(content)

                # Incremental: skip if unchanged
                if incremental and self.file_state.get(rel_path) == f_hash:
                    continue

                self.file_state[rel_path] = f_hash
                changed_count += 1

                line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

                yield CodeFile(
                    rel_path=rel_path,
                    abs_path=str(fpath),
                    content=content,
                    file_hash=f_hash,
                    language=self._detect_language(fpath),
                    size_bytes=len(content.encode("utf-8")),
                    line_count=line_count,
                )
            except (OSError, FileNotFoundError) as e:
                # File disappeared during scan or permission denied
                import logging
                logger = logging.getLogger("kmesh.ingestor")
                logger.debug(f"Skipping {rel_path}: {type(e).__name__}")
                continue
            except (UnicodeDecodeError, UnicodeError) as e:
                # Encoding issues - skip file
                import logging
                logger = logging.getLogger("kmesh.ingestor")
                logger.warning(f"Encoding error in {rel_path}: {str(e)[:100]}")
                continue
            except MemoryError:
                # File too large to read into memory
                import logging
                logger = logging.getLogger("kmesh.ingestor")
                logger.error(f"MemoryError reading {rel_path} - file too large")
                continue
            except Exception as e:
                # Catch-all for unexpected errors
                import logging
                logger = logging.getLogger("kmesh.ingestor")
                logger.error(f"Unexpected error reading {rel_path}: {type(e).__name__} - {str(e)}")
                continue

        # Step 3: Clean up state for deleted files
        deleted = set(self.file_state.keys()) - seen_paths
        for d in deleted:
            del self.file_state[d]

        elapsed = time.perf_counter() - start_time
        self.last_scan_time = elapsed
        self.last_file_count = len(candidates)
        self.last_changed_count = changed_count

    def scan_single_file(self, file_path: str) -> Optional[CodeFile]:
        """Scan a single file and return its CodeFile if changed.

        Used by the file watcher for instant single-file re-indexing.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            CodeFile if the file changed, None otherwise.
        """
        fpath = Path(file_path).resolve()
        if not fpath.is_file():
            return None

        try:
            rel_path = str(fpath.relative_to(self.root))
        except ValueError:
            return None

        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            f_hash = self._calculate_hash(content)

            if self.file_state.get(rel_path) == f_hash:
                return None

            self.file_state[rel_path] = f_hash
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            return CodeFile(
                rel_path=rel_path,
                abs_path=str(fpath),
                content=content,
                file_hash=f_hash,
                language=self._detect_language(fpath),
                size_bytes=len(content.encode("utf-8")),
                line_count=line_count,
            )
        except (OSError, FileNotFoundError) as e:
            import logging
            logger = logging.getLogger("kmesh.ingestor")
            logger.debug(f"File read error {rel_path}: {type(e).__name__}")
            return None
        except (UnicodeDecodeError, UnicodeError) as e:
            import logging
            logger = logging.getLogger("kmesh.ingestor")
            logger.warning(f"Encoding error in {rel_path}: {str(e)[:100]}")
            return None
        except MemoryError:
            import logging
            logger = logging.getLogger("kmesh.ingestor")
            logger.error(f"MemoryError reading {rel_path}")
            return None
        except Exception as e:
            import logging
            logger = logging.getLogger("kmesh.ingestor")
            logger.error(f"Unexpected error reading {rel_path}: {type(e).__name__}")
            return None

    def get_deleted_paths(self) -> List[str]:
        """Return list of paths that existed in state but no longer on disk.

        Useful for cleaning up vectors of deleted files.
        """
        deleted = []
        for rel_path in list(self.file_state.keys()):
            full = self.root / rel_path
            if not full.exists():
                deleted.append(rel_path)
                del self.file_state[rel_path]
        return deleted

    def get_stats(self) -> Dict[str, float]:
        """Return performance statistics from the last scan."""
        return {
            "scan_time_ms": self.last_scan_time * 1000,
            "total_files": self.last_file_count,
            "changed_files": self.last_changed_count,
            "tracked_files": len(self.file_state),
        }
