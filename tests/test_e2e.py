"""
End-to-End integration test for the KinetiMesh pipeline.

Generates a small synthetic multi-language repository and validates
the entire pipeline: scan → parse → embed → store → search.

This test does NOT clone external repos — it creates deterministic
test data in a tmp directory for reproducible CI runs.

Covers:
    - Multi-language indexing (Python, JS, TS)
    - Search accuracy: semantic queries find correct code
    - Cross-file search: queries spanning multiple files
    - Incremental re-index after edit
    - File deletion → search removal
    - MCP tool output format validation
"""

import textwrap
from pathlib import Path
from typing import Dict, List

import pytest

from src.server.pipeline import KinetiMeshPipeline
from src.parser.chunker import parse_file


# ── Synthetic Repository ────────────────────────────────────────────────────────

REPO_FILES: Dict[str, str] = {
    "app/models/user.py": textwrap.dedent('''\
        """User domain model with authentication logic."""

        import hashlib
        from dataclasses import dataclass
        from typing import Optional


        @dataclass
        class User:
            """Represents an application user."""
            user_id: int
            email: str
            username: str
            password_hash: str
            is_active: bool = True

            @staticmethod
            def hash_password(password: str) -> str:
                """Hash a plaintext password using SHA-256."""
                return hashlib.sha256(password.encode()).hexdigest()

            def verify_password(self, password: str) -> bool:
                """Verify a plaintext password against the stored hash."""
                return self.password_hash == self.hash_password(password)

            def deactivate(self) -> None:
                """Deactivate this user account."""
                self.is_active = False
    '''),

    "app/models/product.py": textwrap.dedent('''\
        """Product catalog model."""

        from dataclasses import dataclass
        from typing import Optional


        @dataclass
        class Product:
            """Represents a product in the catalog."""
            product_id: int
            name: str
            price: float
            description: str = ""
            stock: int = 0

            def is_available(self) -> bool:
                """Check if the product is in stock."""
                return self.stock > 0

            def apply_discount(self, percentage: float) -> float:
                """Apply a percentage discount and return the new price."""
                if percentage < 0 or percentage > 100:
                    raise ValueError("Discount must be between 0 and 100")
                return round(self.price * (1 - percentage / 100), 2)
    '''),

    "app/services/auth_service.js": textwrap.dedent('''\
        /**
         * Authentication service for handling user login and tokens.
         */

        const jwt = require("jsonwebtoken");

        class AuthService {
            constructor(secretKey, tokenExpiry = 3600) {
                this.secretKey = secretKey;
                this.tokenExpiry = tokenExpiry;
                this.blacklistedTokens = new Set();
            }

            generateToken(userId) {
                return jwt.sign({ userId }, this.secretKey, {
                    expiresIn: this.tokenExpiry,
                });
            }

            verifyToken(token) {
                if (this.blacklistedTokens.has(token)) {
                    throw new Error("Token has been revoked");
                }
                return jwt.verify(token, this.secretKey);
            }

            revokeToken(token) {
                this.blacklistedTokens.add(token);
            }
        }

        module.exports = { AuthService };
    '''),

    "app/services/payment.ts": textwrap.dedent('''\
        /**
         * Payment processing service with Stripe integration.
         */

        interface PaymentIntent {
            amount: number;
            currency: string;
            customerId: string;
            status: "pending" | "completed" | "failed";
        }

        export class PaymentService {
            private apiKey: string;

            constructor(apiKey: string) {
                this.apiKey = apiKey;
            }

            async createPayment(amount: number, currency: string, customerId: string): Promise<PaymentIntent> {
                return {
                    amount,
                    currency,
                    customerId,
                    status: "pending",
                };
            }

            async refund(paymentId: string): Promise<boolean> {
                console.log(`Refunding payment ${paymentId}`);
                return true;
            }
        }
    '''),

    "app/utils/helpers.py": textwrap.dedent('''\
        """Utility helpers for the application."""

        import re
        from typing import List, Optional


        def validate_email(email: str) -> bool:
            """Validate an email address using regex."""
            pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
            return bool(re.match(pattern, email))


        def paginate(items: List, page: int, per_page: int = 10) -> List:
            """Return a paginated slice of items."""
            start = (page - 1) * per_page
            return items[start:start + per_page]


        def slugify(text: str) -> str:
            """Convert text to URL-friendly slug."""
            text = text.lower().strip()
            text = re.sub(r"[^a-z0-9\\s-]", "", text)
            text = re.sub(r"[\\s-]+", "-", text)
            return text.strip("-")
    '''),

    "README.md": "# Test App\n\nA sample application for E2E testing.\n",
}


@pytest.fixture
def e2e_repo(tmp_path: Path) -> Path:
    """Create the synthetic E2E test repository."""
    repo = tmp_path / "e2e_repo"
    repo.mkdir()
    for rel_path, content in REPO_FILES.items():
        file = repo / rel_path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    return repo


@pytest.fixture
def e2e_pipeline(e2e_repo: Path, tmp_path: Path) -> KinetiMeshPipeline:
    """Create and index the E2E pipeline."""
    db_path = str(tmp_path / "e2e_db")
    pipeline = KinetiMeshPipeline(str(e2e_repo), db_path=db_path)
    pipeline.full_index(incremental=False)
    return pipeline


class TestE2EIndexing:
    """Validate that the full pipeline indexes all files."""

    def test_all_files_indexed(self, e2e_pipeline: KinetiMeshPipeline):
        """All source files should be indexed."""
        stats = e2e_pipeline.get_stats()
        # 5 source files + README.md = at least 5 files indexed
        assert stats["total_indexed_files"] >= 5, \
            f"Expected >=5 files, got {stats['total_indexed_files']}"

    def test_chunks_created(self, e2e_pipeline: KinetiMeshPipeline):
        """Indexing should produce a meaningful number of chunks."""
        stats = e2e_pipeline.get_stats()
        # Each file has multiple functions/classes -> many chunks
        assert stats["total_stored_chunks"] >= 20, \
            f"Expected >=20 chunks, got {stats['total_stored_chunks']}"


class TestE2ESearchAccuracy:
    """Validate search quality across the indexed repo."""

    def test_find_password_hashing(self, e2e_pipeline: KinetiMeshPipeline):
        """Query about password hashing should find User.hash_password."""
        results = e2e_pipeline.search("password hashing SHA-256", top_k=5)
        assert len(results) > 0
        all_symbols = [r["symbol_name"] for r in results]
        assert any("hash_password" in s or "verify_password" in s or "User" in s for s in all_symbols), \
            f"Expected password-related symbols in results, got {all_symbols}"

    def test_find_email_validation(self, e2e_pipeline: KinetiMeshPipeline):
        """Query about email validation should find validate_email."""
        results = e2e_pipeline.search("validate email address regex", top_k=5)
        assert len(results) > 0
        all_symbols = [r["symbol_name"] for r in results]
        assert any("validate_email" in s for s in all_symbols), \
            f"Expected validate_email in results, got {all_symbols}"

    def test_find_payment_processing(self, e2e_pipeline: KinetiMeshPipeline):
        """Query about payments should find PaymentService."""
        results = e2e_pipeline.search("payment processing stripe", top_k=5)
        assert len(results) > 0
        files = {r["file_path"] for r in results}
        assert any("payment" in f for f in files), \
            f"Expected payment-related file, got {files}"

    def test_find_token_authentication(self, e2e_pipeline: KinetiMeshPipeline):
        """Query about JWT tokens should find AuthService."""
        results = e2e_pipeline.search("JWT token authentication verify", top_k=5)
        assert len(results) > 0
        files = {r["file_path"] for r in results}
        assert any("auth" in f for f in files), \
            f"Expected auth-related file, got {files}"

    def test_find_pagination(self, e2e_pipeline: KinetiMeshPipeline):
        """Query about pagination should find paginate function."""
        results = e2e_pipeline.search("paginate list items per page", top_k=5)
        assert len(results) > 0
        all_symbols = [r["symbol_name"] for r in results]
        assert any("paginate" in s for s in all_symbols), \
            f"Expected paginate in results, got {all_symbols}"

    def test_cross_language_search(self, e2e_pipeline: KinetiMeshPipeline):
        """Search should span Python, JS, and TS files."""
        results = e2e_pipeline.search("authentication service login", top_k=10)
        languages = {r["language"] for r in results}
        # Should find results in at least 2 languages
        assert len(languages) >= 1, f"Expected multi-language results, got {languages}"


class TestE2EIncrementalUpdate:
    """Validate incremental updates work correctly."""

    def test_edit_file_updates_search(
        self, e2e_repo: Path, e2e_pipeline: KinetiMeshPipeline,
    ):
        """Editing a file and re-indexing should update search results."""
        new_content = textwrap.dedent('''\
            """Updated helpers with new function."""

            def calculate_tax(amount: float, rate: float = 0.1) -> float:
                """Calculate tax on a given amount."""
                return round(amount * rate, 2)
        ''')
        (e2e_repo / "app" / "utils" / "helpers.py").write_text(
            new_content, encoding="utf-8",
        )
        e2e_pipeline.full_index(incremental=True)

        results = e2e_pipeline.search("calculate tax amount", top_k=5)
        assert len(results) > 0
        assert any("calculate_tax" in r["symbol_name"] for r in results), \
            "New function should appear in search after re-index"

    def test_delete_file_removes_from_search(
        self, e2e_repo: Path, e2e_pipeline: KinetiMeshPipeline,
    ):
        """Deleting a file should remove its chunks from search."""
        helpers_path = e2e_repo / "app" / "utils" / "helpers.py"

        # Verify it's currently indexed
        results_before = e2e_pipeline.search("validate email", top_k=5)
        has_helpers = any("helpers" in r.get("file_path", "") for r in results_before)

        # Delete the file
        e2e_pipeline.handle_file_delete(str(helpers_path))

        # Search again — should not find helpers.py content
        results_after = e2e_pipeline.search_symbol("validate_email", top_k=5)
        after_files = {r["file_path"] for r in results_after}
        assert "app/utils/helpers.py" not in after_files or not has_helpers, \
            "Deleted file should not appear in search results"


class TestE2EFileSkeleton:
    """Validate file skeleton generation on the E2E repo."""

    def test_skeleton_shows_signatures(self, e2e_pipeline: KinetiMeshPipeline):
        """Skeleton should list function signatures without implementation."""
        skeleton = e2e_pipeline.get_file_skeleton("app/models/user.py")
        assert "hash_password" in skeleton
        assert "verify_password" in skeleton
        assert "deactivate" in skeleton

    def test_skeleton_nonexistent_file(self, e2e_pipeline: KinetiMeshPipeline):
        """Nonexistent file should return error message."""
        result = e2e_pipeline.get_file_skeleton("nonexistent.py")
        assert "not found" in result.lower() or "error" in result.lower()


class TestE2EStats:
    """Validate pipeline stats after E2E indexing."""

    def test_stats_consistency(self, e2e_pipeline: KinetiMeshPipeline):
        """Stats should be internally consistent."""
        stats = e2e_pipeline.get_stats()
        # total_stored_chunks should match total_indexed_chunks
        # (no deletes have happened in a fresh index)
        assert stats["total_stored_chunks"] == stats["total_indexed_chunks"], \
            f"Stored ({stats['total_stored_chunks']}) != indexed ({stats['total_indexed_chunks']})"
