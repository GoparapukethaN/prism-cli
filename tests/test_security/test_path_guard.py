"""Tests for prism.security.path_guard.PathGuard."""

from __future__ import annotations

from pathlib import Path

import pytest

from prism.exceptions import ExcludedFileError, PathTraversalError
from prism.security.path_guard import PathGuard

# =====================================================================
# Path traversal
# =====================================================================


class TestPathTraversal:
    """Verify that paths containing ``../`` cannot escape the project root."""

    def test_basic_dotdot_traversal(self, path_guard: PathGuard) -> None:
        with pytest.raises(PathTraversalError):
            path_guard.validate("../../../etc/passwd")

    def test_dotdot_in_middle(self, path_guard: PathGuard) -> None:
        with pytest.raises(PathTraversalError):
            path_guard.validate("src/../../etc/shadow")

    def test_absolute_path_outside_root(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        with pytest.raises(PathTraversalError):
            path_guard.validate("/etc/passwd")

    def test_dotdot_equals_root_parent(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        """Going up to the parent of root and back in should still fail
        if the final resolved path is outside root."""
        outside = project_root.parent / "nope.txt"
        with pytest.raises(PathTraversalError):
            path_guard.validate(str(outside))

    def test_dotdot_resolving_back_inside_root(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        """src/../src/main.py resolves inside root — should succeed."""
        resolved = path_guard.validate("src/../src/main.py")
        assert resolved == (project_root / "src" / "main.py").resolve()


# =====================================================================
# Symlink escape
# =====================================================================


class TestSymlinkEscape:
    """Verify that symlinks pointing outside the root are rejected."""

    def test_symlink_pointing_outside_root(
        self, path_guard: PathGuard, project_root: Path, tmp_path: Path
    ) -> None:
        # Create a file truly outside the project root (use /tmp directly)
        import tempfile

        outside_dir = Path(tempfile.mkdtemp())
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("top secret\n")

        # Create a symlink inside the project that points outside
        symlink_path = project_root / "escape_link"
        symlink_path.symlink_to(outside_file)

        with pytest.raises(PathTraversalError):
            path_guard.validate("escape_link")

    def test_symlink_within_root_is_ok(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        # Symlink that stays within the project
        target = project_root / "src" / "main.py"
        link = project_root / "link_to_main"
        link.symlink_to(target)

        resolved = path_guard.validate("link_to_main")
        assert resolved == target.resolve()


# =====================================================================
# Null byte injection
# =====================================================================


class TestNullByteInjection:
    """Null bytes in paths must be rejected before reaching the OS."""

    def test_null_byte_raises_value_error(self, path_guard: PathGuard) -> None:
        with pytest.raises(ValueError, match="null byte"):
            path_guard.validate("src/main.py\x00.jpg")

    def test_null_byte_at_start(self, path_guard: PathGuard) -> None:
        with pytest.raises(ValueError, match="null byte"):
            path_guard.validate("\x00etc/passwd")

    def test_null_byte_at_end(self, path_guard: PathGuard) -> None:
        with pytest.raises(ValueError, match="null byte"):
            path_guard.validate("src/main.py\x00")


# =====================================================================
# Excluded patterns (user-configured)
# =====================================================================


class TestExcludedPatterns:
    """Verify user-configured excluded patterns are enforced."""

    def test_dotenv_excluded(self, path_guard: PathGuard, project_root: Path) -> None:
        # The .env file exists in the fixture
        with pytest.raises(ExcludedFileError, match=r"\.env"):
            path_guard.validate(".env")

    def test_pem_file_excluded(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        pem = project_root / "certs" / "server.pem"
        pem.parent.mkdir(parents=True, exist_ok=True)
        pem.write_text("-----BEGIN CERTIFICATE-----\n")

        with pytest.raises(ExcludedFileError, match=r"\*\*/\*\.pem"):
            path_guard.validate("certs/server.pem")

    def test_key_file_excluded(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        key = project_root / "private.key"
        key.write_text("-----BEGIN RSA PRIVATE KEY-----\n")

        with pytest.raises(ExcludedFileError):
            path_guard.validate("private.key")

    def test_credentials_json_excluded(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        cred = project_root / "config" / "credentials.json"
        cred.parent.mkdir(parents=True, exist_ok=True)
        cred.write_text("{}\n")

        with pytest.raises(ExcludedFileError):
            path_guard.validate("config/credentials.json")


# =====================================================================
# Always-blocked patterns
# =====================================================================


class TestAlwaysBlockedPatterns:
    """Verify that always-blocked patterns (from defaults) cannot be bypassed."""

    def test_ssh_private_key_blocked(
        self, project_root: Path
    ) -> None:
        guard = PathGuard(project_root=project_root, excluded_patterns=[])
        ssh_dir = project_root / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_rsa").write_text("private key\n")

        with pytest.raises(ExcludedFileError):
            guard.validate(".ssh/id_rsa")

    def test_aws_credentials_blocked(
        self, project_root: Path
    ) -> None:
        guard = PathGuard(project_root=project_root, excluded_patterns=[])
        aws_dir = project_root / ".aws"
        aws_dir.mkdir()
        (aws_dir / "credentials").write_text("[default]\n")

        with pytest.raises(ExcludedFileError):
            guard.validate(".aws/credentials")

    def test_gnupg_dir_blocked(
        self, project_root: Path
    ) -> None:
        guard = PathGuard(project_root=project_root, excluded_patterns=[])
        gpg_dir = project_root / ".gnupg"
        gpg_dir.mkdir()
        (gpg_dir / "secring.gpg").write_text("ring\n")

        with pytest.raises(ExcludedFileError):
            guard.validate(".gnupg/secring.gpg")


# =====================================================================
# Valid paths
# =====================================================================


class TestValidPaths:
    """Ensure legitimate paths are accepted."""

    def test_relative_path_in_root(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        resolved = path_guard.validate("src/main.py")
        assert resolved == (project_root / "src" / "main.py").resolve()

    def test_absolute_path_in_root(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        abs_path = project_root / "data" / "input.txt"
        resolved = path_guard.validate(str(abs_path))
        assert resolved == abs_path.resolve()

    def test_root_itself(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        resolved = path_guard.validate(".")
        assert resolved == project_root.resolve()

    def test_is_safe_returns_true(
        self, path_guard: PathGuard
    ) -> None:
        assert path_guard.is_safe("src/main.py") is True

    def test_is_safe_returns_false_for_traversal(
        self, path_guard: PathGuard
    ) -> None:
        assert path_guard.is_safe("../../../etc/passwd") is False

    def test_is_safe_returns_false_for_excluded(
        self, path_guard: PathGuard
    ) -> None:
        assert path_guard.is_safe(".env") is False

    def test_is_safe_returns_false_for_null_byte(
        self, path_guard: PathGuard
    ) -> None:
        assert path_guard.is_safe("src\x00/main.py") is False

    def test_nonexistent_file_inside_root_still_validates(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        """Even if the file doesn't exist, the path is valid if within root."""
        resolved = path_guard.validate("does_not_exist.txt")
        assert resolved == (project_root / "does_not_exist.txt").resolve()

    def test_deeply_nested_path(
        self, path_guard: PathGuard, project_root: Path
    ) -> None:
        deep = project_root / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "file.txt").write_text("deep\n")
        resolved = path_guard.validate("a/b/c/d/file.txt")
        assert resolved == (deep / "file.txt").resolve()
