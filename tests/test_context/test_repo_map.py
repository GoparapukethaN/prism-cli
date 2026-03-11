"""Tests for prism.context.repo_map — repository map generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.context.repo_map import (
    _load_gitignore_patterns,
    _parse_python_file,
    _should_ignore,
    generate_repo_map,
    invalidate_cache,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Invalidate the module-level cache before each test."""
    invalidate_cache()


# ---------------------------------------------------------------------------
# _parse_python_file
# ---------------------------------------------------------------------------


class TestParsePythonFile:
    def test_detects_function(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("def greet(name: str) -> str:\n    return name\n")
        sigs = _parse_python_file(f)
        assert any("greet" in s for s in sigs)

    def test_detects_class(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("class Foo:\n    pass\n")
        sigs = _parse_python_file(f)
        assert any("Foo" in s for s in sigs)

    def test_detects_class_with_bases(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("class Bar(Foo, Baz):\n    pass\n")
        sigs = _parse_python_file(f)
        joined = " ".join(sigs)
        assert "Bar" in joined
        assert "Foo" in joined

    def test_detects_methods(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text(
            "class MyClass:\n"
            "    def __init__(self, x: int) -> None:\n"
            "        self.x = x\n\n"
            "    def get_x(self) -> int:\n"
            "        return self.x\n"
        )
        sigs = _parse_python_file(f)
        assert any("__init__" in s for s in sigs)
        assert any("get_x" in s for s in sigs)

    def test_detects_return_type(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
        sigs = _parse_python_file(f)
        joined = " ".join(sigs)
        assert "-> int" in joined

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        sigs = _parse_python_file(f)
        assert sigs == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nope.py"
        sigs = _parse_python_file(f)
        assert sigs == []


# ---------------------------------------------------------------------------
# .gitignore / should_ignore
# ---------------------------------------------------------------------------


class TestShouldIgnore:
    def test_ignores_pycache(self, tmp_path: Path) -> None:
        patterns = ["__pycache__"]
        path = tmp_path / "src" / "__pycache__" / "mod.pyc"
        assert _should_ignore(path, tmp_path, patterns) is True

    def test_ignores_glob_pattern(self, tmp_path: Path) -> None:
        patterns = ["*.pyc"]
        path = tmp_path / "src" / "main.pyc"
        assert _should_ignore(path, tmp_path, patterns) is True

    def test_does_not_ignore_normal_file(self, tmp_path: Path) -> None:
        patterns = ["*.pyc", "__pycache__"]
        path = tmp_path / "src" / "main.py"
        assert _should_ignore(path, tmp_path, patterns) is False

    def test_ignores_specific_file(self, tmp_path: Path) -> None:
        patterns = ["secret_config.py"]
        path = tmp_path / "src" / "secret_config.py"
        assert _should_ignore(path, tmp_path, patterns) is True

    def test_load_gitignore_includes_defaults(self, tmp_path: Path) -> None:
        patterns = _load_gitignore_patterns(tmp_path)
        assert "__pycache__" in patterns

    def test_load_gitignore_merges_file(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("my_custom_dir/\n# comment\n\n")
        patterns = _load_gitignore_patterns(tmp_path)
        assert "my_custom_dir/" in patterns
        # Comments should NOT be included
        assert "# comment" not in patterns


# ---------------------------------------------------------------------------
# generate_repo_map
# ---------------------------------------------------------------------------


class TestGenerateRepoMap:
    def test_basic_generation(self, python_project: Path) -> None:
        result = generate_repo_map(python_project, max_tokens=5000, use_cache=False)
        assert "main" in result
        assert "helper" in result

    def test_includes_classes(self, python_project: Path) -> None:
        result = generate_repo_map(python_project, max_tokens=5000, use_cache=False)
        assert "User" in result
        assert "Admin" in result

    def test_includes_methods(self, python_project: Path) -> None:
        result = generate_repo_map(python_project, max_tokens=5000, use_cache=False)
        assert "display" in result

    def test_respects_gitignore(self, python_project: Path) -> None:
        result = generate_repo_map(python_project, max_tokens=5000, use_cache=False)
        # secret_config.py is in .gitignore
        assert "SECRET_KEY" not in result

    def test_respects_max_tokens(self, python_project: Path) -> None:
        small = generate_repo_map(python_project, max_tokens=50, use_cache=False)
        large = generate_repo_map(python_project, max_tokens=50000, use_cache=False)
        assert len(small) <= len(large)

    def test_caching(self, python_project: Path) -> None:
        result1 = generate_repo_map(python_project, max_tokens=5000, use_cache=True)
        result2 = generate_repo_map(python_project, max_tokens=5000, use_cache=True)
        assert result1 == result2

    def test_cache_invalidated_on_change(self, python_project: Path) -> None:
        generate_repo_map(python_project, max_tokens=5000, use_cache=True)
        # Modify a file
        (python_project / "src" / "new_module.py").write_text(
            "def brand_new_function() -> None:\n    pass\n"
        )
        result2 = generate_repo_map(python_project, max_tokens=5000, use_cache=True)
        assert "brand_new_function" in result2

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = generate_repo_map(tmp_path, max_tokens=5000, use_cache=False)
        assert result == ""
