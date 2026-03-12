"""Tests for the smart prompt enhancer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.cli.prompt_enhancer import EnhancedPrompt, PromptEnhancer

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project tree for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        'def main():\n    print("hello")\n', encoding="utf-8"
    )
    (tmp_path / "src" / "utils.py").write_text(
        'def add(a, b):\n    return a + b\n', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test"\n', encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def enhancer(project_root: Path) -> PromptEnhancer:
    """Create a PromptEnhancer against the temp project."""
    return PromptEnhancer(project_root)


@pytest.fixture
def enhancer_with_files(project_root: Path) -> PromptEnhancer:
    """Create a PromptEnhancer with active files set."""
    return PromptEnhancer(project_root, active_files=["src/main.py", "src/utils.py"])


# ---------------------------------------------------------------------------
# EnhancedPrompt dataclass
# ---------------------------------------------------------------------------


class TestEnhancedPrompt:
    """Tests for the EnhancedPrompt dataclass."""

    def test_defaults(self) -> None:
        ep = EnhancedPrompt(original="hi", enhanced="hi")
        assert ep.original == "hi"
        assert ep.enhanced == "hi"
        assert ep.context_added == []
        assert ep.strategy == "question"

    def test_custom_values(self) -> None:
        ep = EnhancedPrompt(
            original="fix bug",
            enhanced="fix bug\n[context]",
            context_added=["Added debugging instructions"],
            strategy="debug",
        )
        assert ep.strategy == "debug"
        assert len(ep.context_added) == 1


# ---------------------------------------------------------------------------
# Strategy detection
# ---------------------------------------------------------------------------


class TestDetectStrategy:
    """Tests for PromptEnhancer._detect_strategy."""

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("fix the error in main.py", "debug"),
            ("There is a bug in the router", "debug"),
            ("My app crashes on startup", "debug"),
            ("I see a traceback when I run tests", "debug"),
            ("The build is broken", "debug"),
            ("debug the login flow", "debug"),
            ("This exception keeps happening", "debug"),
        ],
    )
    def test_debug_strategy(
        self, enhancer: PromptEnhancer, prompt: str, expected: str
    ) -> None:
        assert enhancer._detect_strategy(prompt) == expected

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("create a new API endpoint", "create"),
            ("make a test file for utils", "create"),
            ("build a REST server", "create"),
            ("generate a migration script", "create"),
            ("scaffold a React component", "create"),
            ("I need a new file for auth", "create"),
            ("init the project", "create"),
        ],
    )
    def test_create_strategy(
        self, enhancer: PromptEnhancer, prompt: str, expected: str
    ) -> None:
        assert enhancer._detect_strategy(prompt) == expected

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("edit the config file", "code_edit"),
            ("modify the database schema", "code_edit"),
            ("change the color theme", "code_edit"),
            ("update the version number", "code_edit"),
            ("refactor the router module", "code_edit"),
            ("rename the class to Foo", "code_edit"),
            ("add to the test suite", "code_edit"),
        ],
    )
    def test_edit_strategy(
        self, enhancer: PromptEnhancer, prompt: str, expected: str
    ) -> None:
        assert enhancer._detect_strategy(prompt) == expected

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("explain how the router works", "explain"),
            ("what does this function do", "explain"),
            ("how does the auth module work", "explain"),
            ("why is this variable needed", "explain"),
            ("describe the architecture", "explain"),
            ("tell me about the cost tracker", "explain"),
        ],
    )
    def test_explain_strategy(
        self, enhancer: PromptEnhancer, prompt: str, expected: str
    ) -> None:
        assert enhancer._detect_strategy(prompt) == expected

    def test_question_fallback(self, enhancer: PromptEnhancer) -> None:
        assert enhancer._detect_strategy("list all providers") == "question"

    def test_debug_takes_priority(self, enhancer: PromptEnhancer) -> None:
        """Debug keywords should win even if other keywords are present."""
        assert enhancer._detect_strategy("create a fix for the error") == "debug"


# ---------------------------------------------------------------------------
# File reference extraction
# ---------------------------------------------------------------------------


class TestExtractFileRefs:
    """Tests for PromptEnhancer._extract_file_refs."""

    def test_extracts_existing_file(self, enhancer: PromptEnhancer) -> None:
        refs = enhancer._extract_file_refs("look at src/main.py please")
        assert "src/main.py" in refs

    def test_ignores_nonexistent_file(self, enhancer: PromptEnhancer) -> None:
        refs = enhancer._extract_file_refs("check foo/bar.py")
        assert refs == []

    def test_deduplicates(self, enhancer: PromptEnhancer) -> None:
        refs = enhancer._extract_file_refs(
            "compare src/main.py with src/main.py"
        )
        assert refs.count("src/main.py") == 1

    def test_multiple_files(self, enhancer: PromptEnhancer) -> None:
        refs = enhancer._extract_file_refs(
            "compare src/main.py and src/utils.py"
        )
        assert len(refs) == 2

    def test_no_file_refs(self, enhancer: PromptEnhancer) -> None:
        refs = enhancer._extract_file_refs("just a plain question")
        assert refs == []


# ---------------------------------------------------------------------------
# enhance() integration
# ---------------------------------------------------------------------------


class TestEnhance:
    """Tests for the main enhance() method."""

    def test_returns_enhanced_prompt(self, enhancer: PromptEnhancer) -> None:
        result = enhancer.enhance("list all files")
        assert isinstance(result, EnhancedPrompt)
        assert result.original == "list all files"
        assert result.strategy == "question"

    def test_active_files_appended(
        self, enhancer_with_files: PromptEnhancer
    ) -> None:
        result = enhancer_with_files.enhance("what is this project about")
        assert "Active files:" in result.enhanced
        assert "src/main.py" in result.enhanced

    def test_no_active_files(self, enhancer: PromptEnhancer) -> None:
        result = enhancer.enhance("hello")
        assert "Active files:" not in result.enhanced

    def test_code_edit_includes_file_content(
        self, enhancer: PromptEnhancer
    ) -> None:
        result = enhancer.enhance("edit src/main.py to add logging")
        assert result.strategy == "code_edit"
        assert "Current content of src/main.py" in result.enhanced
        assert 'def main()' in result.enhanced
        assert any("Included content of src/main.py" in c for c in result.context_added)

    def test_code_edit_missing_file(self, enhancer: PromptEnhancer) -> None:
        result = enhancer.enhance("edit nonexistent/file.py")
        assert result.strategy == "code_edit"
        # No file content should be included
        assert "Current content of" not in result.enhanced

    def test_debug_adds_instructions(self, enhancer: PromptEnhancer) -> None:
        result = enhancer.enhance("there is an error in my code")
        assert result.strategy == "debug"
        assert "Debugging context" in result.enhanced
        assert "Added debugging instructions" in result.context_added

    def test_create_adds_project_structure(
        self, enhancer: PromptEnhancer
    ) -> None:
        result = enhancer.enhance("create a new module for caching")
        assert result.strategy == "create"
        assert "Project structure:" in result.enhanced
        assert "src/" in result.enhanced
        assert "Added project structure" in result.context_added

    def test_explain_includes_file_content(
        self, enhancer: PromptEnhancer
    ) -> None:
        result = enhancer.enhance("explain src/main.py")
        assert result.strategy == "explain"
        assert "--- src/main.py ---" in result.enhanced
        assert any("Included src/main.py" in c for c in result.context_added)

    def test_explain_missing_file(self, enhancer: PromptEnhancer) -> None:
        result = enhancer.enhance("explain ghost/module.py")
        assert result.strategy == "explain"
        assert "--- ghost/module.py ---" not in result.enhanced

    def test_question_returns_prompt_unchanged(
        self, enhancer: PromptEnhancer
    ) -> None:
        prompt = "how many providers are supported"
        result = enhancer.enhance(prompt)
        assert result.strategy == "question"
        assert result.enhanced == prompt
        assert result.context_added == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_prompt(self, enhancer: PromptEnhancer) -> None:
        result = enhancer.enhance("")
        assert result.original == ""
        assert result.strategy == "question"

    def test_empty_project_root(self, tmp_path: Path) -> None:
        """Enhancer works even with a completely empty project directory."""
        empty = tmp_path / "empty"
        empty.mkdir()
        e = PromptEnhancer(empty)
        result = e.enhance("create something")
        assert result.strategy == "create"

    def test_code_edit_caps_at_three_files(
        self, project_root: Path
    ) -> None:
        # Create 5 files
        for i in range(5):
            (project_root / f"f{i}.py").write_text(f"# file {i}\n", encoding="utf-8")

        prompt = (
            "edit f0.py f1.py f2.py f3.py f4.py"
        )
        e = PromptEnhancer(project_root)
        result = e.enhance(prompt)
        # At most 3 files should be included
        count = result.enhanced.count("Current content of")
        assert count <= 3

    def test_explain_caps_at_two_files(self, project_root: Path) -> None:
        for i in range(4):
            (project_root / f"e{i}.py").write_text(f"# e{i}\n", encoding="utf-8")

        prompt = "explain e0.py e1.py e2.py e3.py"
        e = PromptEnhancer(project_root)
        result = e.enhance(prompt)
        count = result.enhanced.count("--- e")
        assert count <= 2

    def test_active_files_capped_at_five(self, project_root: Path) -> None:
        many_files = [f"file{i}.py" for i in range(10)]
        e = PromptEnhancer(project_root, active_files=many_files)
        result = e.enhance("what is happening")
        # Only first 5 should appear
        assert "file4.py" in result.enhanced
        assert "file5.py" not in result.enhanced
