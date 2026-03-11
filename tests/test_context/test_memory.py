"""Tests for prism.context.memory — ProjectMemory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.context.memory import ProjectMemory
from prism.exceptions import ContextError

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Basic fact operations
# ---------------------------------------------------------------------------


class TestAddFact:
    def test_add_and_get(self, memory: ProjectMemory) -> None:
        memory.add_fact("stack", "Python 3.12")
        assert memory.get_fact("stack") == "Python 3.12"

    def test_add_overwrites(self, memory: ProjectMemory) -> None:
        memory.add_fact("stack", "Python 3.11")
        memory.add_fact("stack", "Python 3.12")
        assert memory.get_fact("stack") == "Python 3.12"

    def test_add_empty_key_raises(self, memory: ProjectMemory) -> None:
        with pytest.raises(ContextError, match="key"):
            memory.add_fact("", "value")

    def test_add_empty_value_raises(self, memory: ProjectMemory) -> None:
        with pytest.raises(ContextError, match="value"):
            memory.add_fact("key", "")

    def test_add_whitespace_key_raises(self, memory: ProjectMemory) -> None:
        with pytest.raises(ContextError, match="key"):
            memory.add_fact("   ", "value")

    def test_add_strips_whitespace(self, memory: ProjectMemory) -> None:
        memory.add_fact("  key  ", "  value  ")
        assert memory.get_fact("key") == "value"


# ---------------------------------------------------------------------------
# get_fact / get_facts
# ---------------------------------------------------------------------------


class TestGetFacts:
    def test_get_nonexistent(self, memory: ProjectMemory) -> None:
        assert memory.get_fact("nope") is None

    def test_get_all_facts(self, memory: ProjectMemory) -> None:
        memory.add_fact("a", "1")
        memory.add_fact("b", "2")
        memory.add_fact("c", "3")
        facts = memory.get_facts()
        assert facts == {"a": "1", "b": "2", "c": "3"}

    def test_get_facts_empty(self, memory: ProjectMemory) -> None:
        assert memory.get_facts() == {}

    def test_get_facts_returns_copy(self, memory: ProjectMemory) -> None:
        memory.add_fact("x", "1")
        facts = memory.get_facts()
        facts["y"] = "2"
        assert memory.get_fact("y") is None


# ---------------------------------------------------------------------------
# remove_fact
# ---------------------------------------------------------------------------


class TestRemoveFact:
    def test_remove_existing(self, memory: ProjectMemory) -> None:
        memory.add_fact("temp", "val")
        assert memory.remove_fact("temp") is True
        assert memory.get_fact("temp") is None

    def test_remove_nonexistent(self, memory: ProjectMemory) -> None:
        assert memory.remove_fact("nope") is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_file_created(self, memory: ProjectMemory) -> None:
        memory.add_fact("stack", "Python")
        assert memory.memory_path.is_file()

    def test_file_content_is_markdown(self, memory: ProjectMemory) -> None:
        memory.add_fact("stack", "Python 3.12")
        content = memory.memory_path.read_text()
        assert "# Project Memory" in content
        assert "**stack**: Python 3.12" in content

    def test_reload_from_disk(self, tmp_path: Path) -> None:
        mem1 = ProjectMemory(tmp_path)
        mem1.add_fact("key1", "value1")
        mem1.add_fact("key2", "value2")

        # Create a new instance to force reload
        mem2 = ProjectMemory(tmp_path)
        facts = mem2.get_facts()
        assert facts["key1"] == "value1"
        assert facts["key2"] == "value2"

    def test_reload_method(self, memory: ProjectMemory) -> None:
        memory.add_fact("x", "original")

        # Manually modify the file
        content = memory.memory_path.read_text()
        content = content.replace("original", "modified")
        memory.memory_path.write_text(content)

        memory.reload()
        assert memory.get_fact("x") == "modified"

    def test_no_file_returns_empty(self, tmp_path: Path) -> None:
        sub = tmp_path / "empty_project"
        sub.mkdir()
        mem = ProjectMemory(sub)
        assert mem.get_facts() == {}


# ---------------------------------------------------------------------------
# get_context_block
# ---------------------------------------------------------------------------


class TestGetContextBlock:
    def test_empty_returns_empty_string(self, memory: ProjectMemory) -> None:
        assert memory.get_context_block() == ""

    def test_includes_header(self, memory: ProjectMemory) -> None:
        memory.add_fact("stack", "Python")
        block = memory.get_context_block()
        assert "[Project Memory]" in block

    def test_includes_facts(self, memory: ProjectMemory) -> None:
        memory.add_fact("stack", "Python 3.12")
        memory.add_fact("framework", "FastAPI")
        block = memory.get_context_block()
        assert "- stack: Python 3.12" in block
        assert "- framework: FastAPI" in block

    def test_facts_sorted(self, memory: ProjectMemory) -> None:
        memory.add_fact("z_last", "3")
        memory.add_fact("a_first", "1")
        memory.add_fact("m_middle", "2")
        block = memory.get_context_block()
        lines = block.strip().split("\n")
        # Skip header line
        fact_lines = [l for l in lines if l.startswith("- ")]
        keys = [l.split(":")[0].strip("- ") for l in fact_lines]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_removes_all_facts(self, memory: ProjectMemory) -> None:
        memory.add_fact("a", "1")
        memory.add_fact("b", "2")
        memory.clear()
        assert memory.get_facts() == {}

    def test_clear_deletes_file(self, memory: ProjectMemory) -> None:
        memory.add_fact("a", "1")
        assert memory.memory_path.is_file()
        memory.clear()
        assert not memory.memory_path.is_file()

    def test_clear_when_no_file(self, memory: ProjectMemory) -> None:
        # Should not raise
        memory.clear()
        assert memory.get_facts() == {}
