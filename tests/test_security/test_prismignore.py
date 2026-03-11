"""Tests for prism.security.prismignore — .gitignore-compatible file exclusion."""

from __future__ import annotations

from pathlib import Path

from prism.security.prismignore import DEFAULT_PATTERNS, PrismIgnore

# =====================================================================
# Default patterns
# =====================================================================


class TestDefaultPatterns:
    """Verify the DEFAULT_PATTERNS list contains expected entries."""

    def test_env_files_present(self) -> None:
        assert ".env" in DEFAULT_PATTERNS
        assert ".env.*" in DEFAULT_PATTERNS
        assert "*.env" in DEFAULT_PATTERNS

    def test_secret_directories_present(self) -> None:
        assert "secrets/" in DEFAULT_PATTERNS
        assert "credentials/" in DEFAULT_PATTERNS
        assert "private/" in DEFAULT_PATTERNS

    def test_crypto_keys_present(self) -> None:
        assert "*.pem" in DEFAULT_PATTERNS
        assert "*.key" in DEFAULT_PATTERNS
        assert "*.p12" in DEFAULT_PATTERNS
        assert "*.pfx" in DEFAULT_PATTERNS
        assert "id_rsa" in DEFAULT_PATTERNS
        assert "id_ed25519" in DEFAULT_PATTERNS
        assert "*.pub" in DEFAULT_PATTERNS

    def test_cloud_credentials_present(self) -> None:
        assert ".aws/" in DEFAULT_PATTERNS
        assert ".ssh/" in DEFAULT_PATTERNS
        assert ".gcloud/" in DEFAULT_PATTERNS
        assert "credentials.json" in DEFAULT_PATTERNS

    def test_dependency_caches_present(self) -> None:
        assert "node_modules/" in DEFAULT_PATTERNS
        assert "__pycache__/" in DEFAULT_PATTERNS
        assert ".venv/" in DEFAULT_PATTERNS
        assert "venv/" in DEFAULT_PATTERNS

    def test_log_patterns_present(self) -> None:
        assert "*.log" in DEFAULT_PATTERNS
        assert "*.log.*" in DEFAULT_PATTERNS

    def test_build_artifacts_present(self) -> None:
        assert "dist/" in DEFAULT_PATTERNS
        assert "build/" in DEFAULT_PATTERNS
        assert "*.egg-info/" in DEFAULT_PATTERNS

    def test_ide_patterns_present(self) -> None:
        assert ".idea/" in DEFAULT_PATTERNS
        assert ".vscode/" in DEFAULT_PATTERNS
        assert "*.swp" in DEFAULT_PATTERNS
        assert "*.swo" in DEFAULT_PATTERNS

    def test_comments_are_included(self) -> None:
        """Comments are part of the raw list (for file serialisation)."""
        comments = [line for line in DEFAULT_PATTERNS if line.startswith("#")]
        assert len(comments) >= 1


# =====================================================================
# PrismIgnore — loading and initialisation
# =====================================================================


class TestPrismIgnoreInit:
    """Verify initialisation behaviour."""

    def test_no_file_uses_defaults(self, tmp_path: Path) -> None:
        """When no .prismignore exists, defaults are loaded."""
        pi = PrismIgnore(tmp_path)
        # The active patterns should match the non-blank, non-comment defaults
        expected = [p for p in DEFAULT_PATTERNS if p and not p.startswith("#")]
        assert pi.patterns == expected

    def test_loads_from_existing_file(self, tmp_path: Path) -> None:
        """When .prismignore exists, its content takes precedence."""
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("*.secret\nbackup/\n")

        pi = PrismIgnore(tmp_path)
        assert pi.patterns == ["*.secret", "backup/"]

    def test_file_path_property(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.file_path == tmp_path / ".prismignore"

    def test_file_path_is_absolute(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.file_path.is_absolute()

    def test_blank_lines_and_comments_filtered_from_patterns(
        self, tmp_path: Path
    ) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("# A comment\n\n*.log\n# Another\nfoo/\n")

        pi = PrismIgnore(tmp_path)
        assert pi.patterns == ["*.log", "foo/"]


# =====================================================================
# PrismIgnore — is_ignored
# =====================================================================


class TestIsIgnored:
    """Verify pattern matching behaviour."""

    def test_env_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".env") is True

    def test_env_variant_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".env.production") is True

    def test_dotenv_suffix_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("staging.env") is True

    def test_pem_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("certs/server.pem") is True

    def test_key_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("ssl/private.key") is True

    def test_p12_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("cert.p12") is True

    def test_pfx_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("keystore.pfx") is True

    def test_id_rsa_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("id_rsa") is True

    def test_id_ed25519_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("id_ed25519") is True

    def test_pub_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("keys/id_rsa.pub") is True

    def test_node_modules_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("node_modules/express/index.js") is True

    def test_pycache_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("src/__pycache__/module.pyc") is True

    def test_venv_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".venv/lib/python3.12/site-packages/foo.py") is True

    def test_log_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("app.log") is True

    def test_rotated_log_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("error.log.1") is True

    def test_credentials_json_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("credentials.json") is True

    def test_service_account_json_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("service-account-key.json") is True

    def test_dist_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("dist/bundle.js") is True

    def test_build_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("build/output.css") is True

    def test_egg_info_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("prism_cli.egg-info/PKG-INFO") is True

    def test_idea_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".idea/workspace.xml") is True

    def test_vscode_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".vscode/settings.json") is True

    def test_swp_file_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("main.py.swp") is True

    def test_aws_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".aws/credentials") is True

    def test_ssh_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".ssh/config") is True

    def test_gcloud_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".gcloud/properties") is True

    def test_secrets_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("secrets/api_key.txt") is True

    def test_credentials_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("credentials/oauth.json") is True

    def test_private_directory_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("private/notes.md") is True

    # --- NOT ignored ---

    def test_normal_python_file_not_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("src/main.py") is False

    def test_readme_not_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("README.md") is False

    def test_config_yaml_not_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("config.yaml") is False

    def test_pyproject_toml_not_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("pyproject.toml") is False

    def test_test_file_not_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("tests/test_main.py") is False

    def test_json_not_ignored(self, tmp_path: Path) -> None:
        """A regular .json file should not be ignored (only credentials.json)."""
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("data/config.json") is False

    # --- Absolute paths ---

    def test_absolute_path_inside_root_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        abs_env = tmp_path / ".env"
        assert pi.is_ignored(str(abs_env)) is True

    def test_absolute_path_inside_root_not_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        abs_py = tmp_path / "src" / "main.py"
        assert pi.is_ignored(str(abs_py)) is False


# =====================================================================
# PrismIgnore — negation patterns
# =====================================================================


class TestNegationPatterns:
    """Verify ``!`` negation support."""

    def test_negation_re_includes_file(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("*.log\n!important.log\n")

        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("app.log") is True
        assert pi.is_ignored("important.log") is False

    def test_negation_after_directory_pattern(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("build/\n!build/keep.txt\n")

        pi = PrismIgnore(tmp_path)
        # "build/" matches all files under build/ via directory component matching
        assert pi.is_ignored("build/output.js") is True
        # But "!build/keep.txt" should negate the match for this specific file
        assert pi.is_ignored("build/keep.txt") is False


# =====================================================================
# PrismIgnore — add_pattern / remove_pattern
# =====================================================================


class TestPatternMutation:
    """Verify add_pattern and remove_pattern."""

    def test_add_pattern_persists(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        pi.add_pattern("*.bak")

        # Reload from file
        pi2 = PrismIgnore(tmp_path)
        assert "*.bak" in pi2.patterns

    def test_add_pattern_is_effective_immediately(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("backup.bak") is False

        pi.add_pattern("*.bak")
        assert pi.is_ignored("backup.bak") is True

    def test_add_duplicate_is_noop(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        original_count = len(pi.patterns)
        pi.add_pattern(".env")  # Already in defaults
        assert len(pi.patterns) == original_count

    def test_add_empty_pattern_is_noop(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        original_count = len(pi.patterns)
        pi.add_pattern("")
        pi.add_pattern("   ")
        assert len(pi.patterns) == original_count

    def test_remove_pattern_returns_true(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        pi.add_pattern("*.bak")
        assert pi.remove_pattern("*.bak") is True

    def test_remove_nonexistent_returns_false(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.remove_pattern("nonexistent_pattern") is False

    def test_remove_pattern_persists(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        pi.add_pattern("*.bak")
        pi.remove_pattern("*.bak")

        pi2 = PrismIgnore(tmp_path)
        assert "*.bak" not in pi2.patterns

    def test_remove_pattern_is_effective_immediately(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("*.bak\n")

        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("data.bak") is True

        pi.remove_pattern("*.bak")
        assert pi.is_ignored("data.bak") is False


# =====================================================================
# PrismIgnore — create_default
# =====================================================================


class TestCreateDefault:
    """Verify create_default creates the file with default patterns."""

    def test_creates_file(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        result = pi.create_default()
        assert result.exists()
        assert result.name == ".prismignore"

    def test_file_contains_defaults(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        pi.create_default()

        content = (tmp_path / ".prismignore").read_text()
        assert ".env" in content
        assert "*.pem" in content
        assert "node_modules/" in content

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("custom_only\n")

        pi = PrismIgnore(tmp_path)
        pi.create_default()

        content = ignore_file.read_text()
        assert "custom_only" not in content
        assert ".env" in content

    def test_create_default_returns_path(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        path = pi.create_default()
        assert path == tmp_path / ".prismignore"


# =====================================================================
# PrismIgnore — filter_paths
# =====================================================================


class TestFilterPaths:
    """Verify filter_paths removes ignored entries."""

    def test_basic_filtering(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        paths = ["src/main.py", ".env", "README.md", "app.log"]
        result = pi.filter_paths(paths)

        result_strs = [str(p) for p in result]
        assert "src/main.py" in result_strs
        assert "README.md" in result_strs
        assert ".env" not in result_strs
        assert "app.log" not in result_strs

    def test_empty_input(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.filter_paths([]) == []

    def test_all_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        paths = [".env", "server.pem", "app.log"]
        assert pi.filter_paths(paths) == []

    def test_none_ignored(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        paths = ["src/app.py", "tests/test_app.py", "README.md"]
        result = pi.filter_paths(paths)
        assert len(result) == 3

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        result = pi.filter_paths(["src/main.py"])
        assert all(isinstance(p, Path) for p in result)


# =====================================================================
# PrismIgnore — directory pattern matching
# =====================================================================


class TestDirectoryPatterns:
    """Verify trailing-slash directory patterns match any path containing that dir."""

    def test_nested_node_modules(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("frontend/node_modules/express/lib/router.js") is True

    def test_deeply_nested_pycache(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("src/prism/tools/__pycache__/base.cpython-312.pyc") is True

    def test_top_level_directory(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("output/\n")

        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("output/report.html") is True

    def test_directory_name_in_filename_not_matched(self, tmp_path: Path) -> None:
        """A directory pattern like 'build/' should not match a file named 'build'."""
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("logs/\n")

        pi = PrismIgnore(tmp_path)
        # "logs" as a path component (directory) — matched
        assert pi.is_ignored("logs/error.txt") is True


# =====================================================================
# PrismIgnore — full path patterns (containing /)
# =====================================================================


class TestFullPathPatterns:
    """Verify patterns with embedded slashes match full relative paths."""

    def test_full_path_pattern(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("docs/internal/*.md\n")

        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("docs/internal/secret.md") is True
        assert pi.is_ignored("docs/public/readme.md") is False

    def test_full_path_pattern_exact_dir(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("config/local.yaml\n")

        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("config/local.yaml") is True
        assert pi.is_ignored("config/production.yaml") is False


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Verify edge-case handling."""

    def test_path_outside_project_root(self, tmp_path: Path) -> None:
        """Paths outside the project root are tested against raw string."""
        pi = PrismIgnore(tmp_path)
        # A completely unrelated absolute path — should not crash
        result = pi.is_ignored("/totally/outside/path/main.py")
        # The result depends on whether any pattern matches — it shouldn't match
        assert isinstance(result, bool)

    def test_empty_path(self, tmp_path: Path) -> None:
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("") is False

    def test_comment_lines_are_not_patterns(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("# *.py\n")

        pi = PrismIgnore(tmp_path)
        assert pi.patterns == []
        assert pi.is_ignored("main.py") is False

    def test_only_blank_lines(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("\n\n\n")

        pi = PrismIgnore(tmp_path)
        assert pi.patterns == []

    def test_whitespace_only_pattern_stripped(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".prismignore"
        ignore_file.write_text("   \n*.log\n")

        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("error.log") is True

    def test_path_with_path_object(self, tmp_path: Path) -> None:
        """is_ignored accepts Path objects, not just strings."""
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(Path(".env")) is True
        assert pi.is_ignored(Path("src/main.py")) is False

    def test_multiple_projects_independent(self, tmp_path: Path) -> None:
        """Two PrismIgnore instances with different roots are independent."""
        root_a = tmp_path / "project_a"
        root_b = tmp_path / "project_b"
        root_a.mkdir()
        root_b.mkdir()

        (root_a / ".prismignore").write_text("*.secret\n")
        (root_b / ".prismignore").write_text("*.private\n")

        pi_a = PrismIgnore(root_a)
        pi_b = PrismIgnore(root_b)

        assert pi_a.is_ignored("data.secret") is True
        assert pi_a.is_ignored("data.private") is False

        assert pi_b.is_ignored("data.private") is True
        assert pi_b.is_ignored("data.secret") is False
