# TOOL_SPECS.md — Prism Tool Specifications

## Tool Interface

Every tool implements this interface:

```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g., 'read_file')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the model."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""

    @property
    @abstractmethod
    def permission_level(self) -> PermissionLevel:
        """AUTO, CONFIRM, or DANGEROUS."""

    @abstractmethod
    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Execute the tool and return result."""
```

## Tool Registry

Tools available to models during conversation:

| Tool | Permission | Description |
|------|-----------|-------------|
| `read_file` | AUTO | Read file contents |
| `write_file` | CONFIRM | Create or overwrite a file |
| `edit_file` | CONFIRM | Search/replace edit in a file |
| `list_directory` | AUTO | List directory contents |
| `search_codebase` | AUTO | Full-text search across project |
| `execute_command` | CONFIRM | Run shell command |
| `browse_web` | CONFIRM | Fetch and extract web page content |
| `screenshot` | CONFIRM | Capture web page screenshot |

## Tool Specifications

### read_file

```json
{
    "name": "read_file",
    "description": "Read the contents of a file. Returns the full file content with line numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to project root"
            },
            "start_line": {
                "type": "integer",
                "description": "Optional: start reading from this line number (1-indexed)"
            },
            "end_line": {
                "type": "integer",
                "description": "Optional: stop reading at this line number (inclusive)"
            }
        },
        "required": ["path"]
    }
}
```

**Implementation Details**:
- Path validated by PathGuard (resolve realpath, check within project root)
- Excluded file patterns checked (reject .env, credentials, etc.)
- Max file size: 1MB (return truncated with warning if larger)
- Binary file detection: check first 8KB for null bytes, reject with message
- Returns content with line numbers: `1: line content\n2: line content\n...`
- UTF-8 encoding assumed, fallback to latin-1 on decode error

### write_file

```json
{
    "name": "write_file",
    "description": "Create a new file or overwrite an existing file. Shows diff before writing.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to project root"
            },
            "content": {
                "type": "string",
                "description": "The complete file content to write"
            }
        },
        "required": ["path", "content"]
    }
}
```

**Implementation Details**:
- Path validated by PathGuard
- If file exists: show colored diff (old vs new) before confirmation
- If file is new: show full content with "NEW FILE" header
- Create parent directories if they don't exist
- After write: trigger git auto-commit (if git repo)
- Excluded patterns enforced (cannot write .env, credentials, etc.)
- No write to files outside project root

### edit_file

```json
{
    "name": "edit_file",
    "description": "Edit a file using search and replace. The search string must match exactly.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to project root"
            },
            "search": {
                "type": "string",
                "description": "The exact text to find in the file (must match exactly including whitespace)"
            },
            "replace": {
                "type": "string",
                "description": "The text to replace it with"
            }
        },
        "required": ["path", "search", "replace"]
    }
}
```

**Implementation Details**:
- Path validated by PathGuard
- Search string must appear exactly once in the file (error if 0 or 2+ matches)
- Show colored diff of the change before confirmation
- Preserve file encoding and line endings
- After edit: trigger git auto-commit
- Fuzzy matching: if exact match fails, try with normalized whitespace and suggest

### list_directory

```json
{
    "name": "list_directory",
    "description": "List the contents of a directory. Shows files and subdirectories with sizes.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory, relative to project root. Use '.' for project root."
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, list recursively (like tree). Default false.",
                "default": false
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for recursive listing. Default 3.",
                "default": 3
            }
        },
        "required": ["path"]
    }
}
```

**Implementation Details**:
- Path validated by PathGuard
- Respects .gitignore patterns (skip node_modules, .git, etc.)
- Shows: type (file/dir), size, last modified
- Recursive: tree-style output with depth limit
- Max 500 entries (truncate with count of remaining)

### search_codebase

```json
{
    "name": "search_codebase",
    "description": "Search for text patterns across the entire codebase using regex.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for"
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob pattern to filter files (e.g., '*.py', 'src/**/*.ts')"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results. Default 50.",
                "default": 50
            }
        },
        "required": ["pattern"]
    }
}
```

**Implementation Details**:
- Uses ripgrep (`rg`) if available, falls back to Python `re` module
- Respects .gitignore
- Returns: file path, line number, matching line, context (2 lines before/after)
- Binary files excluded automatically
- Max 50 results by default (configurable)
- Regex validation: reject dangerous patterns (catastrophic backtracking)

### execute_command

```json
{
    "name": "execute_command",
    "description": "Execute a shell command in the project directory. Requires confirmation unless in allowlist.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default 30, max 300.",
                "default": 30
            }
        },
        "required": ["command"]
    }
}
```

**Implementation Details**:
- Working directory: project root (cannot change)
- Blocked commands checked first (see SECURITY.md)
- Allowlisted commands skip confirmation
- Environment filtered: all *_API_KEY, *_SECRET, *_TOKEN, *_PASSWORD removed
- Subprocess: `shell=False` when possible (split command), `shell=True` for pipes
- Output capture: 100KB stdout limit, 10KB stderr limit
- Timeout: default 30s, max 300s, process killed on timeout
- Returns: stdout, stderr, exit code, duration, timed_out flag

### browse_web

```json
{
    "name": "browse_web",
    "description": "Fetch a web page and extract its readable content. Disabled by default.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch"
            },
            "extract_mode": {
                "type": "string",
                "enum": ["text", "html", "markdown"],
                "description": "Content extraction format. Default 'text'.",
                "default": "text"
            }
        },
        "required": ["url"]
    }
}
```

**Implementation Details**:
- **Disabled by default** — enabled with `--web` flag or `/web on`
- URL validation: reject file://, javascript:, data: schemes
- Two paths:
  - **Fast path** (httpx + BeautifulSoup): for simple pages, documentation
  - **Full path** (Playwright): for JavaScript-rendered pages
- Content size limit: 1MB extracted text
- No cookies persisted
- No file downloads
- Timeout: 30s
- Content wrapped in delimiters for prompt injection defense

### screenshot

```json
{
    "name": "screenshot",
    "description": "Capture a screenshot of a web page. Returns base64-encoded image for multimodal models.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to screenshot"
            },
            "selector": {
                "type": "string",
                "description": "Optional CSS selector to capture specific element"
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full scrollable page. Default false.",
                "default": false
            }
        },
        "required": ["url"]
    }
}
```

**Implementation Details**:
- **Disabled by default** (requires `--web`)
- Uses Playwright headless Chromium
- Returns base64-encoded PNG
- Only works with multimodal models (GPT-4o, Claude Sonnet, Gemini)
- If routed model doesn't support vision: skip screenshot, return text description
- Viewport: 1280x720 default
- Timeout: 30s for page load
- Max image size: 2MB (resize if larger)

## Tool Context

Every tool execution receives:
```python
@dataclass
class ToolContext:
    project_root: Path
    working_directory: Path
    session_id: str
    auto_approve: bool          # --yes mode
    web_enabled: bool           # --web mode
    timeout: float              # Default tool timeout
    allowed_commands: list[str] # Pre-approved commands
    path_guard: PathGuard
    git: GitOperations | None
    audit: AuditLogger
```

## Tool Output Format

Tools return structured results:
```python
@dataclass
class ToolResult:
    tool_call_id: str
    content: str              # The tool output (file contents, command output, etc.)
    success: bool
    error: str | None = None  # Error message if success=False
    metadata: dict = field(default_factory=dict)  # Extra info (bytes written, exit code, etc.)
```
