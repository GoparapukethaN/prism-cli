# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Prism, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Email your report to the maintainers with the subject line: `[SECURITY] Prism vulnerability report`.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report.
- **Assessment**: We will assess the severity within 1 week.
- **Fix**: Critical vulnerabilities will be patched within 2 weeks. Non-critical issues will be addressed in the next release cycle.
- **Disclosure**: We will coordinate disclosure with you. We request a 90-day disclosure window for critical issues.

### Scope

The following are in scope for security reports:

- **API key exposure**: Any path where API keys could be logged, displayed, or leaked
- **Path traversal**: File operations that escape the project root
- **Command injection**: Unsanitized input reaching shell execution
- **Credential storage**: Weaknesses in keyring/encrypted storage
- **Dependency vulnerabilities**: Known CVEs in direct dependencies

### Out of Scope

- Issues in third-party AI provider APIs
- Denial of service via resource exhaustion (CLI is single-user)
- Social engineering attacks

### Security Design

Prism implements multiple security layers:

- **Path traversal prevention**: All file paths are resolved and validated against the project root
- **Secret filtering**: API keys are scrubbed from logs, error messages, and subprocess environments
- **Command sandboxing**: Terminal commands run with timeouts, output limits, and filtered environment variables
- **Sensitive file blocking**: Patterns like `.env`, `.ssh/`, and credential files are excluded from tool operations
- **Audit logging**: Every tool execution is logged to `~/.prism/audit.log`
- **Credential storage**: API keys are stored in the OS keyring (macOS Keychain, Windows Credential Locker, Linux Secret Service) with encrypted fallback

## Acknowledgments

We appreciate security researchers who help keep Prism safe. Contributors will be acknowledged in release notes (unless they prefer to remain anonymous).
