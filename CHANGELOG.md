# CHANGELOG.md — Prism Release History

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project instruction files (30 files covering architecture, security, testing, etc.)
- Product plan documentation

### Infrastructure
- Git repository initialized
- Project structure defined in PROJECT_GUIDE.md
- All coding conventions established in CONVENTIONS.md
- CI/CD pipeline designed in CI_CD.md
- Security model documented in SECURITY.md

---

## Version History Format

Each release entry follows this format:

### [X.Y.Z] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes to existing functionality

#### Deprecated
- Features that will be removed in future versions

#### Removed
- Features removed in this version

#### Fixed
- Bug fixes

#### Security
- Security-related changes

---

## Roadmap Notes

This repository is currently source-install only and should be treated as an
experimental CLI project.

Short-term cleanup:

- Keep public docs aligned with local verification.
- Reduce strict mypy debt.
- Add live-provider smoke tests before making provider claims stronger.
- Review optional web/vision dependencies on clean machines.
- Publish only under a unique distribution name if packaging becomes useful.

Longer-term ideas:

- More realistic routing benchmarks.
- Better permission UX for file and command tools.
- Clearer provider setup docs.
- Typed boundaries around orchestration modules.
