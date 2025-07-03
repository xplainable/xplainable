# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.9] - 2024-01-XX

### Changed
- **BREAKING CHANGE**: Removed internal client implementation from `xplainable/client/` directory
- **BREAKING CHANGE**: Client functionality now depends on external `xplainable-client>=1.0.0` package
- Updated import structure to use external client with graceful fallback handling
- Modified GUI components to handle external client with proper attribute checking
- Updated documentation to reference external client package

### Added
- New dependency: `xplainable-client>=1.0.0` in pyproject.toml
- Comprehensive smoke test suite (`tests/test_smoke.py`) covering:
  - Basic classification and regression workflows
  - Partitioned classifier and regressor functionality
  - Model evaluation and metrics
  - Client graceful handling
- Graceful fallback handling for missing external client dependency

### Removed
- Internal client implementation (`xplainable/client/` directory)
- Internal client modules: `client.py`, `datasets.py`, `init.py`
- Direct imports from internal client modules
- **SECURITY**: Removed tornado dependency due to multiple high-severity vulnerabilities
  - Tornado vulnerable to excessive logging caused by malformed multipart form data (High)
  - Tornado has an HTTP cookie parsing DoS vulnerability (High)
  - Tornado has a CRLF injection in CurlAsyncHTTPClient headers (Moderate)
  - Inconsistent Interpretation of HTTP Requests (HTTP Request/Response Smuggling) (Moderate)
  - Tornado vulnerable to HTTP request smuggling via improper parsing of Content-Length fields (Moderate)
  - Open redirect in Tornado (Moderate)

### Fixed
- NumPy random API compatibility issues in core models
- Probabilities return format consistency
- Evaluation metrics structure handling
- Target map handling for custom `TargetMap` class
- Partitioned models requiring `add_partition()` method

### Security
- **CRITICAL**: Removed tornado dependency eliminating 6 security vulnerabilities (2 High, 4 Moderate)
- Conducted security audit of codebase
- Identified safe usage of `exec()` in build and documentation scripts
- Confirmed safe usage of `pickle` for serialization (output only, no unsafe deserialization)
- No unsafe code execution patterns found

### Technical Details
- Version bumped from 1.2.8 to 1.2.9 with breaking changes for client functionality
- All 7 smoke tests pass successfully in ~3.3 seconds
- Maintained backward compatibility where possible with graceful error handling
- Updated documentation build process to work with external client
- Tornado removal does not affect functionality as it was not actively used in the codebase

## [1.2.8] - Previous Release
- Last version with internal client implementation 