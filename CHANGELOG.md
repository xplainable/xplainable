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

### Fixed
- NumPy random API compatibility issues in core models
- Probabilities return format consistency
- Evaluation metrics structure handling
- Target map handling for custom `TargetMap` class
- Partitioned models requiring `add_partition()` method

### Security
- Conducted security audit of codebase
- Identified safe usage of `exec()` in build and documentation scripts
- Confirmed safe usage of `pickle` for serialization (output only, no unsafe deserialization)
- No unsafe code execution patterns found

### Technical Details
- Version bumped from 1.2.8 to 1.2.9 with breaking changes for client functionality
- All 7 smoke tests pass successfully in ~3.5 seconds
- Maintained backward compatibility where possible with graceful error handling
- Updated documentation build process to work with external client

## [1.2.8] - Previous Release
- Last version with internal client implementation 