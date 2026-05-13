# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-05-13

### Removed
- **`xplainable.preprocessing`** — The built-in preprocessing module (`XPipeline`, `XPipeline.add_stages()`, and all `transformers` including `ChangeCases`, `DropCols`, `Condense`, `SetDType`, etc.) has been removed. Use the standalone `xplainable-preprocessing` package instead, which provides a spec-driven, JSON-serializable `PipelineSpec` API.
- **`xplainable.gui`** — The legacy ipywidgets-based GUI (`Preprocessor`, `classifier`, `regressor`, `EvaluateClassifier`, `EvaluateRegressor`, `ModelPersist`, `ScenarioClassification`, `ScenarioRegression`, `XFineTuner`) has been removed. Use the Xplainable Cloud platform for interactive model management.
- **`xplainable.callbacks`** — The `RegressionCallback` and `OptCallback` classes (used exclusively by the removed GUI) have been removed.
- **`xplainable._dependencies`** — The `_try_optional_dependencies_gui()` helper and related GUI dependency checks have been removed.
- **`xplainable.utils.xwidgets`** — The ipywidgets wrapper utilities have been removed.
- **GUI optional dependencies** — The `pip install xplainable[gui]` extra (ipywidgets, gradio, drawsvg, traitlets, jupyter-client) has been removed from `pyproject.toml`.

### Changed
- **Slimmed core dependencies** — Removed GUI-only packages from `requirements.txt`: `drawsvg`, `ipywidgets`, `traitlets`, `jupyter-client`, `seaborn`, `matplotlib`, `pyperclip`, `sphinx_rtd_theme`, `plotly`. Core install is now significantly lighter.
- **All example notebooks** updated to use `xplainable-preprocessing` `PipelineSpec`/`StepSpec`/`compile_spec` API instead of the removed `XPipeline`/`xtf` approach.
- **Preprocessor cloud persistence** updated across all notebooks to use `client.preprocessing.create_preprocessor(spec=..., sample_df=...)` with JSON-serializable specs.

### Added
- **New example notebooks**:
  - `Shopify_Customer_Churn.ipynb` — Customer churn prediction from Shopify order data with contribution-driven retention optimization using model partition profiles as counterfactual lever effects.
  - `Shopify_Order_Returns.ipynb` — Order return prediction with contribution-driven intervention routing.

### Migration Guide

**Before (1.3.x):**
```python
from xplainable.preprocessing.pipeline import XPipeline
from xplainable.preprocessing import transformers as xtf

pipeline = XPipeline()
pipeline.add_stages([
    {"transformer": xtf.ChangeCases(columns=[...], case="lower")},
    {"transformer": xtf.DropCols(columns=[...])},
])
df_transformed = pipeline.fit_transform(df)
```

**After (1.4.0):**
```python
from xplainable_preprocessing import PipelineSpec, StepSpec, compile_spec

spec = PipelineSpec(steps=[
    StepSpec(id="lowercase", type="TextCleanTransformer",
             columns=[...], params={"operations": ["lowercase"]}),
    StepSpec(id="drop", type="DropColumnsTransformer",
             params={"columns": [...]}),
])
pipeline = compile_spec(spec)
df_transformed = pipeline.fit_transform(df)
```

Install the new preprocessing package: `pip install xplainable-preprocessing`

## [1.3.1] - 2025-06-01

### Changed
- Updated client and rolled version
- Preserve original exception in TransformerError for better debugging

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
- **Performance metrics tracking system** to monitor algorithm performance and prevent regression
- Automated metrics collection in `tests/performance_metrics.json` with version history
- **Performance checker utility** (`scripts/check_performance.py`) for:
  - Viewing latest performance metrics
  - Detecting algorithm regression with configurable thresholds
  - Analyzing performance trends across versions
  - CI/CD integration support
- **CHANGELOG metrics updater** (`scripts/update_changelog_metrics.py`) for:
  - Automatically running smoke tests and updating CHANGELOG.md
  - Formatting performance metrics in separate classification and regression tables
  - Timestamped updates with latest performance data
  - Integration with release and development workflows

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

### Performance Baseline (v1.2.9) - Updated 2025-07-03

**Classification Tasks:**
| Dataset | Accuracy | F1-Weighted | Samples | Features | Additional |
|---------|----------|-------------|---------|----------|------------|
| Iris Dataset | 0.5667 | 0.4765 | 150 | 4 |  |
| Breast Cancer Dataset | 0.9600 | 0.9595 | 300 | 30 |  |
| Partitioned Classifier | 0.5667 | N/A | 150 | 4 | 3 partitions |
| Model Evaluation | 0.5667 | N/A | N/A | N/A | CM: ✓, Kappa: ✓ |


**Regression Tasks:**
| Dataset | RMSE | R² | Samples | Features | Additional |
|---------|------|----|---------|---------|-----------|
| Diabetes Dataset | 36.8482 | 0.7563 | 300 | 10 |  |
| Partitioned Regressor | N/A | N/A | 442 | 10 | 2 partitions |


*Note: These metrics serve as baseline for detecting algorithm regression in future versions. All metrics are automatically tracked in `tests/performance_metrics.json`.*### Security
- **CRITICAL**: Removed tornado dependency eliminating 6 security vulnerabilities (2 High, 4 Moderate)
- Conducted security audit of codebase
- Identified safe usage of `exec()` in build and documentation scripts
- Confirmed safe usage of `pickle` for serialization (output only, no unsafe deserialization)
- No unsafe code execution patterns found

### Technical Details
- Version bumped from 1.2.8 to 1.2.9 with breaking changes for client functionality
- All 7 smoke tests pass successfully in ~3.6 seconds
- Maintained backward compatibility where possible with graceful error handling
- Updated documentation build process to work with external client
- Tornado removal does not affect functionality as it was not actively used in the codebase
- Performance metrics tracking system keeps last 10 entries per test for trend analysis

## [1.2.8] - Previous Release
- Last version with internal client implementation 