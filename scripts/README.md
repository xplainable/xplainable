# Scripts Directory

This directory contains utility scripts for xplainable development and maintenance.

## Performance Checker (`check_performance.py`)

A utility script to monitor algorithm performance and detect potential regression.

### Usage

```bash
# View latest performance metrics
python scripts/check_performance.py

# Check for regression with custom threshold (default: 5%)
python scripts/check_performance.py --regression-threshold 0.10

# Show performance trends across versions
python scripts/check_performance.py --trends
```

### Features

- **Latest Metrics**: Shows current performance metrics for all test cases
- **Regression Detection**: Compares current metrics with previous versions to detect significant drops
- **Trend Analysis**: Displays performance trends across multiple versions
- **Configurable Thresholds**: Customize sensitivity for regression detection

### Metrics Tracked

**Classification:**
- Accuracy
- F1-weighted score
- Sample and feature counts

**Regression:**
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)
- Sample and feature counts

**Other:**
- Partition counts for partitioned models
- Evaluation API functionality

### Integration

The performance checker integrates with:
- `tests/test_smoke.py` - Automatically collects metrics during test runs
- `tests/performance_metrics.json` - Stores historical performance data
- CI/CD pipelines - Can be used to fail builds on regression

### Example Output

```
📊 Latest Performance Metrics
==================================================

🔍 Iris Classification
   Version: 1.2.9
   accuracy: 0.5667
   f1_weighted: 0.4765

🔍 Checking for Regression (threshold: 5.0%)
==================================================
✅ No significant regression detected.
```

The script will exit with code 1 if regression is detected, making it suitable for CI/CD integration.

## CHANGELOG Metrics Updater (`update_changelog_metrics.py`)

A utility script that automatically runs smoke tests and updates the CHANGELOG.md with the latest performance metrics in formatted tables.

### Usage

```bash
# Run smoke tests and update CHANGELOG with latest metrics
python scripts/update_changelog_metrics.py

# Update CHANGELOG using existing metrics (skip running tests)
python scripts/update_changelog_metrics.py --skip-tests
```

### Features

- **Automated Testing**: Runs smoke tests to generate fresh performance metrics
- **Table Formatting**: Creates separate tables for classification and regression metrics
- **Automatic Updates**: Overwrites existing performance section in CHANGELOG.md
- **Timestamp Tracking**: Adds update timestamp to track when metrics were last refreshed
- **Error Handling**: Validates test results and provides clear error messages

### Output Format

The script generates two formatted tables in the CHANGELOG:

**Classification Tasks Table:**
| Dataset | Accuracy | F1-Weighted | Samples | Features | Additional |
|---------|----------|-------------|---------|----------|------------|
| Iris Dataset | 0.5667 | 0.4765 | 150 | 4 |  |
| Breast Cancer Dataset | 0.9600 | 0.9595 | 300 | 30 |  |
| Partitioned Classifier | 0.5667 | N/A | 150 | 4 | 3 partitions |
| Model Evaluation | 0.5667 | N/A | N/A | N/A | CM: ✓, Kappa: ✓ |

**Regression Tasks Table:**
| Dataset | RMSE | R² | Samples | Features | Additional |
|---------|------|----|---------|---------|-----------|
| Diabetes Dataset | 36.8482 | 0.7563 | 300 | 10 |  |
| Partitioned Regressor | N/A | N/A | 442 | 10 | 2 partitions |

### Integration

- **Release Process**: Run before creating releases to ensure CHANGELOG has latest metrics
- **CI/CD Pipelines**: Can be integrated to automatically update documentation
- **Development Workflow**: Use to keep CHANGELOG current during development

### Example Usage

```bash
# Before a release
python scripts/update_changelog_metrics.py

# During development (using existing metrics)
python scripts/update_changelog_metrics.py --skip-tests
```

The script will automatically:
1. Run smoke tests (unless `--skip-tests` is used)
2. Extract latest performance metrics
3. Format them into classification and regression tables
4. Update the CHANGELOG.md with the new tables
5. Add a timestamp showing when metrics were last updated 