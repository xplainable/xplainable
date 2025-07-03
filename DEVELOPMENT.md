# Development Guide

This document outlines the development workflow and processes for the xplainable package.

## Performance Metrics Workflow

The xplainable package includes an automated performance metrics tracking system to prevent algorithm regression. Follow this workflow for different development scenarios:

### 🔄 Complete Development Workflow

1. **Development Phase**
   ```bash
   # Quick CHANGELOG update using existing metrics
   python scripts/update_changelog_metrics.py --skip-tests
   ```
   - Use during active development when you want to update documentation
   - Skips running full test suite for faster iteration
   - Uses existing performance metrics from last test run

2. **Pre-Release Phase**
   ```bash
   # Fresh metrics with full test suite
   python scripts/update_changelog_metrics.py
   ```
   - Run before creating releases to ensure accuracy
   - Executes complete smoke test suite (~3.6 seconds)
   - Generates fresh performance metrics and updates CHANGELOG

3. **Regression Check**
   ```bash
   # Detect performance issues
   python scripts/check_performance.py
   ```
   - Check for algorithm regression with configurable thresholds
   - Compare current metrics with historical performance
   - Exit code 1 if regression detected (CI/CD friendly)

4. **Release**
   - CHANGELOG is automatically up-to-date with latest performance data
   - Performance metrics are tracked in `tests/performance_metrics.json`
   - Baseline metrics documented in CHANGELOG tables

### 📊 Performance Metrics System

The system automatically tracks:

**Classification Metrics:**
- Accuracy and F1-weighted scores
- Sample and feature counts
- Partition information for partitioned models
- Evaluation API functionality

**Regression Metrics:**
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)
- Sample and feature counts
- Partition information for partitioned models

### 🔍 Advanced Usage

```bash
# View performance trends across versions
python scripts/check_performance.py --trends

# Check regression with custom threshold (default: 5%)
python scripts/check_performance.py --regression-threshold 0.10

# View latest metrics without regression check
python scripts/check_performance.py --help
```

## Testing

### Smoke Tests

Run the comprehensive smoke test suite:

```bash
# Run all smoke tests
python -m pytest tests/test_smoke.py -v

# Run with performance metrics output
python -m pytest tests/test_smoke.py -v -s
```

The smoke tests automatically:
- Test core classification and regression APIs
- Verify partitioned model functionality
- Check model evaluation capabilities
- Collect and store performance metrics
- Maintain rolling history of last 10 test runs

### Performance Metrics Storage

- **File**: `tests/performance_metrics.json`
- **Format**: JSON with timestamp, version, and metrics for each test
- **History**: Maintains last 10 entries per test for trend analysis
- **Automatic**: Updated every time smoke tests run

## CI/CD Integration

### GitHub Actions / CI Pipelines

```yaml
# Example CI step for regression detection
- name: Check for Performance Regression
  run: |
    python scripts/check_performance.py
    # Will exit with code 1 if regression detected
```

### Release Process

1. Run full performance update: `python scripts/update_changelog_metrics.py`
2. Review CHANGELOG for updated performance tables
3. Commit changes: `git add CHANGELOG.md tests/performance_metrics.json`
4. Create release with up-to-date performance documentation

## File Structure

```
xplainable/
├── scripts/
│   ├── check_performance.py          # Performance monitoring
│   ├── update_changelog_metrics.py   # CHANGELOG automation
│   └── README.md                     # Scripts documentation
├── tests/
│   ├── test_smoke.py                 # Smoke tests with metrics
│   └── performance_metrics.json     # Historical performance data
├── CHANGELOG.md                      # Auto-updated with metrics
└── DEVELOPMENT.md                    # This file
```

## Best Practices

### Before Committing

1. Run smoke tests to ensure functionality: `python -m pytest tests/test_smoke.py`
2. Check for performance regression: `python scripts/check_performance.py`
3. Update CHANGELOG if needed: `python scripts/update_changelog_metrics.py --skip-tests`

### Before Releases

1. Generate fresh metrics: `python scripts/update_changelog_metrics.py`
2. Review performance tables in CHANGELOG.md
3. Ensure no regression detected
4. Commit updated metrics and documentation

### During Development

- Use `--skip-tests` flag for quick CHANGELOG updates
- Monitor `tests/performance_metrics.json` for performance trends
- Set custom regression thresholds based on acceptable performance variance

## Troubleshooting

### Common Issues

**Smoke tests failing:**
```bash
# Check test output for specific errors
python -m pytest tests/test_smoke.py -v -s
```

**Missing metrics file:**
```bash
# Run smoke tests to generate metrics
python -m pytest tests/test_smoke.py
```

**Regression detected:**
```bash
# View detailed comparison
python scripts/check_performance.py --trends
```

### Getting Help

- Check `scripts/README.md` for detailed script documentation
- Review `tests/test_smoke.py` for test implementation details
- Examine `tests/performance_metrics.json` for historical data format 