#!/usr/bin/env python3
"""
Update CHANGELOG with latest performance metrics.

This script runs smoke tests and updates the CHANGELOG.md with the latest
performance metrics in table format, separated by classification and regression tasks.
"""

import json
import os
import sys
import subprocess
import re
from datetime import datetime
from typing import Dict, List, Any


def run_smoke_tests():
    """Run smoke tests to generate fresh metrics."""
    print("🧪 Running smoke tests to generate fresh metrics...")
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_smoke.py", "-v"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print("❌ Smoke tests failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            sys.exit(1)
        else:
            print("✅ Smoke tests completed successfully!")
            
    except Exception as e:
        print(f"❌ Error running smoke tests: {e}")
        sys.exit(1)


def load_latest_metrics() -> Dict[str, Dict[str, Any]]:
    """Load the latest metrics from the JSON file."""
    metrics_file = "tests/performance_metrics.json"
    
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        sys.exit(1)
    
    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)
    
    # Extract latest metrics for each test
    latest_metrics = {}
    for test_name, history in all_metrics.items():
        if history:
            latest_metrics[test_name] = history[-1]['metrics']
    
    return latest_metrics


def format_classification_table(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Format classification metrics into a markdown table."""
    
    # Classification tests
    classification_tests = [
        ("Iris Dataset", "iris_classification"),
        ("Breast Cancer Dataset", "breast_cancer_classification"), 
        ("Partitioned Classifier", "partitioned_classifier"),
        ("Model Evaluation", "model_evaluation")
    ]
    
    table = "| Dataset | Accuracy | F1-Weighted | Samples | Features | Additional |\n"
    table += "|---------|----------|-------------|---------|----------|------------|\n"
    
    for display_name, test_key in classification_tests:
        if test_key in metrics:
            test_metrics = metrics[test_key]
            
            # Extract common metrics
            accuracy = test_metrics.get('accuracy', 'N/A')
            f1_weighted = test_metrics.get('f1_weighted', 'N/A')
            n_samples = test_metrics.get('n_samples', 'N/A')
            n_features = test_metrics.get('n_features', 'N/A')
            
            # Format accuracy and f1
            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            f1_str = f"{f1_weighted:.4f}" if isinstance(f1_weighted, (int, float)) else str(f1_weighted)
            
            # Additional info specific to each test
            additional = ""
            if test_key == "partitioned_classifier":
                n_partitions = test_metrics.get('n_partitions', 'N/A')
                additional = f"{n_partitions} partitions"
            elif test_key == "model_evaluation":
                has_cm = test_metrics.get('has_confusion_matrix', False)
                has_kappa = test_metrics.get('has_cohen_kappa', False)
                additional = f"CM: {'✓' if has_cm else '✗'}, Kappa: {'✓' if has_kappa else '✗'}"
            
            table += f"| {display_name} | {accuracy_str} | {f1_str} | {n_samples} | {n_features} | {additional} |\n"
    
    return table


def format_regression_table(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Format regression metrics into a markdown table."""
    
    # Regression tests
    regression_tests = [
        ("Diabetes Dataset", "diabetes_regression"),
        ("Partitioned Regressor", "partitioned_regressor")
    ]
    
    table = "| Dataset | RMSE | R² | Samples | Features | Additional |\n"
    table += "|---------|------|----|---------|---------|-----------|\n"
    
    for display_name, test_key in regression_tests:
        if test_key in metrics:
            test_metrics = metrics[test_key]
            
            # Extract common metrics
            rmse = test_metrics.get('rmse', 'N/A')
            r2 = test_metrics.get('r2', 'N/A')
            n_samples = test_metrics.get('n_samples', 'N/A')
            n_features = test_metrics.get('n_features', 'N/A')
            
            # Format RMSE and R²
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
            
            # Additional info specific to each test
            additional = ""
            if test_key == "partitioned_regressor":
                n_partitions = test_metrics.get('n_partitions', 'N/A')
                additional = f"{n_partitions} partitions"
            
            table += f"| {display_name} | {rmse_str} | {r2_str} | {n_samples} | {n_features} | {additional} |\n"
    
    return table


def update_changelog(classification_table: str, regression_table: str):
    """Update the CHANGELOG.md with new performance metrics tables."""
    
    changelog_file = "CHANGELOG.md"
    
    if not os.path.exists(changelog_file):
        print(f"❌ CHANGELOG file not found: {changelog_file}")
        sys.exit(1)
    
    # Read current changelog
    with open(changelog_file, 'r') as f:
        content = f.read()
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    # Create new performance section
    new_performance_section = f"""### Performance Baseline (v1.2.9) - Updated {timestamp}

**Classification Tasks:**
{classification_table}

**Regression Tasks:**
{regression_table}

*Note: These metrics serve as baseline for detecting algorithm regression in future versions. All metrics are automatically tracked in `tests/performance_metrics.json`.*"""
    
    # Replace the existing performance section
    # Look for the pattern between "### Performance Baseline" and the next "###" or "##"
    pattern = r'### Performance Baseline.*?(?=###|##|\Z)'
    
    if re.search(pattern, content, re.DOTALL):
        # Replace existing section
        new_content = re.sub(pattern, new_performance_section, content, flags=re.DOTALL)
        print("✅ Updated existing performance section in CHANGELOG")
    else:
        # If no existing section found, add it before the Security section
        security_pattern = r'(### Security)'
        if re.search(security_pattern, content):
            new_content = re.sub(security_pattern, f"{new_performance_section}\n\n\\1", content)
            print("✅ Added new performance section to CHANGELOG")
        else:
            print("⚠️  Could not find insertion point in CHANGELOG")
            return False
    
    # Write updated changelog
    with open(changelog_file, 'w') as f:
        f.write(new_content)
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update CHANGELOG with latest performance metrics")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running smoke tests (use existing metrics)")
    
    args = parser.parse_args()
    
    print("📊 Updating CHANGELOG with Performance Metrics")
    print("=" * 50)
    
    # Run smoke tests unless skipped
    if not args.skip_tests:
        run_smoke_tests()
    else:
        print("⏭️  Skipping smoke tests (using existing metrics)")
    
    # Load latest metrics
    print("📖 Loading latest performance metrics...")
    metrics = load_latest_metrics()
    
    # Format tables
    print("📋 Formatting classification metrics table...")
    classification_table = format_classification_table(metrics)
    
    print("📋 Formatting regression metrics table...")
    regression_table = format_regression_table(metrics)
    
    # Update changelog
    print("📝 Updating CHANGELOG.md...")
    success = update_changelog(classification_table, regression_table)
    
    if success:
        print("\n✅ CHANGELOG.md successfully updated with latest performance metrics!")
        print("\nClassification Table:")
        print(classification_table)
        print("\nRegression Table:")
        print(regression_table)
    else:
        print("\n❌ Failed to update CHANGELOG.md")
        sys.exit(1)


if __name__ == "__main__":
    main() 