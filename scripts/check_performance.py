#!/usr/bin/env python3
"""
Performance metrics checker for xplainable.

This script helps developers view performance trends and detect potential
algorithm regression by comparing current metrics with historical data.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any


def load_metrics() -> Dict[str, List[Dict[str, Any]]]:
    """Load performance metrics from JSON file."""
    metrics_file = "tests/performance_metrics.json"
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        print("Run the smoke tests first: python -m pytest tests/test_smoke.py")
        sys.exit(1)
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


def print_latest_metrics(metrics: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print the latest metrics for each test."""
    print("📊 Latest Performance Metrics")
    print("=" * 50)
    
    for test_name, history in metrics.items():
        if not history:
            continue
            
        latest = history[-1]
        version = latest['version']
        timestamp = latest['timestamp']
        test_metrics = latest['metrics']
        
        print(f"\n🔍 {test_name.replace('_', ' ').title()}")
        print(f"   Version: {version}")
        print(f"   Timestamp: {timestamp}")
        
        for metric_name, value in test_metrics.items():
            if isinstance(value, float):
                print(f"   {metric_name}: {value:.4f}")
            else:
                print(f"   {metric_name}: {value}")


def check_regression(metrics: Dict[str, List[Dict[str, Any]]], threshold: float = 0.05) -> bool:
    """Check for potential regression by comparing latest metrics with previous ones."""
    print(f"\n🔍 Checking for Regression (threshold: {threshold:.1%})")
    print("=" * 50)
    
    has_regression = False
    
    for test_name, history in metrics.items():
        if len(history) < 2:
            print(f"⚠️  {test_name}: Not enough history for comparison")
            continue
            
        current = history[-1]['metrics']
        previous = history[-2]['metrics']
        
        print(f"\n📈 {test_name.replace('_', ' ').title()}")
        
        # Check key performance metrics
        key_metrics = ['accuracy', 'f1_weighted', 'r2']
        for metric in key_metrics:
            if metric in current and metric in previous:
                current_val = current[metric]
                previous_val = previous[metric]
                
                if previous_val != 0:
                    change = (current_val - previous_val) / previous_val
                    
                    if change < -threshold:  # Significant decrease
                        print(f"   ❌ {metric}: {previous_val:.4f} → {current_val:.4f} ({change:.1%})")
                        has_regression = True
                    elif change > threshold:  # Significant improvement
                        print(f"   ✅ {metric}: {previous_val:.4f} → {current_val:.4f} ({change:.1%})")
                    else:
                        print(f"   ➡️  {metric}: {previous_val:.4f} → {current_val:.4f} ({change:.1%})")
        
        # Check RMSE (lower is better)
        if 'rmse' in current and 'rmse' in previous:
            current_val = current['rmse']
            previous_val = previous['rmse']
            
            if previous_val != 0:
                change = (current_val - previous_val) / previous_val
                
                if change > threshold:  # Significant increase (worse)
                    print(f"   ❌ rmse: {previous_val:.4f} → {current_val:.4f} ({change:.1%})")
                    has_regression = True
                elif change < -threshold:  # Significant decrease (better)
                    print(f"   ✅ rmse: {previous_val:.4f} → {current_val:.4f} ({change:.1%})")
                else:
                    print(f"   ➡️  rmse: {previous_val:.4f} → {current_val:.4f} ({change:.1%})")
    
    return has_regression


def print_trend_summary(metrics: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print a summary of trends across all versions."""
    print(f"\n📊 Performance Trend Summary")
    print("=" * 50)
    
    for test_name, history in metrics.items():
        if len(history) < 2:
            continue
            
        print(f"\n📈 {test_name.replace('_', ' ').title()}")
        print(f"   Versions tracked: {len(history)}")
        
        # Show version range
        oldest = history[0]
        newest = history[-1]
        print(f"   Version range: {oldest['version']} → {newest['version']}")
        
        # Show key metric trends
        for metric in ['accuracy', 'f1_weighted', 'r2', 'rmse']:
            values = [entry['metrics'].get(metric) for entry in history if metric in entry['metrics']]
            if len(values) >= 2:
                first, last = values[0], values[-1]
                if first != 0:
                    trend = (last - first) / first
                    direction = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                    print(f"   {metric}: {first:.4f} → {last:.4f} {direction}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check xplainable performance metrics")
    parser.add_argument("--regression-threshold", type=float, default=0.05,
                       help="Threshold for regression detection (default: 0.05 = 5%)")
    parser.add_argument("--trends", action="store_true",
                       help="Show performance trends across versions")
    
    args = parser.parse_args()
    
    # Load metrics
    metrics = load_metrics()
    
    # Print latest metrics
    print_latest_metrics(metrics)
    
    # Check for regression
    has_regression = check_regression(metrics, args.regression_threshold)
    
    # Print trends if requested
    if args.trends:
        print_trend_summary(metrics)
    
    # Summary
    print(f"\n{'='*50}")
    if has_regression:
        print("❌ POTENTIAL REGRESSION DETECTED!")
        print("Please review the changes and investigate any significant performance drops.")
        sys.exit(1)
    else:
        print("✅ No significant regression detected.")
        print("Performance metrics are within acceptable ranges.")


if __name__ == "__main__":
    main() 