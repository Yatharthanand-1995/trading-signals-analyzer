#!/usr/bin/env python3
"""
Test Runner for Trading System
Runs all tests and generates coverage report
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """
    Run all tests with pytest
    """
    print("🧪 RUNNING TRADING SYSTEM TESTS")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Set test environment
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['FINNHUB_API_KEY'] = 'test_key'
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest', 'pytest-cov'])
        import pytest
    
    # Run tests with coverage
    print("\n🏃 Running tests...\n")
    
    pytest_args = [
        '-v',  # Verbose
        '--tb=short',  # Short traceback
        '--color=yes',  # Colored output
        '--cov=core_scripts',  # Coverage for core_scripts
        '--cov=config',  # Coverage for config
        '--cov=ml_models',  # Coverage for ml_models
        '--cov-report=term-missing',  # Show missing lines
        '--cov-report=html:coverage_report',  # HTML report
        'tests/'  # Test directory
    ]
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        print("\n📈 Coverage report generated at: coverage_report/index.html")
    else:
        print("\n❌ Some tests failed!")
    
    return exit_code

def run_specific_test(test_file):
    """
    Run a specific test file
    """
    print(f"\n🏃 Running {test_file}...\n")
    
    pytest_args = [
        '-v',
        '--tb=short',
        '--color=yes',
        test_file
    ]
    
    return pytest.main(pytest_args)

def run_security_check():
    """
    Run security check
    """
    print("\n🔍 Running security check...\n")
    
    security_script = Path(__file__).parent / 'scripts' / 'security_check.py'
    if security_script.exists():
        subprocess.run([sys.executable, str(security_script)])
    else:
        print("⚠️ Security check script not found")

def main():
    """
    Main test runner
    """
    if len(sys.argv) > 1:
        # Run specific test
        test_file = sys.argv[1]
        exit_code = run_specific_test(test_file)
    else:
        # Run all tests
        exit_code = run_tests()
        
        # Also run security check
        run_security_check()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()