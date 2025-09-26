#!/usr/bin/env python3
"""
SAP Incident Analyzer Test Runner
Robust test execution with proper termination handling.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_runner.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def display_banner():
    """Display application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║              SAP INCIDENT ANALYZER                       ║
    ║                   TEST RUNNER                            ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_menu():
    """Display the main test menu."""
    menu = """
    ┌─────────────────────────────────────────────────────────┐
    │                    TEST OPTIONS                         │
    ├─────────────────────────────────────────────────────────┤
    │  1. Run Quality Analyzer Tests                          │
    │  2. Run Trend Analyzer Tests                            │
    │  3. Run All Analyzer Tests                              │
    │  4. View Test Logs                                      │
    │  5. Clear Test Logs                                     │
    │  6. Exit                                                │
    └─────────────────────────────────────────────────────────┘
    """
    print(menu)


def pause_for_user(message="Press ENTER to continue..."):
    """Pause execution and wait for user input."""
    print(f"\n{message}")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return False
    return True


def run_quality_tests():
    """Run Quality Analyzer tests."""
    print("\n🔍 Running Quality Analyzer Tests...")
    logger.info("Starting Quality Analyzer tests")
    
    try:
        # Import and run quality tests
        from analysis.quality_analyzer import QualityAnalyzer
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'Number': ['INC001', 'INC002', 'INC003', 'INC001'],
            'Priority': ['High', 'Medium', None, 'Low'],
            'State': ['Open', 'Closed', 'Open', 'Closed']
        })
        
        # Run analyzer
        analyzer = QualityAnalyzer()
        results = analyzer.analyze_data_quality(test_data)
        
        # Display results
        print(f"✅ Quality Analysis Complete!")
        print(f"   Total Records: {results.total_records}")
        print(f"   Completeness Score: {results.completeness_score:.1f}%")
        print(f"   Duplicate Count: {results.duplicate_count}")
        
        logger.info("Quality Analyzer tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quality Analyzer test failed: {e}")
        logger.error(f"Quality Analyzer test error: {e}")
        return False


def run_trend_tests():
    """Run Trend Analyzer tests."""
    print("\n📈 Running Trend Analyzer Tests...")
    logger.info("Starting Trend Analyzer tests")
    
    try:
        # Import and run trend tests
        from analysis.trend_analyzer import TrendAnalyzer
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'Number': ['INC001', 'INC002', 'INC003', 'INC004'],
            'Priority': ['High', 'High', 'Medium', 'Low'],
            'State': ['Open', 'Closed', 'Open', 'Closed']
        })
        
        # Run analyzer
        analyzer = TrendAnalyzer()
        results = analyzer.analyze_trends(test_data)
        
        # Display results
        print(f"✅ Trend Analysis Complete!")
        print(f"   Total Incidents: {results.total_incidents}")
        print(f"   Priority Distribution: {len(results.priority_distribution)} categories")
        print(f"   Status Distribution: {len(results.status_distribution)} categories")
        
        logger.info("Trend Analyzer tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Trend Analyzer test failed: {e}")
        logger.error(f"Trend Analyzer test error: {e}")
        return False


def run_all_tests():
    """Run all analyzer tests."""
    print("\n🚀 Running Complete Test Suite...")
    logger.info("Starting complete test suite")
    
    results = []
    
    # Run Quality Tests
    results.append(("Quality Analyzer", run_quality_tests()))
    
    # Run Trend Tests  
    results.append(("Trend Analyzer", run_trend_tests()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        logger.info("All tests completed successfully")
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        logger.warning(f"Test suite completed with {total-passed} failures")
    
    return passed == total


def view_logs():
    """Display test logs."""
    print("\n📋 Test Logs:")
    print("="*60)
    
    log_file = Path('test_runner.log')
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = f.read()
                if logs.strip():
                    print(logs)
                else:
                    print("Log file is empty.")
        except Exception as e:
            print(f"Error reading log file: {e}")
    else:
        print("No log file found.")
    
    print("="*60)


def clear_logs():
    """Clear test logs."""
    log_file = Path('test_runner.log')
    if log_file.exists():
        try:
            log_file.unlink()
            print("✅ Test logs cleared successfully.")
            logger.info("Test logs cleared by user")
        except Exception as e:
            print(f"❌ Error clearing logs: {e}")
    else:
        print("No log file to clear.")


def main():
    """
    Main function - entry point for the test runner.
    Implements proper script execution pattern.
    """
    logger.info("Test runner started")
    
    try:
        display_banner()
        
        while True:
            display_menu()
            
            try:
                choice = input("Select option (1-6): ").strip()
                
                if choice == '1':
                    run_quality_tests()
                    if not pause_for_user():
                        break
                        
                elif choice == '2':
                    run_trend_tests()
                    if not pause_for_user():
                        break
                        
                elif choice == '3':
                    run_all_tests()
                    if not pause_for_user():
                        break
                        
                elif choice == '4':
                    view_logs()
                    if not pause_for_user():
                        break
                        
                elif choice == '5':
                    clear_logs()
                    if not pause_for_user():
                        break
                        
                elif choice == '6':
                    print("\n👋 Goodbye!")
                    logger.info("Test runner exited normally")
                    return 0
                    
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation interrupted by user.")
                break
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in main loop: {e}")
                if not pause_for_user("Press ENTER to continue or Ctrl+C to exit..."):
                    break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test runner interrupted by user.")
        logger.info("Test runner interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error in test runner: {e}")
        return 1
    
    finally:
        print("\nTest runner session ended.")
        logger.info("Test runner session ended")
    
    return 0


# This is the proper Python script execution pattern
if __name__ == '__main__':
    """
    Script execution entry point.
    This ensures the script runs properly when executed directly.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Critical error: {e}")
        logging.error(f"Critical error: {e}")
        sys.exit(1)