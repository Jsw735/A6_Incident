"""
Comprehensive test suite for incident analyzers with proper logging and termination.
"""

import logging
import sys
import pytest
import pandas as pd
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import your analyzers
sys.path.append(str(Path(__file__).parent.parent))
from analysis.quality_analyzer import QualityAnalyzer
from analysis.trend_analyzer import TrendAnalyzer


class TestQualityAnalyzer:
    """Test suite for Quality Analyzer with proper logging."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        logger.info("Setting up QualityAnalyzer test")
        self.analyzer = QualityAnalyzer()
        
        # Create test data with known characteristics
        self.test_data = pd.DataFrame({
            'Number': ['INC001', 'INC002', 'INC003', 'INC001'],  # 1 duplicate
            'Priority': ['High', 'Medium', None, 'Low'],         # 1 missing (25%)
            'State': ['Open', 'Closed', 'Open', 'Closed'],
            'Description': ['Test 1', 'Test 2', 'Test 3', 'Test 4']
        })
        logger.info(f"Created test dataset with {len(self.test_data)} records")
    
    def test_data_quality_metrics(self):
        """Test quality metrics calculation with known data."""
        logger.info("Testing data quality metrics calculation")
        
        results = self.analyzer.analyze_data_quality(self.test_data)
        
        # Validate completeness (3/4 = 75% for Priority column)
        expected_completeness = 75.0
        assert abs(results.completeness_score - expected_completeness) < 0.1, \
            f"Expected completeness {expected_completeness}, got {results.completeness_score}"
        
        # Validate duplicate count
        expected_duplicates = 1
        assert results.duplicate_count == expected_duplicates, \
            f"Expected {expected_duplicates} duplicates, got {results.duplicate_count}"
        
        logger.info(f"✅ Quality metrics test passed - Completeness: {results.completeness_score}%, Duplicates: {results.duplicate_count}")
    
    def test_empty_data_handling(self):
        """Test analyzer behavior with empty data."""
        logger.info("Testing empty data handling")
        
        empty_df = pd.DataFrame()
        results = self.analyzer.analyze_data_quality(empty_df)
        
        assert results.total_records == 0
        assert results.completeness_score == 0.0
        
        logger.info("✅ Empty data handling test passed")
    
    def teardown_method(self):
        """Cleanup after each test."""
        logger.info("QualityAnalyzer test completed\n")


class TestTrendAnalyzer:
    """Test suite for Trend Analyzer with proper logging."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        logger.info("Setting up TrendAnalyzer test")
        self.analyzer = TrendAnalyzer()
        
        # Create test data with known patterns
        self.test_data = pd.DataFrame({
            'Number': ['INC001', 'INC002', 'INC003', 'INC004'],
            'Priority': ['High', 'High', 'Medium', 'Low'],  # 2 High, 1 Medium, 1 Low
            'State': ['Open', 'Closed', 'Open', 'Closed'],  # 2 Open, 2 Closed
            'Created': pd.date_range('2024-01-01', periods=4, freq='D')
        })
        logger.info(f"Created test dataset with {len(self.test_data)} records")
    
    def test_trend_analysis(self):
        """Test trend analysis with known data patterns."""
        logger.info("Testing trend analysis calculations")
        
        results = self.analyzer.analyze_trends(self.test_data)
        
        # Validate total incidents
        expected_total = 4
        assert results.total_incidents == expected_total, \
            f"Expected {expected_total} incidents, got {results.total_incidents}"
        
        # Validate priority distribution
        expected_high_priority = 2
        actual_high_priority = results.priority_distribution.get('High', 0)
        assert actual_high_priority == expected_high_priority, \
            f"Expected {expected_high_priority} High priority, got {actual_high_priority}"
        
        logger.info(f"✅ Trend analysis test passed - Total: {results.total_incidents}, High Priority: {actual_high_priority}")
    
    def test_missing_columns_handling(self):
        """Test analyzer behavior with missing columns."""
        logger.info("Testing missing columns handling")
        
        minimal_df = pd.DataFrame({
            'Number': ['INC001', 'INC002']
        })
        
        results = self.analyzer.analyze_trends(minimal_df)
        
        assert results.total_incidents == 2
        assert len(results.priority_distribution) == 0  # No Priority column
        
        logger.info("✅ Missing columns handling test passed")
    
    def teardown_method(self):
        """Cleanup after each test."""
        logger.info("TrendAnalyzer test completed\n")


def run_comprehensive_test_suite():
    """
    Run all tests with proper logging and user interaction.
    Includes stop screen and detailed reporting.
    """
    print("=" * 60)
    print("SAP INCIDENT ANALYZER - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    logger.info("Starting comprehensive test suite")
    
    # Run tests with pytest
    test_results = pytest.main([
        __file__,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--log-cli-level=INFO'  # Show logs in console
    ])
    
    # Test completion summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETION SUMMARY")
    print("=" * 60)
    
    if test_results == 0:
        print("🎉 ALL TESTS PASSED! Your analyzers are working correctly.")
        logger.info("All tests completed successfully")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for details.")
        logger.error(f"Test suite failed with exit code: {test_results}")
    
    print(f"📋 Detailed logs saved to: tests/test_results.log")
    print("=" * 60)
    
    # Interactive stop screen
    input("\nPress ENTER to continue or Ctrl+C to exit...")
    
    return test_results


if __name__ == '__main__':
    """
    Main execution with proper script termination handling.
    """
    try:
        # Ensure test directory exists
        Path('tests').mkdir(exist_ok=True)
        
        # Run the comprehensive test suite
        exit_code = run_comprehensive_test_suite()
        
        # Clean exit
        print("Test execution completed. Goodbye!")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user.")
        logger.info("Test execution interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)