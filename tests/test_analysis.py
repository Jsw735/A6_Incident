#!/usr/bin/env python3
"""
Comprehensive test suite for the analysis module.
Tests all analysis components following clean code testing principles.
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add project root to path for imports
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to Python path for module discovery
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import the analysis module
from analysis import (
    QualityAnalyzer, 
    TrendAnalyzer, 
    MetricsCalculator, 
    KeywordManager
)

def test_analysis_module():
    """Test the complete analysis module functionality."""
    print("🔍 Testing Analysis Module Components...")
    
    try:
        # Test 1: Module Import Test
        print("\nStep 1: Testing module imports...")
        from analysis import (
            IncidentQualityAnalyzer, QualityAnalyzer,
            IncidentTrendAnalyzer, TrendAnalyzer, 
            IncidentMetricsCalculator, MetricsCalculator,
            IncidentKeywordManager, KeywordManager
        )
        print("✅ All analysis module imports successful")
        
        # Test 2: Class Instantiation Test
        print("\nStep 2: Testing class instantiation...")
        quality_analyzer = IncidentQualityAnalyzer()
        trend_analyzer = IncidentTrendAnalyzer()
        metrics_calculator = IncidentMetricsCalculator()
        keyword_manager = IncidentKeywordManager()
        print("✅ All analysis classes instantiated successfully")
        
        # Test 3: Create Test Data
        print("\nStep 3: Creating test incident data...")
        test_data = create_test_incident_data()
        print(f"✅ Test data created: {len(test_data)} rows")
        
        # Test 4: Quality Analysis Test
        print("\nStep 4: Testing quality analysis...")
        quality_results = quality_analyzer.analyze_quality(test_data)
        validate_quality_results(quality_results)
        print("✅ Quality analysis completed successfully")
        
        # Test 5: Trend Analysis Test
        print("\nStep 5: Testing trend analysis...")
        trend_results = trend_analyzer.analyze_trends(test_data)
        validate_trend_results(trend_results)
        print("✅ Trend analysis completed successfully")
        
        # Test 6: Metrics Calculation Test
        print("\nStep 6: Testing metrics calculation...")
        metrics_results = metrics_calculator.calculate_all_metrics(test_data)
        validate_metrics_results(metrics_results)
        print("✅ Metrics calculation completed successfully")
        
        # Test 7: Keyword Analysis Test
        print("\nStep 7: Testing keyword analysis...")
        keyword_results = test_keyword_functionality(keyword_manager, test_data)
        validate_keyword_results(keyword_results)
        print("✅ Keyword analysis completed successfully")
        
        # Test 8: Integration Test
        print("\nStep 8: Testing analysis integration...")
        integration_results = test_analysis_integration(
            quality_analyzer, trend_analyzer, metrics_calculator, keyword_manager, test_data
        )
        print("✅ Analysis integration test completed successfully")
        
        print("\n🎉 Analysis Module test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis Module test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_test_incident_data():
    """Create comprehensive test data for analysis testing."""
    base_date = datetime.now() - timedelta(days=30)
    
    test_data = []
    for i in range(50):  # Create 50 test incidents
        incident_date = base_date + timedelta(days=i % 30, hours=i % 24)
        resolved_date = incident_date + timedelta(hours=2 + (i % 48))
        
        incident = {
            'Number': f'INC{1000000 + i:07d}',
            'Priority': ['Critical', 'High', 'Medium', 'Low'][i % 4],
            'State': ['New', 'In Progress', 'Resolved', 'Closed'][i % 4],
            'Created': incident_date,
            'Resolved': resolved_date if i % 3 == 0 else None,  # Some unresolved
            'Short Description': f'SAP issue {i} - ABAP enhancement needed',
            'Description': f'Detailed description of SAP incident {i} involving MM module and RFC calls',
            'Assignment Group': ['SAP Basis', 'SAP Functional', 'SAP Development'][i % 3],
            'Category': 'SAP Application',
            'Subcategory': 'Performance Issue'
        }
        test_data.append(incident)
    
    return pd.DataFrame(test_data)

def validate_quality_results(results):
    """Validate quality analysis results structure and content."""
    required_keys = ['analysis_timestamp', 'total_incidents', 'quality_dimensions', 'overall_quality']
    
    for key in required_keys:
        assert key in results, f"Missing required key: {key}"
    
    # Validate quality dimensions
    quality_dims = results['quality_dimensions']
    expected_dims = ['completeness', 'consistency', 'accuracy', 'timeliness']
    
    for dim in expected_dims:
        assert dim in quality_dims, f"Missing quality dimension: {dim}"
        assert 'score' in quality_dims[dim], f"Missing score for dimension: {dim}"
        assert 0 <= quality_dims[dim]['score'] <= 1, f"Invalid score for {dim}"
    
    # Validate overall quality
    overall = results['overall_quality']
    assert 'score' in overall, "Missing overall quality score"
    assert 0 <= overall['score'] <= 1, "Invalid overall quality score"

def validate_trend_results(results):
    """Validate trend analysis results structure."""
    # TrendMetrics dataclass validation
    required_attrs = [
        'total_incidents', 'trend_period_days', 'incident_velocity',
        'priority_distribution', 'status_distribution', 'top_categories',
        'resolution_trends', 'peak_hours', 'seasonal_patterns'
    ]
    
    for attr in required_attrs:
        assert hasattr(results, attr), f"Missing trend attribute: {attr}"
    
    assert results.total_incidents > 0, "Invalid total incidents count"
    assert results.trend_period_days >= 0, "Invalid trend period"

def validate_metrics_results(results):
    """Validate metrics calculation results structure."""
    required_sections = [
        'volume_metrics', 'capacity_metrics', 'trend_metrics',
        'performance_metrics', 'projection_metrics'
    ]
    
    for section in required_sections:
        assert section in results, f"Missing metrics section: {section}"
    
    # Validate volume metrics
    volume = results['volume_metrics']
    assert 'total_incidents' in volume, "Missing total incidents in volume metrics"
    assert volume['total_incidents'] > 0, "Invalid total incidents count"

def test_keyword_functionality(keyword_manager, test_data):
    """Test keyword analysis functionality."""
    # Test keyword analysis on sample text
    sample_text = "SAP ABAP enhancement needed for MM module RFC connection"
    keyword_results = keyword_manager.analyze_text_for_keywords(sample_text)
    
    assert 'keywords_found' in keyword_results, "Missing keywords_found in results"
    assert 'classification' in keyword_results, "Missing classification in results"
    assert 'confidence' in keyword_results, "Missing confidence in results"
    
    # Test learning from data
    learning_results = keyword_manager.learn_from_data(test_data)
    
    return {
        'keyword_analysis': keyword_results,
        'learning_results': learning_results
    }

def validate_keyword_results(results):
    """Validate keyword analysis results."""
    keyword_analysis = results['keyword_analysis']
    assert isinstance(keyword_analysis['keywords_found'], list), "Keywords found should be a list"
    assert isinstance(keyword_analysis['classification'], str), "Classification should be a string"
    assert 0 <= keyword_analysis['confidence'] <= 1, "Confidence should be between 0 and 1"

def test_analysis_integration(quality_analyzer, trend_analyzer, metrics_calculator, keyword_manager, test_data):
    """Test integration between all analysis components."""
    # Run all analyses
    quality_results = quality_analyzer.analyze_quality(test_data)
    trend_results = trend_analyzer.analyze_trends(test_data)
    metrics_results = metrics_calculator.calculate_all_metrics(test_data)
    
    # Test report generation
    quality_report = quality_analyzer.generate_quality_report(quality_results)
    trend_report = trend_analyzer.generate_trend_report(trend_results)
    
    # Validate reports are strings and not empty
    assert isinstance(quality_report, str) and len(quality_report) > 0, "Quality report should be non-empty string"
    assert isinstance(trend_report, str) and len(trend_report) > 0, "Trend report should be non-empty string"
    
    return {
        'quality_report_length': len(quality_report),
        'trend_report_length': len(trend_report),
        'integration_successful': True
    }

if __name__ == "__main__":
    print("=" * 60)
    print("SAP INCIDENT REPORTING SYSTEM - ANALYSIS MODULE TEST")
    print("=" * 60)
    
    success = test_analysis_module()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS MODULE TEST PASSED!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ ANALYSIS MODULE TEST FAILED!")
        print("=" * 60)
        sys.exit(1)