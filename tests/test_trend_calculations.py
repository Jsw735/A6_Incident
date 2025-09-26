# Test with controlled incident data
def test_trend_calculations():
    # Create test data with known patterns
    test_incidents = create_test_incident_data()
    
    ta = TrendAnalyzer()
    results = ta.analyze_trends(test_incidents)
    
    # Validate against expected results
    assert results.total_incidents == expected_count
    assert results.priority_distribution == expected_priorities