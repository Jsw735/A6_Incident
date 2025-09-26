import pandas as pd
from analysis.quality_analyzer import QualityAnalyzer


# Test with known data samples
def test_quality_metrics():
    # Create test DataFrame with known issues
    test_data = pd.DataFrame({
        'col1': [1, 2, None, 4],  # 25% missing
        'col2': [1, 1, 2, 3]      # 1 duplicate
    })

    qa = QualityAnalyzer()
    results = qa.analyze_data_quality(test_data)

    # Validate calculations
    assert results.completeness_score == 75.0
    assert results.duplicate_count == 1