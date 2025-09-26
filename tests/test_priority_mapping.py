import pandas as pd
from data_processing.data_cleaner import IncidentDataCleaner
from config.settings import Settings


def test_urgency_to_priority_mapping():
    settings = Settings()
    mapping = settings.get_urgency_mapping()

    # Build test DataFrame with mixed urgency types
    df = pd.DataFrame({
        'Number': ['INC1', 'INC2', 'INC3', 'INC4'],
        'Urgency': [1, '2', 3.0, '4']
    })

    cleaner = IncidentDataCleaner()
    cleaned_df = cleaner.clean_incident_data(df)

    # Expect a Priority column normalized to configured labels
    assert 'Priority' in cleaned_df.columns

    expected = [mapping.get('1'), mapping.get('2'), mapping.get('3'), mapping.get('4')]
    got = cleaned_df['Priority'].astype(str).tolist()
    assert got == [str(x) for x in expected]
