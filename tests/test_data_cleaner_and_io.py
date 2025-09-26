import unittest
import pandas as pd
from datetime import datetime

from data_processing.data_cleaner import IncidentDataCleaner


class TestDataCleanerCanonicalColumns(unittest.TestCase):
    def test_ensure_canonical_output_columns_maps_sys_created_on_and_resolved_at(self):
        cleaner = IncidentDataCleaner()

        df = pd.DataFrame({
            'Number': ['INC001'],
            'sys_created_on': ['2025-09-20 10:00:00'],
            'resolved_at': ['2025-09-21 12:00:00'],
            'Urgency': ['High']
        })

        cleaned = cleaner.clean_incident_data(df)

        # After cleaning, canonical columns should exist
        self.assertIn('Created', cleaned.columns)
        self.assertIn('Resolved', cleaned.columns)
        self.assertIn('Priority', cleaned.columns)

        # Created/Resolved should be datetimes
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['Created']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['Resolved']))

    def test_excel_output_roundtrip(self):
        # Ensure writing and reading back an executive summary DataFrame works
        import pandas as pd
        from reporting.executive_summary import EnhancedExecutiveSummary

        gen = EnhancedExecutiveSummary()
        rows = [('Metric A', 1), ('Metric B', 'x')]
        path = gen._create_excel_output(rows)

        df = pd.read_excel(path, sheet_name='Executive Summary')
        self.assertEqual(len(df), 2)


if __name__ == '__main__':
    unittest.main()
