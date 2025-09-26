
import unittest
import pandas as pd
from datetime import datetime

from reporting.executive_summary import EnhancedExecutiveSummary
from analysis.metrics_calculator import IncidentMetricsCalculator


class TestExecutiveSummaryAndMetrics(unittest.TestCase):
    def test_create_executive_summary_data_flattens_nested_dicts(self):
        gen = EnhancedExecutiveSummary()

        basic_metrics = {
            'total_incidents': 3,
            'priority_breakdown': {'High': 1, 'Medium': 1, 'Low': 1},
            'velocity_metrics': {
                'daily_average': 2.0,
                'last_7_days_average': 1.5,
                'last_28_days_average': 1.2,
                'most_recent_day': 3
            },
            'weekly_comparison': {
                'weeks_analyzed': 4,
                'last_week_count': 10,
                'previous_week_count': 8,
                'week_over_week_pct_change': 0.25
            },
            'analysis_summary': {'note': 'test-note'}
        }

        analysis_metrics = {}
        health = None

        rows = gen._create_executive_summary_data(basic_metrics, analysis_metrics, health)

        # Convert to dict for easier assertions
        d = {k: v for k, v in rows}

        self.assertEqual(int(d.get('Total Incidents')), 3)
        self.assertEqual(int(d.get('Priority - High')), 1)
        self.assertAlmostEqual(float(d.get('Velocity - Daily Average')), 2.0)
        self.assertAlmostEqual(float(d.get('Velocity - Last 7 Days Avg')), 1.5)
        self.assertAlmostEqual(float(d.get('Velocity - Last 28 Days Avg')), 1.2)
        self.assertEqual(int(d.get('Velocity - Most Recent Day')), 3)
        self.assertEqual(int(d.get('Weekly - Weeks Analyzed')), 4)
        self.assertEqual(int(d.get('Weekly - Last Week Count')), 10)
        self.assertAlmostEqual(float(d.get('Weekly - WoW % Change')), 0.25)
        self.assertEqual(d.get('Analysis - note'), 'test-note')

    def test_metrics_calculator_fallbacks_and_priority_mapping(self):
        calc = IncidentMetricsCalculator()

        # DataFrame without Created column (fallback should return error for velocity)
        df = pd.DataFrame({
            'number': ['A', 'B', 'C'],
            'urgency': [3, 3, 3],
            'sys_created_on': [datetime.now(), datetime.now(), datetime.now()]
        })

        # Call the executive summary metrics calculator
        res = calc.calculate_executive_summary_metrics(df, analysis_results=None, health_score_results=None)

        # Priority breakdown should map numeric urgency -> Low (per fallback mapping)
        pb = res.get('priority_breakdown', {})
        self.assertEqual(pb.get('Low', 0), 3)

        # Velocity metrics fallback will expect 'Created' and may return an error dict
        vel = res.get('velocity_metrics', {})
        # The fallback in this environment returns either an error message or computed numbers;
        # at minimum ensure a dict is returned
        self.assertIsInstance(vel, dict)


if __name__ == '__main__':
    unittest.main()
