#!/usr/bin/env python3
"""
Metrics Calculator for SAP Incident Analysis
Calculates trends, capacity metrics, and projections.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from typing import Dict, List, Set, Any, Optional, Union, Tuple


class IncidentMetricsCalculator:
    """
    Calculates key metrics for incident analysis including trends,
    capacity planning, and performance projections.
    """
    
    def __init__(self):
        """Initialize the Metrics Calculator."""
        self.logger = logging.getLogger(__name__)
        self.metrics = {}

    # ------------------------------------------------------------------
    # Helper Utilities
    # ------------------------------------------------------------------
    def _get_open_incidents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return subset of incidents considered OPEN.

        Definition (conservative):
          - If a resolved_date style column exists (resolved_date / Resolved / Closed / Close date), treat rows
            with a non-null timestamp in any of those columns as closed.
          - Additionally inspect a status/state column (State / Status / Incident State) and exclude rows whose
            value is in a known CLOSED set.

        This dual approach avoids misclassification when one dimension is missing or dirty.
        """
        try:
            if df is None or len(df) == 0:
                return df

            df_copy = df.copy()

            # Normalize possible resolved indicators
            resolved_cols = [c for c in df_copy.columns if c.lower() in ['resolved', 'closed', 'resolved at', 'close date', 'resolved_date']]
            for c in resolved_cols:
                df_copy[c] = pd.to_datetime(df_copy[c], errors='coerce')

            # Determine closed via any resolved timestamp present
            if resolved_cols:
                closed_mask = None
                for c in resolved_cols:
                    col_mask = df_copy[c].notna()
                    closed_mask = col_mask if closed_mask is None else (closed_mask | col_mask)
            else:
                closed_mask = pd.Series([False] * len(df_copy), index=df_copy.index)

            # Status-based refinement
            status_cols = ['State', 'Status', 'Incident State']
            status_col = None
            for col in status_cols:
                if col in df_copy.columns:
                    status_col = col
                    break

            closed_status_values = {'closed', 'resolved', 'cancelled', 'canceled', 'done', 'completed'}
            if status_col:
                status_series = df_copy[status_col].astype(str).str.lower()
                status_closed_mask = status_series.isin(closed_status_values)
                closed_mask = closed_mask | status_closed_mask

            open_df = df_copy[~closed_mask]
            return open_df
        except Exception as e:
            self.logger.error(f"Error filtering open incidents: {e}")
            return df
    
    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for incident data.
        
        Args:
            df: DataFrame containing incident data
            
        Returns:
            Dictionary containing all calculated metrics
        """
        try:
            self.logger.info(f"Starting metrics calculation for {len(df)} incidents")
            
            self.metrics = {
                'volume_metrics': self._calculate_volume_metrics(df),
                'capacity_metrics': self._calculate_capacity_metrics(df),
                'trend_metrics': self._calculate_trend_metrics(df),
                'performance_metrics': self._calculate_performance_metrics(df),
                'projection_metrics': self._calculate_projections(df)
            }
            
            self.logger.info("Metrics calculation completed successfully")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based metrics."""
        total_incidents = len(df)
        
        # Daily volume calculations
        if 'Created' in df.columns:
            df_copy = df.copy()
            df_copy['Created'] = pd.to_datetime(df_copy['Created'], errors='coerce')
            valid_dates = df_copy['Created'].dropna()
            
            if len(valid_dates) > 0:
                daily_volumes = valid_dates.groupby(valid_dates.dt.date).size()
                
                return {
                    'total_incidents': total_incidents,
                    'daily_average': float(daily_volumes.mean()),
                    'daily_peak': int(daily_volumes.max()),
                    'daily_minimum': int(daily_volumes.min()),
                    'volume_variance': float(daily_volumes.var())
                }
        
        return {'total_incidents': total_incidents}
    
    def calculate_priority_breakdown(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate priority breakdown using specified business rules:
        P0 + P1 = High Priority
        P2 = Medium Priority  
        P3 = Low Priority
        
        Following context documentation on groupby operations and aggregations.
        """
        try:
            # Use authoritative 'urgency' column if present; else fall back
            if 'urgency' in df.columns:
                raw_series = df['urgency'].astype(str).fillna('')
            elif 'Urgency' in df.columns:
                raw_series = df['Urgency'].astype(str).fillna('')
            elif 'priority' in df.columns:
                raw_series = df['priority'].astype(str).fillna('')
            elif 'Priority' in df.columns:
                raw_series = df['Priority'].astype(str).fillna('')
            else:
                self.logger.warning("Priority/urgency column not found in data")
                return {'High': 0, 'Medium': 0, 'Low': 0}

            # Simple mapping for patterns like '1 - High', '2 - Medium', '3 - Low'
            def simple_map(v: str) -> str:
                s = v.strip()
                if not s:
                    return 'Low'
                lower = s.lower()
                if 'high' in lower:
                    return 'High'
                if 'medium' in lower:
                    return 'Medium'
                if 'low' in lower:
                    return 'Low'
                # Digit-led pattern
                if s[0].isdigit():
                    d = s[0]
                    if d == '1' or d == '0':
                        return 'High'
                    if d == '2':
                        return 'Medium'
                    if d == '3':
                        return 'Low'
                return 'Low'

            mapped = raw_series.map(simple_map)
            counts = mapped.value_counts().to_dict()

            result = {
                'High': int(counts.get('High', 0)),
                'Medium': int(counts.get('Medium', 0)),
                'Low': int(counts.get('Low', 0))
            }

            self.logger.info(f"Priority breakdown calculated: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in priority breakdown calculation: {str(e)}")
            return {'High': 0, 'Medium': 0, 'Low': 0}

    def calculate_executive_summary_metrics(self, df: pd.DataFrame, analysis_results: Dict[str, Any], health_score_results: Any) -> Dict[str, Any]:
        """
        Calculate all executive summary metrics including comprehensive dashboard metrics.
        Coordinates multiple metric calculations for executive dashboard.
        """
        try:
            # Priority breakdown (core business logic)
            priority_breakdown = self.calculate_priority_breakdown(df)
            
            # Enhanced priority metrics with targets and trends
            enhanced_priority_metrics = self._calculate_enhanced_priority_metrics(df)
            
            # Backlog and week-over-week comparison metrics
            backlog_metrics = self._calculate_backlog_metrics(df)
            
            # Progress to target metrics
            target_metrics = self._calculate_target_progress_metrics(df)
            
            # Workstream breakdown metrics
            workstream_metrics = self._calculate_workstream_breakdown(df)
            
            # Velocity metrics using rolling averages
            if hasattr(self, '_calculate_velocity_metrics'):
                velocity_metrics = self._calculate_velocity_metrics(df)
            else:
                velocity_metrics = self._fallback_velocity_metrics(df)
            
            # Week-over-week comparison using date operations
            if hasattr(self, '_calculate_weekly_comparison'):
                weekly_comparison = self._calculate_weekly_comparison(df)
            else:
                weekly_comparison = self._fallback_weekly_comparison(df)
            
            # Capacity utilization metrics
            capacity_metrics = self._calculate_capacity_metrics(df)
            
            # Projection metrics for future planning
            projection_metrics = self._calculate_projections(df)
            
            # Incorporate analysis results and health score
            if hasattr(self, '_process_analysis_results'):
                analysis_summary = self._process_analysis_results(analysis_results)
            else:
                analysis_summary = self._fallback_process_analysis_results(analysis_results)

            if hasattr(self, '_process_health_score'):
                health_summary = self._process_health_score(health_score_results)
            else:
                health_summary = self._fallback_process_health_score(health_score_results)
            
            # Combine all metrics for executive summary
            executive_metrics = {
                'priority_breakdown': priority_breakdown,
                'enhanced_priority_metrics': enhanced_priority_metrics,
                'backlog_metrics': backlog_metrics,
                'target_metrics': target_metrics,
                'workstream_metrics': workstream_metrics,
                'velocity_metrics': velocity_metrics,
                'weekly_comparison': weekly_comparison,
                'capacity_metrics': capacity_metrics,
                'projection_metrics': projection_metrics,
                'analysis_summary': analysis_summary,
                'health_summary': health_summary,
                'total_incidents': len(df),
                'calculation_timestamp': pd.Timestamp.now()
            }
            
            return executive_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating executive summary metrics: {str(e)}")
            return {}
    
    def _calculate_capacity_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate capacity and workload metrics."""
        capacity_metrics = {}
        
        # Assignment group capacity
        assignment_cols = ['Assignment Group', 'Assignment group', 'Assigned to']
        assignment_col = None
        for col in assignment_cols:
            if col in df.columns:
                assignment_col = col
                break
        
        if assignment_col:
            group_workload = df[assignment_col].value_counts()
            capacity_metrics['workload_distribution'] = group_workload.head(10).to_dict()
            capacity_metrics['max_workload_group'] = str(group_workload.index[0])
            capacity_metrics['workload_imbalance'] = float(group_workload.std())
        
        # Priority-based capacity
        if 'Priority' in df.columns:
            priority_counts = df['Priority'].value_counts()
            high_priority = priority_counts.get('Critical', 0) + priority_counts.get('High', 0)
            capacity_metrics['high_priority_load'] = int(high_priority)
            capacity_metrics['priority_ratio'] = float(high_priority / len(df)) if len(df) > 0 else 0
        
        # Open vs Closed capacity
        status_cols = ['State', 'Status', 'Incident State']
        status_col = None
        for col in status_cols:
            if col in df.columns:
                status_col = col
                break
        
        if status_col:
            open_states = ['New', 'In Progress', 'Open', 'Active', 'Assigned']
            open_incidents = df[df[status_col].isin(open_states)]
            capacity_metrics['open_incident_count'] = len(open_incidents)
            capacity_metrics['closure_rate'] = float((len(df) - len(open_incidents)) / len(df)) if len(df) > 0 else 0
        
        return capacity_metrics
    
    def _calculate_trend_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate 4-6 week trend metrics with open-focus option."""
        if 'Created' not in df.columns:
            return {'error': 'Created date column required for trend analysis'}
        
        try:
            df_copy = df.copy()
            df_copy['Created'] = pd.to_datetime(df_copy['Created'], errors='coerce')
            valid_data = df_copy.dropna(subset=['Created'])
            
            if len(valid_data) == 0:
                return {'error': 'No valid dates found'}
            
            # Get open incidents subset for trend analysis
            open_data = self._get_open_incidents(valid_data)
            if 'Created' in open_data.columns:
                open_data = open_data.dropna(subset=['Created'])
            
            # Calculate weekly trends for last 6 weeks
            end_date = valid_data['Created'].max()
            start_date = end_date - timedelta(weeks=6)
            
            recent_total = valid_data[valid_data['Created'] >= start_date]
            recent_open = open_data[open_data['Created'] >= start_date] if len(open_data) > 0 else pd.DataFrame()
            
            if len(recent_total) == 0:
                return {'error': 'No recent data found'}
            
            total_weekly_counts = recent_total.groupby(recent_total['Created'].dt.to_period('W')).size()
            open_weekly_counts = recent_open.groupby(recent_open['Created'].dt.to_period('W')).size() if len(recent_open) > 0 else pd.Series()
            
            # Trend calculation for both total and open
            def calculate_trend(weekly_counts):
                if len(weekly_counts) >= 2:
                    x_values = range(len(weekly_counts))
                    y_values = weekly_counts.values
                    slope = np.polyfit(x_values, y_values, 1)[0]
                    direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                    return slope, direction, weekly_counts.mean()
                else:
                    return 0, 'insufficient_data', weekly_counts.mean() if len(weekly_counts) > 0 else 0
            
            total_slope, total_direction, total_avg = calculate_trend(total_weekly_counts)
            open_slope, open_direction, open_avg = calculate_trend(open_weekly_counts) if len(open_weekly_counts) > 0 else (0, 'no_open_data', 0)
            
            return {
                # Open-focused trend metrics (primary)
                'open_weekly_counts': {str(k): int(v) for k, v in open_weekly_counts.items()},
                'open_trend_slope': float(open_slope),
                'open_trend_direction': open_direction,
                'open_average_weekly_volume': float(open_avg),
                'open_weeks_analyzed': len(open_weekly_counts),
                # Total trend metrics (historical context)
                'total_weekly_counts': {str(k): int(v) for k, v in total_weekly_counts.items()},
                'total_trend_slope': float(total_slope),
                'total_trend_direction': total_direction,
                'total_average_weekly_volume': float(total_avg),
                'total_weeks_analyzed': len(total_weekly_counts),
                # Legacy keys for backward compatibility
                'weekly_counts': {str(k): int(v) for k, v in open_weekly_counts.items()},
                'trend_slope': float(open_slope),
                'trend_direction': open_direction,
                'weeks_analyzed': len(open_weekly_counts),
                'average_weekly_volume': float(open_avg)
            }
            
        except Exception as e:
            return {'error': f'Error in trend calculation: {e}'}
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance and SLA metrics."""
        performance = {}
        
        # Resolution time metrics
        created_col = 'Created'
        resolved_cols = ['Resolved', 'Closed', 'Resolved at', 'Close date']
        resolved_col = None
        
        for col in resolved_cols:
            if col in df.columns:
                resolved_col = col
                break
        
        if created_col in df.columns and resolved_col:
            df_copy = df.copy()
            df_copy[created_col] = pd.to_datetime(df_copy[created_col], errors='coerce')
            df_copy[resolved_col] = pd.to_datetime(df_copy[resolved_col], errors='coerce')
            
            resolved_incidents = df_copy.dropna(subset=[created_col, resolved_col])
            
            if len(resolved_incidents) > 0:
                resolution_times = (resolved_incidents[resolved_col] - 
                                  resolved_incidents[created_col]).dt.total_seconds() / 3600
                
                performance['average_resolution_hours'] = float(resolution_times.mean())
                performance['median_resolution_hours'] = float(resolution_times.median())
                performance['resolution_rate'] = float(len(resolved_incidents) / len(df))
                
                # SLA performance (assuming 24h for High, 72h for Medium)
                if 'Priority' in df.columns:
                    high_priority = resolved_incidents[resolved_incidents['Priority'].isin(['Critical', 'High'])]
                    if len(high_priority) > 0:
                        high_res_times = (high_priority[resolved_col] - 
                                        high_priority[created_col]).dt.total_seconds() / 3600
                        performance['high_priority_sla_compliance'] = float((high_res_times <= 24).mean())
        
        return performance
    
    def _calculate_projections(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate future projections based on trends with open-focus."""
        if 'Created' not in df.columns:
            return {'error': 'Created date required for projections'}
        
        try:
            df_copy = df.copy()
            df_copy['Created'] = pd.to_datetime(df_copy['Created'], errors='coerce')
            valid_data = df_copy.dropna(subset=['Created'])
            
            if len(valid_data) == 0:
                return {'error': 'No valid dates found'}
            
            # Get open incidents subset for projections
            open_data = self._get_open_incidents(valid_data)
            if 'Created' in open_data.columns:
                open_data = open_data.dropna(subset=['Created'])
            
            # Get last 4 weeks of data for projection
            end_date = valid_data['Created'].max()
            start_date = end_date - timedelta(weeks=4)
            recent_total = valid_data[valid_data['Created'] >= start_date]
            recent_open = open_data[open_data['Created'] >= start_date] if len(open_data) > 0 else pd.DataFrame()
            
            def calculate_projections(data, prefix=""):
                if len(data) == 0:
                    return {f'{prefix}error': 'No recent data found'}
                
                daily_counts = data.groupby(data['Created'].dt.date).size()
                
                if len(daily_counts) >= 7:  # Need at least a week of data
                    # Simple linear projection
                    x_values = range(len(daily_counts))
                    y_values = daily_counts.values
                    trend_slope = np.polyfit(x_values, y_values, 1)[0]
                    current_average = daily_counts.mean()
                    
                    # Project next 2 weeks
                    projected_daily = current_average + (trend_slope * 7)  # 7 days ahead
                    projected_weekly = projected_daily * 7
                    projected_monthly = projected_daily * 30
                    
                    return {
                        f'{prefix}current_daily_average': float(current_average),
                        f'{prefix}projected_daily_volume': float(max(0, projected_daily)),
                        f'{prefix}projected_weekly_volume': float(max(0, projected_weekly)),
                        f'{prefix}projected_monthly_volume': float(max(0, projected_monthly)),
                        f'{prefix}trend_confidence': float(min(len(daily_counts) / 28, 1.0))
                    }
                
                return {f'{prefix}error': 'Insufficient data for projections'}
            
            # Calculate projections for both open and total
            open_projections = calculate_projections(recent_open, "open_")
            total_projections = calculate_projections(recent_total, "total_")
            
            # Combine results with legacy keys pointing to open projections
            result = {}
            result.update(open_projections)
            result.update(total_projections)
            
            # Add legacy keys for backward compatibility (pointing to open projections)
            if 'open_current_daily_average' in open_projections:
                result.update({
                    'current_daily_average': open_projections['open_current_daily_average'],
                    'projected_daily_volume': open_projections['open_projected_daily_volume'],
                    'projected_weekly_volume': open_projections['open_projected_weekly_volume'],
                    'projected_monthly_volume': open_projections['open_projected_monthly_volume'],
                    'trend_confidence': open_projections['open_trend_confidence']
                })
            
            return result
            
        except Exception as e:
            return {'error': f'Error in projections: {e}'}

    def _fallback_velocity_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple velocity metrics used as a fallback when advanced method is missing."""
        try:
            if 'Created' not in df.columns:
                return {'error': 'Created column required for velocity metrics'}

            df_copy = df.copy()
            df_copy['Created'] = pd.to_datetime(df_copy['Created'], errors='coerce')
            valid = df_copy.dropna(subset=['Created'])
            if len(valid) == 0:
                return {'error': 'No valid created dates for velocity metrics'}

            # OPEN incidents subset (business request: base metrics on open orders)
            open_valid = self._get_open_incidents(valid)
            if 'Created' in open_valid.columns:
                open_valid = open_valid.dropna(subset=['Created'])
            else:
                open_valid = valid  # fallback

            # Calculate daily counts and simple rolling averages
            total_daily = valid.groupby(valid['Created'].dt.date).size()
            open_daily = open_valid.groupby(open_valid['Created'].dt.date).size()

            def _avg(series, n=None):
                if n is None:
                    return float(series.mean()) if len(series) > 0 else 0.0
                tail = series.tail(n)
                return float(tail.mean()) if len(tail) > 0 else 0.0

            total_daily_mean = _avg(total_daily)
            open_daily_mean = _avg(open_daily)

            metrics = {
                # Open-order focused metrics (primary)
                'open_daily_average': open_daily_mean,
                'open_last_7_days_average': _avg(open_daily, 7),
                'open_last_28_days_average': _avg(open_daily, 28),
                'open_most_recent_day': int(open_daily.tail(1).iloc[0]) if len(open_daily) > 0 else 0,
                'open_incident_count': int(len(open_valid)),
                # Legacy key for backward compatibility
                'daily_average': open_daily_mean,
                # Reference total creation velocity (historical context)
                'total_daily_average': total_daily_mean,
                'total_last_7_days_average': _avg(total_daily, 7),
                'total_last_28_days_average': _avg(total_daily, 28),
                'total_most_recent_day': int(total_daily.tail(1).iloc[0]) if len(total_daily) > 0 else 0
            }
            return metrics
        except Exception as e:
            return {'error': f'Fallback velocity error: {e}'}

    def _fallback_weekly_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple week-over-week comparison fallback."""
        try:
            if 'Created' not in df.columns:
                return {'error': 'Created column required for weekly comparison'}

            df_copy = df.copy()
            df_copy['Created'] = pd.to_datetime(df_copy['Created'], errors='coerce')
            valid = df_copy.dropna(subset=['Created'])
            if len(valid) == 0:
                return {'error': 'No valid dates for weekly comparison'}
            # OPEN incidents subset
            open_valid = self._get_open_incidents(valid)
            if 'Created' in open_valid.columns:
                open_valid = open_valid.dropna(subset=['Created'])
            else:
                open_valid = valid

            # Group by ISO week (historical total vs current open based on creation week)
            total_weekly_counts = valid.groupby(valid['Created'].dt.isocalendar().week).size()
            open_weekly_counts = open_valid.groupby(open_valid['Created'].dt.isocalendar().week).size()

            # Current week data (additional metrics requested)
            current_week = pd.Timestamp.now().isocalendar().week
            current_week_start = pd.Timestamp.now().replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=pd.Timestamp.now().weekday())
            
            # Current week incidents (created this week)
            current_week_total = len(valid[valid['Created'] >= current_week_start])
            current_week_open = len(open_valid[open_valid['Created'] >= current_week_start]) if len(open_valid) > 0 else 0
            
            # Open incidents this week (regardless of creation date)
            open_this_week_total = len(self._get_open_incidents(valid))

            if len(total_weekly_counts) < 2 or len(open_weekly_counts) < 2:
                return {
                    'error': 'Insufficient weeks for comparison',
                    'weeks_analyzed': int(min(len(total_weekly_counts), len(open_weekly_counts))),
                    'current_week_created': current_week_total,
                    'current_week_open_created': current_week_open,
                    'total_open_this_week': open_this_week_total
                }

            total_last = int(total_weekly_counts.iloc[-1])
            total_prev = int(total_weekly_counts.iloc[-2]) if len(total_weekly_counts) >= 2 else 0
            open_last = int(open_weekly_counts.iloc[-1])
            open_prev = int(open_weekly_counts.iloc[-2]) if len(open_weekly_counts) >= 2 else 0

            def pct(new, old):
                if old <= 0:
                    return None
                return float((new - old) / old)

            return {
                'weeks_analyzed': int(len(total_weekly_counts)),
                # Current week metrics (new addition)
                'current_week_created': current_week_total,
                'current_week_open_created': current_week_open,
                'total_open_this_week': open_this_week_total,
                # Open-order focused metrics (primary)
                'open_last_week_count': open_last,
                'open_previous_week_count': open_prev,
                'open_week_over_week_pct_change': pct(open_last, open_prev),
                # Legacy keys for backward compatibility
                'last_week_count': open_last,
                'previous_week_count': open_prev,
                'week_over_week_pct_change': pct(open_last, open_prev),
                # Historical totals for reference
                'total_last_week_count': total_last,
                'total_previous_week_count': total_prev,
                'total_week_over_week_pct_change': pct(total_last, total_prev)
            }
        except Exception as e:
            return {'error': f'Fallback weekly comparison error: {e}'}
    
    def get_capacity_recommendations(self) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        if not self.metrics:
            return ["Run metrics calculation first"]
        
        # Check capacity metrics
        capacity = self.metrics.get('capacity_metrics', {})
        
        if 'high_priority_load' in capacity:
            if capacity.get('priority_ratio', 0) > 0.3:  # More than 30% high priority
                recommendations.append("High priority incident load is elevated - consider additional resources")
        
        if 'closure_rate' in capacity:
            if capacity.get('closure_rate', 0) < 0.8:  # Less than 80% closure rate
                recommendations.append("Low closure rate detected - review resolution processes")
        
        # Check trend metrics
        trends = self.metrics.get('trend_metrics', {})
        if trends.get('trend_direction') == 'increasing':
            recommendations.append("Incident volume trending upward - plan for increased capacity")
        
        # Check performance metrics
        performance = self.metrics.get('performance_metrics', {})
        if performance.get('high_priority_sla_compliance', 1.0) < 0.8:
            recommendations.append("High priority SLA compliance below 80% - review escalation processes")
        
        return recommendations if recommendations else ["Current capacity appears adequate"]
    
    def generate_metrics_summary(self) -> Dict[str, Any]:
        """Generate a summary of key metrics."""
        if not self.metrics:
            return {'error': 'No metrics calculated yet'}
        
        summary = {
            'total_incidents': self.metrics.get('volume_metrics', {}).get('total_incidents', 0),
            'daily_average': self.metrics.get('volume_metrics', {}).get('daily_average', 0),
            'resolution_rate': self.metrics.get('performance_metrics', {}).get('resolution_rate', 0),
            'trend_direction': self.metrics.get('trend_metrics', {}).get('trend_direction', 'unknown'),
            'recommendations': self.get_capacity_recommendations()
        }
        
        return summary

    def _fallback_process_analysis_results(self, analysis_results: Any) -> Dict[str, Any]:
        """Fallback summarizer for analysis_results returned by other analyzers."""
        try:
            if not analysis_results:
                return {'note': 'No analysis results provided'}

            # If it's a dict, extract top-level keys and short summaries
            if isinstance(analysis_results, dict):
                summary = {'keys_present': list(analysis_results.keys())}
                # Provide counts if any values are lists or dataframes
                for k, v in analysis_results.items():
                    if hasattr(v, '__len__') and not isinstance(v, (str, bytes, dict)):
                        try:
                            summary[f'{k}_count'] = len(v)
                        except Exception:
                            pass
                return summary

            # For other types, return a simple string representation
            return {'summary': str(analysis_results)[:200]}
        except Exception as e:
            return {'error': f'Fallback analysis processing error: {e}'}

    def _fallback_process_health_score(self, health_score_results: Any) -> Dict[str, Any]:
        """Fallback processing for health score inputs."""
        try:
            if not health_score_results:
                return {'note': 'No health score provided'}

            if isinstance(health_score_results, dict):
                # If there's a numeric score, return it
                if 'score' in health_score_results:
                    return {'health_score': health_score_results.get('score')}
                # Otherwise return keys
                return {'health_keys': list(health_score_results.keys())}

            # If it's a numeric value
            if isinstance(health_score_results, (int, float)):
                return {'health_score': float(health_score_results)}

            return {'health_summary': str(health_score_results)[:200]}
        except Exception as e:
            return {'error': f'Fallback health processing error: {e}'}

    def _calculate_enhanced_priority_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced priority metrics with targets and trends."""
        try:
            # Get basic priority breakdown
            priority_breakdown = self.calculate_priority_breakdown(df)
            
            # Calculate open incidents by priority (assuming unresolved = no resolved_date)
            open_incidents = df[df['resolved_date'].isna()] if 'resolved_date' in df.columns else df
            
            open_priority = self.calculate_priority_breakdown(open_incidents)
            
            # Set targets based on business rules (you can adjust these)
            targets = {
                'high_priority_target': 30,    # Target for high priority incidents
                'medium_priority_target': 150, # Target for medium priority incidents
                'required_daily_closure_rate': 23  # Required daily closure rate
            }
            
            # Calculate individual and total target gaps
            high_gap = open_priority.get('High', 0) - targets['high_priority_target']
            medium_gap = open_priority.get('Medium', 0) - targets['medium_priority_target']
            total_gap = high_gap + medium_gap
            
            return {
                'total_open_high': open_priority.get('High', 0),
                'total_open_medium': open_priority.get('Medium', 0),
                'total_open_low': open_priority.get('Low', 0),
                'high_priority_target': targets['high_priority_target'],
                'medium_priority_target': targets['medium_priority_target'],
                'required_daily_rate': targets['required_daily_closure_rate'],
                'high_vs_target': high_gap,
                'medium_vs_target': medium_gap,
                'total_target_gap': total_gap
            }
        except Exception as e:
            self.logger.error(f"Error calculating enhanced priority metrics: {e}")
            return {}

    def _calculate_backlog_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate backlog and week-over-week comparison metrics."""
        try:
            # Get current date and calculate week boundaries
            current_date = pd.Timestamp.now()
            this_week_start = current_date - pd.Timedelta(days=current_date.dayofweek)
            last_week_start = this_week_start - pd.Timedelta(weeks=1)
            last_week_end = this_week_start - pd.Timedelta(days=1)
            
            # Filter for incidents created in different time periods
            if 'created_date' in df.columns:
                df_copy = df.copy()
                df_copy['created_date'] = pd.to_datetime(df_copy['created_date'], errors='coerce')
                
                # This week incidents
                this_week_incidents = df_copy[
                    df_copy['created_date'] >= this_week_start
                ]
                
                # Last week incidents  
                last_week_incidents = df_copy[
                    (df_copy['created_date'] >= last_week_start) & 
                    (df_copy['created_date'] <= last_week_end)
                ]
                
                # Calculate backlog (open incidents)
                open_condition = df_copy['resolved_date'].isna() if 'resolved_date' in df_copy.columns else True
                current_backlog = len(df_copy[open_condition])
                
                this_week_count = len(this_week_incidents)
                last_week_count = len(last_week_incidents)
                net_change = this_week_count - last_week_count
                
                return {
                    'current_backlog': current_backlog,
                    'this_week_created': this_week_count,
                    'last_week_created': last_week_count,
                    'net_change': net_change,
                    'week_over_week_pct': (net_change / max(last_week_count, 1)) * 100
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating backlog metrics: {e}")
            
        # Fallback calculation
        return {
            'current_backlog': len(df),
            'this_week_created': 0,
            'last_week_created': 0,
            'net_change': 0,
            'week_over_week_pct': 0
        }

    def _calculate_target_progress_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate progress toward targets and daily rates."""
        try:
            # Calculate actual daily closure rate
            if 'resolved_date' in df.columns:
                df_copy = df.copy()
                df_copy['resolved_date'] = pd.to_datetime(df_copy['resolved_date'], errors='coerce')
                
                # Recent resolved incidents (last 30 days)
                recent_date = pd.Timestamp.now() - pd.Timedelta(days=30)
                recent_resolved = df_copy[
                    (df_copy['resolved_date'] >= recent_date) & 
                    (df_copy['resolved_date'].notna())
                ]
                
                actual_daily_rate = len(recent_resolved) / 30 if len(recent_resolved) > 0 else 0
            else:
                actual_daily_rate = 0
                
            required_rate = 23  # From your dashboard target - this is backlog clearance only, not including new tickets
            
            return {
                'required_daily_rate': required_rate,
                'required_daily_rate_note': 'Backlog clearance target (excludes projected new tickets)',
                'actual_daily_rate': round(actual_daily_rate, 1),
                'daily_rate_gap': required_rate - actual_daily_rate,
                'rate_performance_pct': (actual_daily_rate / required_rate) * 100 if required_rate > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error calculating target progress metrics: {e}")
            return {'required_daily_rate': 23, 'actual_daily_rate': 0}

    def _calculate_workstream_breakdown(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate workstream breakdown by SAP components."""
        try:
            workstreams = {}
            
            # Look for workstream indicators in various columns
            workstream_columns = ['category', 'subcategory', 'assignment_group', 'cmdb_ci']
            
            for col in workstream_columns:
                if col in df.columns:
                    # Extract SAP workstream patterns
                    df_copy = df.copy()
                    df_copy[col] = df_copy[col].fillna('')
                    
                    # Define SAP workstream patterns based on common naming
                    workstream_patterns = {
                        'SAP-S4': df_copy[col].str.contains('S4|S/4', case=False, na=False),
                        'SAP-ECC': df_copy[col].str.contains('ECC|R/3', case=False, na=False),
                        'SAP-BW': df_copy[col].str.contains('BW|Business.*Warehouse', case=False, na=False),
                        'SAP-PI': df_copy[col].str.contains('PI|Process.*Integration', case=False, na=False),
                        'SAP-HANA': df_copy[col].str.contains('HANA', case=False, na=False),
                        'SAP-Basis': df_copy[col].str.contains('Basis|Infrastructure', case=False, na=False),
                        'SAP-Security': df_copy[col].str.contains('Security|Authorization', case=False, na=False),
                        'SAP-Fiori': df_copy[col].str.contains('Fiori|UI5', case=False, na=False)
                    }
                    
                    for workstream, mask in workstream_patterns.items():
                        workstream_data = df_copy[mask]
                        if len(workstream_data) > 0:
                            # OPEN subset
                            open_subset = self._get_open_incidents(workstream_data)
                            priority_total = self.calculate_priority_breakdown(workstream_data)
                            priority_open = self.calculate_priority_breakdown(open_subset)

                            workstreams[workstream] = {
                                'total_incidents': len(workstream_data),
                                'open_backlog': len(open_subset),
                                # Total priority distribution
                                'high_priority_total': priority_total.get('High', 0),
                                'medium_priority_total': priority_total.get('Medium', 0),
                                'low_priority_total': priority_total.get('Low', 0),
                                # Open priority distribution (requested focus)
                                'high_priority_open': priority_open.get('High', 0),
                                'medium_priority_open': priority_open.get('Medium', 0),
                                'low_priority_open': priority_open.get('Low', 0),
                                'completion_rate': ((len(workstream_data) - len(open_subset)) / max(len(workstream_data), 1)) * 100
                            }
                    break  # Use first available column
            
            return workstreams
            
        except Exception as e:
            self.logger.error(f"Error calculating workstream breakdown: {e}")
            return {}


# Alias for backward compatibility
MetricsCalculator = IncidentMetricsCalculator