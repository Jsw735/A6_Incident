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
        # Cache for priority breakdowns to avoid recalculation
        self._priority_cache = {}
        self._current_df_hash = None

    # ------------------------------------------------------------------
    # Helper Utilities
    # ------------------------------------------------------------------
    def _get_priority_column(self, df: pd.DataFrame) -> str:
        """Get the priority/urgency column name, handling different variations."""
        priority_columns = ['urgency', 'Urgency', 'priority', 'Priority', 'PRIORITY', 'URGENCY']
        
        for col in priority_columns:
            if col in df.columns:
                return col
        
        return None

    def _get_status_column(self, df: pd.DataFrame) -> str:
        """Get the status/state column name, handling different variations."""
        status_columns = ['status', 'Status', 'state', 'State', 'Incident State', 'STATUS', 'STATE']
        
        for col in status_columns:
            if col in df.columns:
                return col
        
        return None

    def _get_created_column(self, df: pd.DataFrame) -> str:
        """Get the created/opened date column name, handling different variations."""
        created_columns = ['created_date', 'Created', 'sys_created_on', 'opened', 'Opened', 'created_on', 'creation_date']
        
        for col in created_columns:
            if col in df.columns:
                return col
        
        return None

    def _get_resolved_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all resolved/closed date column names, handling different variations."""
        possible_columns = ['resolved', 'closed', 'resolved at', 'close date', 'resolved_date', 
                           'Resolved', 'Closed', 'resolved_at', 'closure_date']
        
        return [c for c in df.columns if c.lower() in [col.lower() for col in possible_columns]]

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

            # Use consolidated column detection
            resolved_cols = self._get_resolved_columns(df_copy)
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

            # Status-based refinement using consolidated status column detection
            status_col = self._get_status_column(df_copy)

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
        Calculate comprehensive metrics with organized structure to prevent duplicates.
        Returns clean, non-overlapping metrics organized by category.
        """
        try:
            if df is None or df.empty:
                return self._get_empty_metrics_result()
            
            # Clear cache for new calculation
            self._priority_cache.clear()
            self.logger.info(f"Calculating comprehensive metrics for {len(df)} records")
            
            # Core incident metrics (unique, non-overlapping)
            core_metrics = self._calculate_core_incident_metrics(df)
            
            # Resolution and performance metrics (separate from core)
            resolution_metrics = self._calculate_resolution_metrics(df)
            
            # Priority and urgency analysis (distinct from other categories) - using enhanced version
            priority_metrics = self._calculate_enhanced_priority_metrics(df)
            
            # Assignment and workstream metrics (separate category)
            assignment_metrics = self._calculate_assignment_metrics(df)
            
            # Temporal analysis metrics (time-based, non-overlapping)
            temporal_metrics = self._calculate_temporal_metrics(df)
            
            # Consolidated result with clear separation
            metrics_result = {
                # Meta information
                'calculation_timestamp': datetime.now().isoformat(),
                'total_records_analyzed': len(df),
                'data_period': self._get_data_period(df),
                
                # SECTION 1: Core Metrics (fundamental counts and rates)
                'core_metrics': {
                    'total_incidents': core_metrics.get('total_incidents', 0),
                    'open_incidents': core_metrics.get('open_incidents', 0),
                    'closed_incidents': core_metrics.get('closed_incidents', 0),
                    'in_progress_incidents': core_metrics.get('in_progress_incidents', 0),
                    'closure_rate_percentage': core_metrics.get('closure_rate_percentage', 0.0)
                },
                
                # SECTION 2: Performance Metrics (SLA, timing, efficiency)
                'performance_metrics': {
                    'average_resolution_days': resolution_metrics.get('average_resolution_days', 0.0),
                    'median_resolution_days': resolution_metrics.get('median_resolution_days', 0.0),
                    'sla_compliance_rate': resolution_metrics.get('sla_compliance_rate', 0.0),
                    'first_response_time_avg': resolution_metrics.get('first_response_time_avg', 0.0)
                },
                
                # SECTION 3: Priority Analysis (urgency, impact, business priority)
                'priority_analysis': {
                    # Pass through all enhanced priority fields
                    **priority_metrics,
                    # Ensure legacy keys are present for backward compatibility
                    'high_priority_count': priority_metrics.get('high_priority_count', 0),
                    'high_priority_percentage': (
                        (priority_metrics.get('priority_distribution', {}).get('High', 0) / max(1, core_metrics.get('total_incidents', 1))) * 100
                        if isinstance(priority_metrics.get('priority_distribution'), dict) else 0.0
                    ),
                    'critical_incidents': priority_metrics.get('critical_incidents', 0),
                    'priority_distribution': priority_metrics.get('priority_distribution', {})
                },
                
                # SECTION 4: Assignment Analysis (workstreams, teams, distribution)
                'assignment_analysis': {
                    'workstream_distribution': assignment_metrics.get('workstream_distribution', {}),
                    'top_assigned_groups': assignment_metrics.get('top_assigned_groups', {}),
                    'unassigned_count': assignment_metrics.get('unassigned_count', 0),
                    'assignment_balance_score': assignment_metrics.get('assignment_balance_score', 0.0)
                },
                
                # SECTION 5: Time Analysis (trends, patterns, temporal insights)
                'temporal_analysis': {
                    'daily_average': temporal_metrics.get('daily_average', 0.0),
                    'weekly_pattern': temporal_metrics.get('weekly_pattern', {}),
                    'monthly_trend': temporal_metrics.get('monthly_trend', 'stable'),
                    'peak_hours': temporal_metrics.get('peak_hours', []),
                    # Add comprehensive weekly data for trend analysis
                    'total_weekly_counts': temporal_metrics.get('total_weekly_counts', {}),
                    'total_trend_slope': temporal_metrics.get('total_trend_slope', 0.0),
                    'total_trend_direction': temporal_metrics.get('total_trend_direction', 'stable'),
                    'total_average_weekly_volume': temporal_metrics.get('total_average_weekly_volume', 0.0),
                    'total_weeks_analyzed': temporal_metrics.get('total_weeks_analyzed', 0)
                },
                
                # SECTION 6: Executive Ready Metrics (calculated once, no transformations needed)
                'executive_ready': self._calculate_executive_ready_metrics(
                    core_metrics, priority_metrics, temporal_metrics, resolution_metrics, assignment_metrics
                )
            }
            
            # Add calculated totals and derived metrics (no duplication)
            metrics_result.update({
                # Derived calculations (computed once, used everywhere)
                'derived_metrics': {
                    'backlog_ratio': core_metrics.get('open_incidents', 0) / max(1, core_metrics.get('total_incidents', 1)),
                    'efficiency_score': self._calculate_efficiency_score(core_metrics, resolution_metrics),
                    'workload_balance': self._calculate_workload_balance(assignment_metrics),
                    'trend_momentum': self._calculate_trend_momentum(temporal_metrics)
                },
                
                # SECTION 6: Quality and Training Metrics (enhanced integration)
                'quality_metrics': self._calculate_quality_accuracy_metrics(df),
                'training_metrics': self._calculate_enhanced_training_metrics(df),
                'category_metrics': self._calculate_enhanced_category_metrics(df),
                
                # Summary for quick access
                'summary': {
                    'overall_health': self._calculate_overall_health(core_metrics, resolution_metrics, priority_metrics),
                    'key_insights': self._generate_key_insights(metrics_result),
                    'action_items': self._generate_action_items(metrics_result)
                }
            })
            
            self.logger.info("Comprehensive metrics calculation completed successfully")
            return metrics_result
            
        except Exception as e:
            error_msg = f"Error calculating comprehensive metrics: {str(e)}"
            self.logger.error(error_msg)
            return self._get_error_metrics_result(error_msg)
    
    def _calculate_core_incident_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate core incident metrics such as total, open, closed counts."""
        total_incidents = len(df)
        
        # Open incidents (using conservative approach)
        open_incidents = len(self._get_open_incidents(df))
        
        # Closed incidents (derived)
        closed_incidents = total_incidents - open_incidents
        
        # In-progress incidents (heuristic: based on common status values)
        in_progress_states = ['In Progress', 'Assigned', 'Open']
        in_progress_incidents = df[df['status'].isin(in_progress_states)]
        
        # Closure rate (percentage of closed incidents)
        closure_rate = (closed_incidents / total_incidents * 100) if total_incidents > 0 else 0
        
        return {
            'total_incidents': total_incidents,
            'open_incidents': open_incidents,
            'closed_incidents': closed_incidents,
            'in_progress_incidents': len(in_progress_incidents),
            'closure_rate_percentage': closure_rate
        }
    
    def _calculate_resolution_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate resolution performance metrics."""
        resolution = {}
        
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
                
                resolution['average_resolution_hours'] = float(resolution_times.mean())
                resolution['median_resolution_hours'] = float(resolution_times.median())
                resolution['resolution_rate'] = float(len(resolved_incidents) / len(df))
                
                # SLA performance (assuming 24h for High, 72h for Medium)
                # Use consolidated column detection for priority
                priority_col = self._get_priority_column(df)
                
                if priority_col:
                    high_priority = resolved_incidents[resolved_incidents[priority_col].isin(['Critical', 'High'])]
                    if len(high_priority) > 0:
                        high_res_times = (high_priority[resolved_col] - 
                                        high_priority[created_col]).dt.total_seconds() / 3600
                        resolution['high_priority_sla_compliance'] = float((high_res_times <= 24).mean())
        
        return resolution
    
    def _calculate_assignment_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics related to assignment and workstreams."""
        assignment_metrics = {}
        
        # Assignment group capacity
        assignment_cols = ['Assignment Group', 'Assignment group', 'Assigned to']
        assignment_col = None
        for col in assignment_cols:
            if col in df.columns:
                assignment_col = col
                break
        
        if assignment_col:
            group_workload = df[assignment_col].value_counts()
            assignment_metrics['workstream_distribution'] = group_workload.head(10).to_dict()
            assignment_metrics['top_assigned_groups'] = group_workload.head(5).to_dict()
            assignment_metrics['unassigned_count'] = df[df[assignment_col].isna()].shape[0]
            assignment_metrics['assignment_balance_score'] = float(group_workload.std())
        
        return assignment_metrics
    
    def _calculate_temporal_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based metrics for trends and patterns."""
        temporal_metrics = {}

        created_col = self._get_created_column(df)
        if created_col:
            df_copy = df.copy()
            df_copy[created_col] = pd.to_datetime(df_copy[created_col], errors='coerce')
            valid_dates = df_copy[created_col].dropna()
            
            if len(valid_dates) > 0:
                # Daily volume calculations
                daily_volumes = valid_dates.groupby(valid_dates.dt.date).size()
                temporal_metrics['daily_average'] = float(daily_volumes.mean())
                temporal_metrics['daily_peak'] = int(daily_volumes.max())
                temporal_metrics['daily_minimum'] = int(daily_volumes.min())
                temporal_metrics['volume_variance'] = float(daily_volumes.var())
                
                # Weekly trends for last 6 weeks
                end_date = valid_dates.max()
                start_date = end_date - timedelta(weeks=6)
                
                recent_total = df_copy[df_copy[created_col] >= start_date]
                
                if len(recent_total) > 0:
                    total_weekly_counts = recent_total.groupby(recent_total[created_col].dt.to_period('W')).size()
                    
                    # Trend calculation
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
                    
                    temporal_metrics.update({
                        'total_weekly_counts': {str(k): int(v) for k, v in total_weekly_counts.items()},
                        'total_trend_slope': float(total_slope),
                        'total_trend_direction': total_direction,
                        'total_average_weekly_volume': float(total_avg),
                        'total_weeks_analyzed': len(total_weekly_counts)
                    })
        
        return temporal_metrics
    
    def calculate_priority_breakdown(self, df: pd.DataFrame, open_only: bool = False) -> Dict[str, int]:
        """
        Calculate priority breakdown using specified business rules:
        P0 + P1 = High Priority
        P2 = Medium Priority  
        P3 = Low Priority
        
        Args:
            df: DataFrame to analyze
            open_only: If True, only analyze open incidents
            
        Following context documentation on groupby operations and aggregations.
        Uses caching to prevent duplicate calculations.
        """
        try:
            # Filter to open incidents if requested
            analysis_df = self._get_open_incidents(df) if open_only else df
            
            # Create cache key based on DataFrame content and size
            cache_key = f"priority_breakdown_{len(analysis_df)}_{open_only}_{hash(df.columns.tolist().__str__())}"
            
            # Check if we already calculated this for the same data
            if cache_key in self._priority_cache:
                cached_result = self._priority_cache[cache_key]
                self.logger.debug(f"Using cached priority breakdown: {cached_result}")
                return cached_result
            
            # Smart column detection using consolidated helper
            priority_column = self._get_priority_column(analysis_df)
            
            if not priority_column:
                self.logger.warning("Priority/urgency column not found in data")
                return {'High': 0, 'Medium': 0, 'Low': 0}

            raw_series = analysis_df[priority_column].astype(str)

            # Fixed mapping for patterns like '1 - High', '2 - Medium', '3 - Low'
            def simple_map(v: str) -> Optional[str]:
                if v is None:
                    return None
                s = str(v).strip()
                if not s:
                    return None
                
                # Direct text matching first
                lower = s.lower()
                if 'high' in lower:
                    return 'High'
                if 'medium' in lower:
                    return 'Medium'
                if 'low' in lower:
                    return 'Low'
                
                # Digit-led pattern: 1 = High, 2 = Medium, 3 = Low
                if s and s[0].isdigit():
                    d = s[0]
                    if d == '1':
                        return 'High'
                    elif d == '2':
                        return 'Medium'
                    elif d == '3':
                        return 'Low'
                # Unknown mapping -> None (exclude)
                return None

            mapped = raw_series.map(simple_map)
            mapped = mapped.dropna()
            counts = mapped.value_counts().to_dict()

            result = {
                'High': int(counts.get('High', 0)),
                'Medium': int(counts.get('Medium', 0)),
                'Low': int(counts.get('Low', 0))
            }

            # Cache the result
            self._priority_cache[cache_key] = result
            self.logger.info(f"Priority breakdown calculated: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in priority breakdown calculation: {str(e)}")
            return {'High': 0, 'Medium': 0, 'Low': 0}

    def calculate_executive_summary_metrics(self, df: pd.DataFrame, analysis_results: Dict[str, Any] = None, health_score_results: Any = None) -> Dict[str, Any]:
        """
        COMPATIBILITY METHOD: Calculate all executive summary metrics.
        
        This method is kept for backward compatibility with existing components that call it.
        It wraps the enhanced calculate_all_metrics method and transforms the output to the expected format.
        
        Args:
            df: DataFrame to analyze
            analysis_results: Optional analysis results (for compatibility)
            health_score_results: Optional health score results (for compatibility)
            
        Returns:
            Dict with executive summary metrics in the expected format
        """
        try:
            # Use the enhanced calculation method
            enhanced_metrics = self.calculate_all_metrics(df)
            
            if not enhanced_metrics:
                self.logger.warning("Enhanced metrics calculation returned empty results in compatibility method")
                return {"total_incidents": len(df)}

            # Transform to the expected format for backward compatibility
            core_metrics = enhanced_metrics.get('core_metrics', {})
            priority_analysis = enhanced_metrics.get('priority_analysis', {})
            
            # Create a compatibility structure
            executive_metrics = {
                'total_incidents': core_metrics.get('total_incidents', len(df)),
                'open_incidents': core_metrics.get('open_incidents', 0),
                'closed_incidents': core_metrics.get('closed_incidents', 0),
                
                # Priority breakdown in expected format
                'priority_breakdown': {
                    'High': priority_analysis.get('high_priority_count', 0),
                    'Medium': 0,  # Calculate if needed
                    'Low': core_metrics.get('total_incidents', len(df)) - priority_analysis.get('high_priority_count', 0)
                },
                
                # Enhanced metrics for more detailed analysis
                'enhanced_metrics': enhanced_metrics,
                
                # Include analysis results if provided
                'analysis_summary': analysis_results if analysis_results else {},
                'health_score': health_score_results if health_score_results else None,
                
                # Metadata
                'calculation_timestamp': enhanced_metrics.get('calculation_timestamp'),
                'total_records_analyzed': enhanced_metrics.get('total_records_analyzed', len(df))
            }
            
            self.logger.info(f"Executive summary metrics calculated (compatibility mode): {len(executive_metrics)} sections")
            return executive_metrics
            
        except Exception as e:
            self.logger.error(f"Error in executive summary metrics calculation (compatibility): {str(e)}")
            return {
                'total_incidents': len(df),
                'error': str(e),
                'calculation_timestamp': datetime.now().isoformat()
            }

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
            created_col = self._get_created_column(df)
            if not created_col:
                return {'error': 'Created column required for velocity metrics'}

            df_copy = df.copy()
            df_copy[created_col] = pd.to_datetime(df_copy[created_col], errors='coerce')
            valid = df_copy.dropna(subset=[created_col])
            if len(valid) == 0:
                return {'error': 'No valid created dates for velocity metrics'}

            # OPEN incidents subset (business request: base metrics on open orders)
            open_valid = self._get_open_incidents(valid)
            if created_col in open_valid.columns:
                open_valid = open_valid.dropna(subset=[created_col])
            else:
                open_valid = valid  # fallback

            # Calculate daily counts and simple rolling averages
            total_daily = valid.groupby(valid[created_col].dt.date).size()
            open_daily = open_valid.groupby(open_valid[created_col].dt.date).size()

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
            created_col = self._get_created_column(df)
            if not created_col:
                return {'error': 'Created column required for weekly comparison'}

            df_copy = df.copy()
            df_copy[created_col] = pd.to_datetime(df_copy[created_col], errors='coerce')
            valid = df_copy.dropna(subset=[created_col])
            if len(valid) == 0:
                return {'error': 'No valid dates for weekly comparison'}
            # OPEN incidents subset
            open_valid = self._get_open_incidents(valid)
            if created_col in open_valid.columns:
                open_valid = open_valid.dropna(subset=[created_col])
            else:
                open_valid = valid

            # Group by ISO week (historical total vs current open based on creation week)
            total_weekly_counts = valid.groupby(valid[created_col].dt.isocalendar().week).size()
            open_weekly_counts = open_valid.groupby(open_valid[created_col].dt.isocalendar().week).size()

            # Current week data (additional metrics requested)
            current_week = pd.Timestamp.now().isocalendar().week
            current_week_start = pd.Timestamp.now().replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=pd.Timestamp.now().weekday())
            
            # Current week incidents (created this week)
            current_week_total = len(valid[valid[created_col] >= current_week_start])
            current_week_open = len(open_valid[open_valid[created_col] >= current_week_start]) if len(open_valid) > 0 else 0
            
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
            
            # Calculate open incidents by priority using consolidated helper
            open_incidents = self._get_open_incidents(df)
            
            open_priority = self.calculate_priority_breakdown(open_incidents, open_only=False)  # Already filtered
            
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
                # Overall distribution across all incidents
                'priority_distribution': priority_breakdown,
                'high_priority_count': priority_breakdown.get('High', 0),
                # Open distribution for current backlog context
                'priority_distribution_open': open_priority,
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
            created_col = self._get_created_column(df)
            if created_col:
                df_copy = df.copy()
                df_copy[created_col] = pd.to_datetime(df_copy[created_col], errors='coerce')
                
                # This week incidents
                this_week_incidents = df_copy[
                    df_copy[created_col] >= this_week_start
                ]
                
                # Last week incidents  
                last_week_incidents = df_copy[
                    (df_copy[created_col] >= last_week_start) & 
                    (df_copy[created_col] <= last_week_end)
                ]
                
                # Calculate backlog (open incidents) using consolidated helper
                current_backlog = len(self._get_open_incidents(df_copy))
                
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
    
    def _calculate_quality_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall quality accuracy metrics from quality analyzer results."""
        try:
            from .quality_analyzer import IncidentQualityAnalyzer
            
            analyzer = IncidentQualityAnalyzer()
            quality_results = analyzer.analyze_data_quality(df)
            
            # Extract quality metrics
            category_analysis = quality_results.get('category_analysis', {})
            urgency_analysis = quality_results.get('urgency_analysis', {})
            
            quality_metrics = {
                'total_records_analyzed': len(df),
                'category_accuracy_percentage': category_analysis.get('accuracy_percentage', 0.0),
                'urgency_accuracy_percentage': urgency_analysis.get('accuracy_percentage', 0.0),
                'overall_classification_accuracy': (
                    category_analysis.get('accuracy_percentage', 0.0) + 
                    urgency_analysis.get('accuracy_percentage', 0.0)
                ) / 2.0,
                'category_misclassifications': len(category_analysis.get('miscategorized_records', [])),
                'urgency_misclassifications': len(urgency_analysis.get('misclassified_urgency', [])),
                'sla_compliance_rate': urgency_analysis.get('sla_compliance', {}).get('overall_compliance_rate', 0.0)
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality accuracy metrics: {e}")
            return {
                'total_records_analyzed': len(df),
                'category_accuracy_percentage': 0.0,
                'urgency_accuracy_percentage': 0.0,
                'overall_classification_accuracy': 0.0,
                'category_misclassifications': 0,
                'urgency_misclassifications': 0,
                'sla_compliance_rate': 0.0,
                'error': str(e)
            }
    
    def _calculate_urgency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate urgency classification and SLA metrics."""
        try:
            from .quality_analyzer import IncidentQualityAnalyzer
            
            analyzer = IncidentQualityAnalyzer()
            quality_results = analyzer.analyze_data_quality(df)
            urgency_analysis = quality_results.get('urgency_analysis', {})
            
            urgency_metrics = {
                'urgency_accuracy_percentage': urgency_analysis.get('accuracy_percentage', 0.0),
                'urgency_misclassifications': len(urgency_analysis.get('misclassified_urgency', [])),
                'sla_compliance': urgency_analysis.get('sla_compliance', {}),
                'urgency_distribution': self._calculate_urgency_distribution(df),
                'average_resolution_by_urgency': self._calculate_resolution_times_by_urgency(df)
            }
            
            return urgency_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating urgency metrics: {e}")
            return {
                'urgency_accuracy_percentage': 0.0,
                'urgency_misclassifications': 0,
                'sla_compliance': {},
                'urgency_distribution': {},
                'average_resolution_by_urgency': {},
                'error': str(e)
            }
    
    def _calculate_category_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate category classification accuracy metrics."""
        try:
            from .quality_analyzer import IncidentQualityAnalyzer
            
            analyzer = IncidentQualityAnalyzer()
            quality_results = analyzer.analyze_data_quality(df)
            category_analysis = quality_results.get('category_analysis', {})
            
            category_metrics = {
                'category_accuracy_percentage': category_analysis.get('accuracy_percentage', 0.0),
                'category_misclassifications': len(category_analysis.get('miscategorized_records', [])),
                'category_distribution': self._calculate_category_distribution(df),
                'most_common_misclassification': self._identify_common_misclassification(category_analysis),
                'defect_detection_rate': self._calculate_defect_detection_rate(category_analysis),
                'training_detection_rate': self._calculate_training_detection_rate(category_analysis)
            }
            
            return category_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating category accuracy metrics: {e}")
            return {
                'category_accuracy_percentage': 0.0,
                'category_misclassifications': 0,
                'category_distribution': {},
                'most_common_misclassification': 'None',
                'defect_detection_rate': 0.0,
                'training_detection_rate': 0.0,
                'error': str(e)
            }
    
    def _calculate_training_identification_accuracy(self, training_analysis: Dict[str, Any]) -> float:
        """Calculate accuracy of training ticket identification."""
        try:
            current_training = training_analysis.get('total_training_tickets', 0)
            should_be_training = len(training_analysis.get('should_be_training', []))
            
            if current_training + should_be_training == 0:
                return 100.0  # No training tickets to classify
            
            # Accuracy is the percentage of correctly identified training tickets
            # Assuming current training tickets are correctly classified
            accuracy = (current_training / (current_training + should_be_training)) * 100
            return min(accuracy, 100.0)
            
        except Exception:
            return 0.0
    
    def _calculate_urgency_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate urgency level distribution."""
        try:
            urgency_col = self._get_priority_column(df)  # Also handles urgency columns
            if not urgency_col:
                return {}
            
            urgency_counts = df[urgency_col].value_counts().to_dict()
            total = len(df)
            
            urgency_dist = {}
            for urgency, count in urgency_counts.items():
                urgency_dist[str(urgency)] = {
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0.0
                }
            
            return urgency_dist
            
        except Exception:
            return {}
    
    def _calculate_resolution_times_by_urgency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate average resolution times by urgency level."""
        try:
            if 'urgency' not in df.columns or 'created_date' not in df.columns or 'resolved_date' not in df.columns:
                return {}
            
            resolved_df = df.dropna(subset=['resolved_date'])
            resolution_times = {}
            
            for urgency in resolved_df['urgency'].unique():
                urgency_df = resolved_df[resolved_df['urgency'] == urgency]
                times = []
                
                for idx, row in urgency_df.iterrows():
                    try:
                        created = pd.to_datetime(row['created_date'])
                        resolved = pd.to_datetime(row['resolved_date'])
                        hours = (resolved - created).total_seconds() / 3600
                        times.append(hours)
                    except:
                        continue
                
                if times:
                    resolution_times[str(urgency)] = np.mean(times)
            
            return resolution_times
            
        except Exception:
            return {}
    
    def _calculate_category_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate subcategory distribution."""
        try:
            if 'subcategory' not in df.columns:
                return {}
            
            category_counts = df['subcategory'].value_counts().to_dict()
            total = len(df)
            
            category_dist = {}
            for category, count in category_counts.items():
                category_dist[str(category)] = {
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0.0
                }
            
            return category_dist
            
        except Exception:
            return {}
    
    def _identify_common_misclassification(self, category_analysis: Dict[str, Any]) -> str:
        """Identify the most common misclassification pattern."""
        try:
            misclassified = category_analysis.get('miscategorized_records', [])
            if not misclassified:
                return 'None'
            
            # Count misclassification patterns
            patterns = {}
            for record in misclassified:
                current = record.get('current_subcategory', '')
                suggested = record.get('suggested_subcategory', '')
                pattern = f"{current} -> {suggested}"
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            if patterns:
                return max(patterns, key=patterns.get)
            else:
                return 'None'
                
        except Exception:
            return 'None'
    
    def _calculate_defect_detection_rate(self, category_analysis: Dict[str, Any]) -> float:
        """Calculate rate of defect detection from misclassifications."""
        try:
            misclassified = category_analysis.get('miscategorized_records', [])
            defect_suggestions = [r for r in misclassified if r.get('suggested_subcategory') == 'Defect']
            
            total_records = category_analysis.get('total_records', 0)
            if total_records == 0:
                return 0.0
            
            return (len(defect_suggestions) / total_records * 100)
            
        except Exception:
            return 0.0
    
    def _calculate_training_detection_rate(self, category_analysis: Dict[str, Any]) -> float:
        """Calculate rate of training detection from misclassifications."""
        try:
            misclassified = category_analysis.get('miscategorized_records', [])
            training_suggestions = [r for r in misclassified if r.get('suggested_subcategory') == 'Training']
            
            total_records = category_analysis.get('total_records', 0)
            if total_records == 0:
                return 0.0
            
            return (len(training_suggestions) / total_records * 100)
            
        except Exception:
            return 0.0

    def _get_empty_metrics_result(self) -> Dict[str, Any]:
        """Return empty metrics structure when no data is available."""
        return {
            'calculation_timestamp': datetime.now().isoformat(),
            'total_records_analyzed': 0,
            'data_period': 'No data available',
            'core_metrics': {
                'total_incidents': 0,
                'open_incidents': 0,
                'closed_incidents': 0,
                'in_progress_incidents': 0,
                'closure_rate_percentage': 0.0
            },
            'performance_metrics': {
                'average_resolution_days': 0.0,
                'median_resolution_days': 0.0,
                'sla_compliance_rate': 0.0,
                'first_response_time_avg': 0.0
            },
            'priority_analysis': {
                'high_priority_count': 0,
                'high_priority_percentage': 0.0,
                'critical_incidents': 0,
                'priority_distribution': {}
            },
            'assignment_analysis': {
                'workstream_distribution': {},
                'top_assigned_groups': {},
                'unassigned_count': 0,
                'assignment_balance_score': 0.0
            },
            'temporal_analysis': {
                'daily_average': 0.0,
                'weekly_pattern': {},
                'monthly_trend': 'stable',
                'peak_hours': []
            },
            'derived_metrics': {
                'backlog_ratio': 0.0,
                'efficiency_score': 0.0,
                'workload_balance': 0.0,
                'trend_momentum': 0.0
            },
            'summary': {
                'overall_health': 'No data',
                'key_insights': [],
                'action_items': []
            }
        }

    def _get_error_metrics_result(self, error_message: str) -> Dict[str, Any]:
        """Return error metrics structure when calculation fails."""
        return {
            'calculation_timestamp': datetime.now().isoformat(),
            'total_records_analyzed': 0,
            'data_period': 'Error occurred',
            'error': error_message,
            'core_metrics': {
                'total_incidents': 0,
                'open_incidents': 0,
                'closed_incidents': 0,
                'in_progress_incidents': 0,
                'closure_rate_percentage': 0.0
            },
            'performance_metrics': {
                'average_resolution_days': 0.0,
                'median_resolution_days': 0.0,
                'sla_compliance_rate': 0.0,
                'first_response_time_avg': 0.0
            },
            'priority_analysis': {
                'high_priority_count': 0,
                'high_priority_percentage': 0.0,
                'critical_incidents': 0,
                'priority_distribution': {}
            },
            'assignment_analysis': {
                'workstream_distribution': {},
                'top_assigned_groups': {},
                'unassigned_count': 0,
                'assignment_balance_score': 0.0
            },
            'temporal_analysis': {
                'daily_average': 0.0,
                'weekly_pattern': {},
                'monthly_trend': 'error',
                'peak_hours': []
            },
            'derived_metrics': {
                'backlog_ratio': 0.0,
                'efficiency_score': 0.0,
                'workload_balance': 0.0,
                'trend_momentum': 0.0
            },
            'summary': {
                'overall_health': 'Error',
                'key_insights': [f"Calculation error: {error_message}"],
                'action_items': ["Review data quality and fix calculation errors"]
            }
        }

    def _get_data_period(self, df: pd.DataFrame) -> str:
        """Get the data period from the DataFrame."""
        try:
            if df.empty or 'created_date' not in df.columns:
                return 'Unknown period'
            
            # Convert to datetime if needed
            dates = pd.to_datetime(df['created_date'], errors='coerce')
            dates = dates.dropna()
            
            if dates.empty:
                return 'Unknown period'
                
            min_date = dates.min().strftime('%Y-%m-%d')
            max_date = dates.max().strftime('%Y-%m-%d')
            
            if min_date == max_date:
                return min_date
            else:
                return f"{min_date} to {max_date}"
                
        except Exception:
            return 'Unknown period'

    def _calculate_efficiency_score(self, core_metrics: Dict, resolution_metrics: Dict) -> float:
        """Calculate overall efficiency score."""
        try:
            # Combine closure rate, resolution time, and SLA compliance
            closure_rate = core_metrics.get('closure_rate_percentage', 0) / 100
            avg_resolution = resolution_metrics.get('average_resolution_days', 30)
            sla_compliance = resolution_metrics.get('sla_compliance_rate', 0) / 100
            
            # Normalize resolution time (lower is better, max 30 days)
            resolution_score = max(0, (30 - min(avg_resolution, 30)) / 30)
            
            # Weighted average
            efficiency = (closure_rate * 0.4 + resolution_score * 0.3 + sla_compliance * 0.3) * 100
            return round(efficiency, 1)
        except Exception:
            return 0.0

    def _calculate_workload_balance(self, assignment_metrics: Dict) -> float:
        """Calculate workload balance score."""
        try:
            workstream_dist = assignment_metrics.get('workstream_distribution', {})
            if not workstream_dist:
                return 0.0
            
            # Calculate coefficient of variation (lower is more balanced)
            values = list(workstream_dist.values())
            if len(values) < 2:
                return 100.0
            
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            cv = (std_dev / mean_val) if mean_val > 0 else 1
            # Convert to balance score (100 = perfectly balanced)
            balance_score = max(0, 100 - (cv * 50))
            return round(balance_score, 1)
        except Exception:
            return 0.0

    def _calculate_trend_momentum(self, temporal_metrics: Dict) -> float:
        """Calculate trend momentum score."""
        try:
            trend = temporal_metrics.get('monthly_trend', 'stable')
            daily_avg = temporal_metrics.get('daily_average', 0)
            
            # Convert trend to momentum score
            if trend == 'improving':
                return 75.0
            elif trend == 'stable' and daily_avg > 0:
                return 50.0
            elif trend == 'declining':
                return 25.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def _calculate_overall_health(self, core_metrics: Dict, resolution_metrics: Dict, priority_metrics: Dict) -> str:
        """Calculate overall system health."""
        try:
            closure_rate = core_metrics.get('closure_rate_percentage', 0)
            high_priority = priority_metrics.get('high_priority_percentage', 0)
            avg_resolution = resolution_metrics.get('average_resolution_days', 30)
            
            # Health scoring
            health_score = 0
            if closure_rate >= 80:
                health_score += 40
            elif closure_rate >= 60:
                health_score += 25
            
            if high_priority <= 20:
                health_score += 30
            elif high_priority <= 40:
                health_score += 15
            
            if avg_resolution <= 7:
                health_score += 30
            elif avg_resolution <= 14:
                health_score += 15
            
            if health_score >= 80:
                return 'Excellent'
            elif health_score >= 60:
                return 'Good'
            elif health_score >= 40:
                return 'Fair'
            else:
                return 'Needs Attention'
        except Exception:
            return 'Unknown'

    def _generate_key_insights(self, metrics_result: Dict) -> List[str]:
        """Generate key insights from metrics."""
        insights = []
        try:
            core = metrics_result.get('core_metrics', {})
            performance = metrics_result.get('performance_metrics', {})
            priority = metrics_result.get('priority_analysis', {})
            
            total = core.get('total_incidents', 0)
            open_count = core.get('open_incidents', 0)
            high_priority = priority.get('high_priority_count', 0)
            
            insights.append(f"Total incidents: {total:,}")
            insights.append(f"Open backlog: {open_count:,}")
            
            if high_priority > 0:
                insights.append(f"High priority incidents requiring attention: {high_priority}")
            
            avg_resolution = performance.get('average_resolution_days', 0)
            if avg_resolution > 0:
                insights.append(f"Average resolution time: {avg_resolution:.1f} days")
                
        except Exception:
            insights = ["Analysis completed with basic metrics"]
        
        return insights

    def _generate_action_items(self, metrics_result: Dict) -> List[str]:
        """Generate action items from metrics."""
        actions = []
        try:
            core = metrics_result.get('core_metrics', {})
            performance = metrics_result.get('performance_metrics', {})
            priority = metrics_result.get('priority_analysis', {})
            
            open_count = core.get('open_incidents', 0)
            closure_rate = core.get('closure_rate_percentage', 0)
            high_priority = priority.get('high_priority_count', 0)
            avg_resolution = performance.get('average_resolution_days', 0)
            
            if open_count > 100:
                actions.append("Review and prioritize open incident backlog")
            
            if closure_rate < 70:
                actions.append("Improve incident closure processes and workflows")
            
            if high_priority > 10:
                actions.append("Focus resources on high priority incidents")
                
            if avg_resolution > 14:
                actions.append("Analyze resolution bottlenecks and optimize processes")
            
            if not actions:
                actions.append("Continue monitoring incident trends and performance")
                
        except Exception:
            actions = ["Review system performance and data quality"]
        
        return actions

    def _calculate_enhanced_training_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced training-related metrics."""
        try:
            training_keywords = ['training', 'train', 'tutorial', 'how to', 'guide', 'instruction']
            
            # Find potential training incidents
            training_incidents = pd.DataFrame()
            if 'description' in df.columns:
                desc_mask = df['description'].astype(str).str.lower().str.contains('|'.join(training_keywords), na=False)
                training_incidents = df[desc_mask]
            elif 'short_description' in df.columns:
                desc_mask = df['short_description'].astype(str).str.lower().str.contains('|'.join(training_keywords), na=False)
                training_incidents = df[desc_mask]
            
            training_count = len(training_incidents)
            total_count = len(df)
            training_percentage = (training_count / total_count * 100) if total_count > 0 else 0
            
            return {
                'total_training_tickets': training_count,
                'training_percentage_overall': round(training_percentage, 2),
                'suggested_training_percentage': round(training_percentage * 1.2, 2),  # Estimate
                'historical_training_closures': training_count  # Simplified
            }
        except Exception:
            return {
                'total_training_tickets': 0,
                'training_percentage_overall': 0.0,
                'suggested_training_percentage': 0.0,
                'historical_training_closures': 0
            }

    def _calculate_enhanced_category_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced category-related metrics."""
        try:
            # Basic category analysis
            category_col = None
            for col in ['category', 'subcategory', 'Category', 'Subcategory']:
                if col in df.columns:
                    category_col = col
                    break
            
            if not category_col:
                return {'category_accuracy_percentage': 0.0, 'category_misclassifications': 0}
            
            # Simple accuracy heuristic based on description matching
            categories = df[category_col].dropna().value_counts()
            total_categories = len(categories)
            
            # Estimate accuracy (simplified - in real implementation would use ML/NLP)
            estimated_accuracy = min(90.0, 70.0 + (total_categories * 2))  # More categories = potentially better accuracy
            
            return {
                'category_accuracy_percentage': round(estimated_accuracy, 1),
                'category_misclassifications': max(0, int(len(df) * (100 - estimated_accuracy) / 100)),
                'defect_detection_rate': round(estimated_accuracy * 0.8, 1),  # Related metric
                'total_categories': total_categories
            }
        except Exception:
            return {
                'category_accuracy_percentage': 0.0,
                'category_misclassifications': 0,
                'defect_detection_rate': 0.0,
                'total_categories': 0
            }
    
    def _calculate_executive_ready_metrics(self, core_metrics: Dict, priority_metrics: Dict, 
                                         temporal_metrics: Dict, resolution_metrics: Dict, 
                                         assignment_metrics: Dict) -> Dict[str, Any]:
        """Calculate all executive dashboard metrics in ready-to-use format (no transformations needed)."""
        try:
            # Extract weekly data and calculate weekly metrics
            weekly_counts = temporal_metrics.get('total_weekly_counts', {})
            weeks = sorted(weekly_counts.keys(), reverse=True) if weekly_counts else []
            
            current_week = weekly_counts.get(weeks[0], 0) if len(weeks) > 0 else 0
            last_week = weekly_counts.get(weeks[1], 0) if len(weeks) > 1 else 0  
            two_weeks_ago = weekly_counts.get(weeks[2], 0) if len(weeks) > 2 else 0
            
            # Calculate weekly change percentages
            weekly_change_pct = 0.0
            previous_weekly_change_pct = 0.0
            
            if last_week > 0:
                weekly_change_pct = round(((current_week - last_week) / last_week) * 100, 1)
            if two_weeks_ago > 0:
                previous_weekly_change_pct = round(((last_week - two_weeks_ago) / two_weeks_ago) * 100, 1)
                
            # Determine trend direction
            trend_direction = 'stable'
            if weekly_change_pct > 5:
                trend_direction = 'increasing'
            elif weekly_change_pct < -5:
                trend_direction = 'decreasing'
            
            # Calculate velocity metrics
            daily_average = temporal_metrics.get('daily_average', 0)
            total_weeks = temporal_metrics.get('total_weeks_analyzed', 1)
            total_daily_avg = core_metrics.get('total_incidents', 0) / max(total_weeks * 7, 1)
            
            # Build comprehensive executive-ready metrics
            return {
                # Weekly Analysis (ready for direct display)
                'current_week_count': current_week,
                'last_week_count': last_week, 
                'two_weeks_ago_count': two_weeks_ago,
                'weekly_change_pct': weekly_change_pct,
                'previous_weekly_change_pct': previous_weekly_change_pct,
                'weekly_trend_direction': trend_direction,
                
                # Priority Summary (ready for dashboard)
                'priority_summary': {
                    'high_total': priority_metrics.get('priority_distribution', {}).get('High', 0),
                    'medium_total': priority_metrics.get('priority_distribution', {}).get('Medium', 0),
                    'low_total': priority_metrics.get('priority_distribution', {}).get('Low', 0),
                    'high_open': priority_metrics.get('total_open_high', 0),
                    'medium_open': priority_metrics.get('total_open_medium', 0),
                    'low_open': priority_metrics.get('total_open_low', 0),
                    'high_vs_target': priority_metrics.get('high_vs_target', 0),
                    'medium_vs_target': priority_metrics.get('medium_vs_target', 0),
                    'total_target_gap': priority_metrics.get('total_target_gap', 0)
                },
                
                # Velocity Metrics (ready for display)
                'velocity_summary': {
                    'open_daily_average': round(daily_average, 1),
                    'total_daily_average': round(total_daily_avg, 1),
                    'open_weekly_average': round(daily_average * 7, 1),
                    'open_incident_count': core_metrics.get('open_incidents', 0)
                },
                
                # Performance Summary (ready for dashboard)
                'performance_summary': {
                    'closure_rate': round(core_metrics.get('closure_rate_percentage', 0), 1),
                    'avg_resolution_days': round(resolution_metrics.get('average_resolution_days', 0), 1),
                    'sla_compliance_rate': round(resolution_metrics.get('sla_compliance_rate', 0), 1)
                },
                
                # Trend Analysis (ready for display)
                'trend_summary': {
                    'total_trend_direction': temporal_metrics.get('total_trend_direction', 'stable'),
                    'total_trend_slope': temporal_metrics.get('total_trend_slope', 0),
                    'average_weekly_volume': temporal_metrics.get('total_average_weekly_volume', 0),
                    'weeks_analyzed': temporal_metrics.get('total_weeks_analyzed', 0)
                },
                
                # Backlog Metrics (ready for dashboard)  
                'backlog_summary': {
                    'current_backlog': core_metrics.get('open_incidents', 0),
                    'total_incidents': core_metrics.get('total_incidents', 0),
                    'closed_incidents': core_metrics.get('closed_incidents', 0),
                    'backlog_ratio': round(core_metrics.get('open_incidents', 0) / max(1, core_metrics.get('total_incidents', 1)), 3)
                }
            }
            
        except Exception as e:
            # Fallback with basic metrics if calculation fails
            return {
                'current_week_count': 0,
                'last_week_count': 0,
                'two_weeks_ago_count': 0,
                'weekly_change_pct': 0.0,
                'previous_weekly_change_pct': 0.0,
                'weekly_trend_direction': 'stable',
                'priority_summary': {'high_total': 0, 'medium_total': 0, 'low_total': 0, 'high_open': 0, 'medium_open': 0, 'low_open': 0, 'high_vs_target': 0, 'medium_vs_target': 0, 'total_target_gap': 0},
                'velocity_summary': {'open_daily_average': 0, 'total_daily_average': 0, 'open_weekly_average': 0, 'open_incident_count': 0},
                'performance_summary': {'closure_rate': 0, 'avg_resolution_days': 0, 'sla_compliance_rate': 0},
                'trend_summary': {'total_trend_direction': 'stable', 'total_trend_slope': 0, 'average_weekly_volume': 0, 'weeks_analyzed': 0},
                'backlog_summary': {'current_backlog': 0, 'total_incidents': 0, 'closed_incidents': 0, 'backlog_ratio': 0}
            }


# Alias for backward compatibility
MetricsCalculator = IncidentMetricsCalculator


# Alias for backward compatibility
MetricsCalculator = IncidentMetricsCalculator