#!/usr/bin/env python3
"""
Enhanced Trend Analysis for SAP Incident Data
Implements clean pandas patterns with method chaining and proper refactoring principles.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import Counter
import warnings

@dataclass
class TrendMetrics:
    """
    Enhanced data class to hold comprehensive trend analysis results.
    Implements dictionary-like interface for seamless integration.
    """
    total_incidents: int
    trend_period_days: int
    incident_velocity: float
    priority_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
    top_categories: List[Tuple[str, int]]
    resolution_trends: Dict[str, float]
    peak_hours: List[int]
    seasonal_patterns: Dict[str, Any]
    trend_direction: str = "stable"
    health_score: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary for report generation."""
        base_dict = asdict(self)
        base_dict.update({
            'summary_stats': self._generate_summary_stats(),
            'recommendations': self._generate_recommendations()
        })
        return base_dict
    
    def get(self, key: str, default=None):
        """Provide dictionary-like get method for backward compatibility."""
        return getattr(self, key, default)
    
    def items(self):
        """Provide dictionary-like items method."""
        return self.to_dict().items()
    
    def keys(self):
        """Provide dictionary-like keys method."""
        return self.to_dict().keys()
    
    def values(self):
        """Provide dictionary-like values method."""
        return self.to_dict().values()
    
    def __getitem__(self, key):
        """Enable dictionary-style access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in TrendMetrics")
    
    def __contains__(self, key):
        """Enable 'in' operator."""
        return hasattr(self, key)
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate executive summary statistics."""
        resolution_rate = self.resolution_trends.get('resolution_rate', 0)
        avg_resolution_hours = self.resolution_trends.get('avg_resolution_hours', 0)
        
        critical_count = self.priority_distribution.get('Critical', 0)
        high_count = self.priority_distribution.get('High', 0)
        high_priority_ratio = (critical_count + high_count) / max(self.total_incidents, 1)
        
        return {
            'resolution_efficiency': 'excellent' if avg_resolution_hours < 12 else 
                                   'good' if avg_resolution_hours < 24 else 'needs_improvement',
            'priority_health': 'good' if high_priority_ratio < 0.2 else 'concerning',
            'volume_assessment': 'high' if self.incident_velocity > 15 else 'normal',
            'overall_status': 'healthy' if self.health_score > 80 else 
                            'warning' if self.health_score > 60 else 'critical'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        if self.incident_velocity > 20:
            recommendations.append("Consider implementing proactive monitoring to reduce incident volume")
        
        critical_ratio = self.priority_distribution.get('Critical', 0) / max(self.total_incidents, 1)
        if critical_ratio > 0.1:
            recommendations.append("High critical incident ratio - review system stability")
        
        avg_resolution = self.resolution_trends.get('avg_resolution_hours', 0)
        if avg_resolution > 48:
            recommendations.append("Resolution times exceed targets - optimize support processes")
        
        if not recommendations:
            recommendations.append("Current performance metrics are within acceptable ranges")
        
        return recommendations


class IncidentTrendAnalyzer:
    """
    Enhanced trend analyzer implementing clean pandas patterns and method chaining.
    Follows refactoring principles for maintainable, testable code.
    
    NOTE: This is the main class name that your application expects to import.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configurable parameters for future flexibility."""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration with ability to override
        self.config = {
            'trend_window_days': 30,
            'top_categories_limit': 10,
            'min_incidents_for_trend': 5,
            'peak_hours_count': 3,
            'resolution_sla_hours': 24
        }
        
        if config:
            self.config.update(config)
        
        # Column mapping for flexible data sources
        self.column_mapping = {
            'created_date': ['Created', 'Created Date', 'Opened', 'Date Created'],
            'resolved_date': ['Resolved', 'Closed', 'Resolved Date'],
            'priority': ['Priority', 'Incident Priority', 'Severity'],
            'status': ['State', 'Status', 'Incident State'],
            'assignment_group': ['Assignment Group', 'Assigned To', 'Team'],
            'description': ['Short Description', 'Description', 'Summary'],
            'category': ['Category', 'Incident Category', 'Type']
        }
        
        print(f"DEBUG: IncidentTrendAnalyzer initialized successfully")
    
    def analyze_trends(self, df: pd.DataFrame) -> TrendMetrics:
        """
        Perform comprehensive trend analysis using clean pandas patterns.
        Implements method chaining for readable, maintainable code.
        """
        print(f"DEBUG: Starting trend analysis for {len(df)} incidents")
        self.logger.info(f"Starting enhanced trend analysis for {len(df)} incidents")
        
        try:
            # Clean and prepare data using method chaining
            prepared_df = self._prepare_data_pipeline(df)
            print(f"DEBUG: Data preparation completed, {len(prepared_df)} valid records")
            
            if prepared_df.empty:
                print("DEBUG: No valid data after preparation, returning empty metrics")
                return self._create_empty_metrics()
            
            # Calculate metrics using focused, testable methods
            print("DEBUG: Calculating trend metrics...")
            metrics_data = {
                'total_incidents': len(prepared_df),
                'trend_period_days': self._calculate_trend_period(prepared_df),
                'incident_velocity': self._calculate_velocity(prepared_df),
                'priority_distribution': self._analyze_priority_distribution(prepared_df),
                'status_distribution': self._analyze_status_distribution(prepared_df),
                'top_categories': self._identify_top_categories(prepared_df),
                'resolution_trends': self._analyze_resolution_trends(prepared_df),
                'peak_hours': self._identify_peak_hours(prepared_df),
                'seasonal_patterns': self._analyze_seasonal_patterns(prepared_df)
            }
            
            # Calculate derived metrics
            metrics_data['trend_direction'] = self._determine_trend_direction(
                metrics_data['incident_velocity'], prepared_df
            )
            metrics_data['health_score'] = self._calculate_health_score(metrics_data)
            
            print(f"DEBUG: Metrics calculation completed successfully")
            print(f"DEBUG: Velocity: {metrics_data['incident_velocity']:.2f}, Direction: {metrics_data['trend_direction']}")
            
            metrics = TrendMetrics(**metrics_data)
            
            self.logger.info(f"Trend analysis completed successfully")
            return metrics
            
        except Exception as e:
            print(f"DEBUG: Error in trend analysis: {str(e)}")
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return self._create_empty_metrics()
    
    def _prepare_data_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data using pandas method chaining.
        Implements the refactoring principles from the context documentation.
        """
        try:
            return (df
                    .copy()
                    .pipe(self._standardize_columns)
                    .pipe(self._convert_date_columns)
                    .pipe(self._clean_categorical_data)
                    .pipe(self._filter_valid_records)
                    )
        except Exception as e:
            print(f"DEBUG: Error in data pipeline: {str(e)}")
            return df.copy()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistent analysis."""
        standardized_df = df.copy()
        
        for standard_name, possible_names in self.column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    standardized_df[standard_name] = df[possible_name]
                    break
        
        return standardized_df
    
    def _convert_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns using pandas best practices."""
        date_columns = ['created_date', 'resolved_date']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    print(f"DEBUG: Could not convert {col} to datetime: {str(e)}")
        
        return df
    
    def _clean_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize categorical data."""
        categorical_columns = ['priority', 'status', 'assignment_group']
        
        for col in categorical_columns:
            if col in df.columns:
                try:
                    df[col] = (df[col]
                              .astype(str)
                              .str.strip()
                              .str.title()
                              .replace({'Nan': None, 'None': None})
                              )
                except Exception as e:
                    print(f"DEBUG: Could not clean {col}: {str(e)}")
        
        return df
    
    def _filter_valid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid records using pandas filtering."""
        if 'created_date' in df.columns:
            return df[df['created_date'].notna()]
        return df
    
    def _calculate_velocity(self, df: pd.DataFrame) -> float:
        """Calculate incident velocity using mathematical operations."""
        if 'created_date' not in df.columns:
            return 0.0
        
        try:
            date_range = (df['created_date'].max() - df['created_date'].min()).days
            return len(df) / max(date_range, 1)
        except Exception:
            return 0.0
    
    def _analyze_priority_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze priority distribution using pandas value_counts."""
        if 'priority' not in df.columns:
            return {}
        
        try:
            return df['priority'].value_counts().to_dict()
        except Exception:
            return {}
    
    def _analyze_status_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze status distribution using pandas operations."""
        if 'status' not in df.columns:
            return {}
        
        try:
            return df['status'].value_counts().to_dict()
        except Exception:
            return {}
    
    def _identify_top_categories(self, df: pd.DataFrame) -> List[Tuple[str, int]]:
        """Identify top categories using pandas aggregation."""
        if 'assignment_group' not in df.columns:
            return []
        
        try:
            top_categories = (df['assignment_group']
                            .value_counts()
                            .head(self.config['top_categories_limit'])
                            )
            return [(str(cat), int(count)) for cat, count in top_categories.items()]
        except Exception:
            return []
    
    def _analyze_resolution_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze resolution trends using pandas datetime operations."""
        trends = {
            'avg_resolution_hours': 0.0,
            'resolution_rate': 0.0,
            'pending_incidents': 0,
            'sla_compliance': 0.0
        }
        
        try:
            if 'created_date' not in df.columns:
                return trends
            
            # Calculate resolution metrics for resolved incidents
            if 'resolved_date' in df.columns:
                resolved_mask = df['resolved_date'].notna()
                resolved_df = df[resolved_mask]
                
                if not resolved_df.empty:
                    resolution_times = (resolved_df['resolved_date'] - 
                                      resolved_df['created_date']).dt.total_seconds() / 3600
                    
                    trends['avg_resolution_hours'] = resolution_times.mean()
                    trends['resolution_rate'] = len(resolved_df) / len(df)
                    
                    # SLA compliance calculation
                    sla_met = resolution_times <= self.config['resolution_sla_hours']
                    trends['sla_compliance'] = sla_met.mean()
            
            # Count pending incidents
            if 'status' in df.columns:
                pending_statuses = ['New', 'In Progress', 'Pending', 'Open']
                trends['pending_incidents'] = df['status'].isin(pending_statuses).sum()
        
        except Exception as e:
            print(f"DEBUG: Error calculating resolution trends: {str(e)}")
        
        return trends
    
    def _identify_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """Identify peak hours using pandas datetime extraction."""
        if 'created_date' not in df.columns:
            return []
        
        try:
            return (df['created_date']
                    .dt.hour
                    .value_counts()
                    .head(self.config['peak_hours_count'])
                    .index
                    .tolist()
                    )
        except Exception:
            return []
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns using pandas datetime operations."""
        patterns = {
            'busiest_day_of_week': None,
            'busiest_month': None,
            'weekend_vs_weekday_ratio': 0.0,
            'monthly_trend': {}
        }
        
        try:
            if 'created_date' not in df.columns:
                return patterns
            
            # Day of week analysis
            day_counts = df['created_date'].dt.day_name().value_counts()
            if not day_counts.empty:
                patterns['busiest_day_of_week'] = day_counts.index[0]
            
            # Month analysis
            month_counts = df['created_date'].dt.month_name().value_counts()
            if not month_counts.empty:
                patterns['busiest_month'] = month_counts.index[0]
                patterns['monthly_trend'] = month_counts.to_dict()
            
            # Weekend vs weekday analysis
            is_weekend = df['created_date'].dt.weekday >= 5
            weekend_count = is_weekend.sum()
            weekday_count = len(df) - weekend_count
            
            if weekday_count > 0:
                patterns['weekend_vs_weekday_ratio'] = weekend_count / weekday_count
        
        except Exception as e:
            print(f"DEBUG: Error in seasonal analysis: {str(e)}")
        
        return patterns
    
    def _calculate_trend_period(self, df: pd.DataFrame) -> int:
        """Calculate analysis period in days."""
        try:
            if 'created_date' not in df.columns or df.empty:
                return 1
            
            return max(1, (df['created_date'].max() - df['created_date'].min()).days)
        except Exception:
            return 1
    
    def _determine_trend_direction(self, velocity: float, df: pd.DataFrame) -> str:
        """Determine trend direction using velocity and time-based analysis."""
        try:
            if velocity > 15:
                return 'increasing'
            elif velocity < 3:
                return 'decreasing'
            
            # Additional time-based trend analysis
            if 'created_date' in df.columns and len(df) > 10:
                # Compare first and second half of data
                df_sorted = df.sort_values('created_date')
                midpoint = len(df_sorted) // 2
                
                first_half_velocity = len(df_sorted[:midpoint]) / max(1, midpoint)
                second_half_velocity = len(df_sorted[midpoint:]) / max(1, len(df_sorted) - midpoint)
                
                if second_half_velocity > first_half_velocity * 1.2:
                    return 'increasing'
                elif second_half_velocity < first_half_velocity * 0.8:
                    return 'decreasing'
        
        except Exception:
            pass
        
        return 'stable'
    
    def _calculate_health_score(self, metrics_data: Dict[str, Any]) -> int:
        """Calculate overall system health score using mathematical operations."""
        try:
            score = 100
            
            # Velocity impact
            velocity = metrics_data['incident_velocity']
            if velocity > 20:
                score -= 30
            elif velocity > 15:
                score -= 15
            
            # Resolution impact
            resolution_trends = metrics_data['resolution_trends']
            resolution_rate = resolution_trends.get('resolution_rate', 0)
            if resolution_rate < 0.7:
                score -= 25
            elif resolution_rate < 0.8:
                score -= 10
            
            # Priority impact
            priority_dist = metrics_data['priority_distribution']
            total_incidents = metrics_data['total_incidents']
            critical_ratio = priority_dist.get('Critical', 0) / max(total_incidents, 1)
            
            if critical_ratio > 0.15:
                score -= 20
            elif critical_ratio > 0.1:
                score -= 10
            
            return max(0, score)
        
        except Exception:
            return 50  # Default middle score if calculation fails
    
    def _create_empty_metrics(self) -> TrendMetrics:
        """Create empty metrics object for edge cases."""
        return TrendMetrics(
            total_incidents=0,
            trend_period_days=1,
            incident_velocity=0.0,
            priority_distribution={},
            status_distribution={},
            top_categories=[],
            resolution_trends={},
            peak_hours=[],
            seasonal_patterns={},
            trend_direction='unknown',
            health_score=0
        )
    
    def generate_trend_report(self, metrics: TrendMetrics) -> str:
        """Generate comprehensive trend report with enhanced formatting."""
        try:
            report_sections = [
                self._format_header(metrics),
                self._format_overview(metrics),
                self._format_distributions(metrics),
                self._format_performance_metrics(metrics),
                self._format_patterns(metrics),
                self._format_recommendations(metrics)
            ]
            
            return '\n'.join(report_sections)
        
        except Exception as e:
            return f"Error generating trend report: {str(e)}"
    
    def _format_header(self, metrics: TrendMetrics) -> str:
        """Format report header section."""
        return f"""
{'='*70}
SAP INCIDENT MANAGEMENT - ENHANCED TREND ANALYSIS REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {metrics.trend_period_days} days
System Health Score: {metrics.health_score}/100
"""
    
    def _format_overview(self, metrics: TrendMetrics) -> str:
        """Format overview section."""
        return f"""
EXECUTIVE OVERVIEW
{'-'*50}
Total Incidents: {metrics.total_incidents:,}
Incident Velocity: {metrics.incident_velocity:.2f} incidents/day
Trend Direction: {metrics.trend_direction.upper()}
Overall Status: {metrics._generate_summary_stats()['overall_status'].upper()}
"""
    
    def _format_distributions(self, metrics: TrendMetrics) -> str:
        """Format distribution analysis."""
        sections = ["\nDISTRIBUTION ANALYSIS", "-"*50]
        
        if metrics.priority_distribution:
            sections.append("Priority Distribution:")
            for priority, count in metrics.priority_distribution.items():
                percentage = (count / metrics.total_incidents) * 100
                sections.append(f"  • {priority}: {count} ({percentage:.1f}%)")
        
        if metrics.status_distribution:
            sections.append("\nStatus Distribution:")
            for status, count in metrics.status_distribution.items():
                percentage = (count / metrics.total_incidents) * 100
                sections.append(f"  • {status}: {count} ({percentage:.1f}%)")
        
        return '\n'.join(sections)
    
    def _format_performance_metrics(self, metrics: TrendMetrics) -> str:
        """Format performance metrics section."""
        sections = ["\nPERFORMANCE METRICS", "-"*50]
        
        rt = metrics.resolution_trends
        if rt:
            sections.extend([
                f"Average Resolution Time: {rt.get('avg_resolution_hours', 0):.1f} hours",
                f"Resolution Rate: {rt.get('resolution_rate', 0):.1%}",
                f"SLA Compliance: {rt.get('sla_compliance', 0):.1%}",
                f"Pending Incidents: {rt.get('pending_incidents', 0)}"
            ])
        
        return '\n'.join(sections)
    
    def _format_patterns(self, metrics: TrendMetrics) -> str:
        """Format pattern analysis section."""
        sections = ["\nPATTERN ANALYSIS", "-"*50]
        
        if metrics.peak_hours:
            sections.append(f"Peak Hours: {', '.join(map(str, metrics.peak_hours))}")
        
        sp = metrics.seasonal_patterns
        if sp.get('busiest_day_of_week'):
            sections.append(f"Busiest Day: {sp['busiest_day_of_week']}")
        if sp.get('busiest_month'):
            sections.append(f"Busiest Month: {sp['busiest_month']}")
        
        return '\n'.join(sections)
    
    def _format_recommendations(self, metrics: TrendMetrics) -> str:
        """Format recommendations section."""
        sections = ["\nRECOMMENDATIONS", "-"*50]
        
        recommendations = metrics._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            sections.append(f"{i}. {rec}")
        
        sections.append(f"\n{'='*70}")
        return '\n'.join(sections)


# Maintain backward compatibility with multiple aliases
TrendAnalyzer = IncidentTrendAnalyzer
EnhancedIncidentTrendAnalyzer = IncidentTrendAnalyzer

print("DEBUG: trend_analyzer.py loaded successfully with IncidentTrendAnalyzer class")