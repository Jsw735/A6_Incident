#!/usr/bin/env python3
"""
Incident Quality Analyzer Module
Handles comprehensive data quality analysis and metrics calculation.
Following clean code principles and data analysis best practices.
"""

import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class IncidentQualityAnalyzer:
    """
    Professional quality analyzer for incident management systems.
    Implements comprehensive data analysis following clean code principles.
    """
    
    def __init__(self):
        """Initialize the quality analyzer with comprehensive configuration."""
        self.logger = logging.getLogger(__name__)
        self.analysis_results = {}
        self.quality_metrics = {}
        self.trend_analysis = {}
        
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method for comprehensive data quality analysis.
        This method name matches the interface contract expected by main application.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Comprehensive analysis results dictionary
        """
        try:
            if df is None or df.empty:
                return self._create_error_result("DataFrame is empty or None")
            
            self.logger.info(f"Starting comprehensive data quality analysis for {len(df)} records")
            
            # Initialize analysis results structure
            self.analysis_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'dataset_overview': self._analyze_dataset_overview(df),
                'quality_metrics': self._calculate_quality_metrics(df),
                'trend_analysis': self._perform_trend_analysis(df),
                'pattern_analysis': self._analyze_patterns(df),
                'anomaly_detection': self._detect_anomalies(df),
                'performance_metrics': self._calculate_performance_metrics(df),
                'recommendations': self._generate_analysis_recommendations(df),
                'summary': {}
            }
            
            # Generate executive summary
            self.analysis_results['summary'] = self._generate_executive_summary()
            
            self.logger.info("Data quality analysis completed successfully")

            # Create a result wrapper that supports attribute access and
            # provides flattened compatibility keys expected by older tests.
            class ResultDict(dict):
                def __getattr__(self, name):
                    if name in self:
                        return self[name]
                    raise AttributeError(name)

            result = ResultDict(self.analysis_results)

            # Flatten commonly-requested compatibility fields
            try:
                result['total_records'] = self.analysis_results.get('dataset_overview', {}).get('total_records', 0)
                # completeness as percentage of rows with no missing values
                if df is not None and len(df) > 0:
                    complete_rows = int(df.dropna().shape[0])
                    completeness_pct = float((complete_rows / len(df)) * 100)
                else:
                    completeness_pct = 0.0
                result['completeness_score'] = completeness_pct

                # duplicate count for 'number' field if available
                uniqueness = self.analysis_results.get('quality_metrics', {}).get('uniqueness', {})
                by_field = uniqueness.get('by_field', {}) if isinstance(uniqueness, dict) else {}
                number_info = by_field.get('number', {}) if isinstance(by_field, dict) else {}
                dup_count = number_info.get('duplicate_count', None)
                # Fallback: compute max duplicates across columns if possible
                if dup_count is None and df is not None and len(df) > 0:
                    max_dup = 0
                    for col in df.columns:
                        total = int(df[col].notna().sum())
                        unique = int(df[col].nunique(dropna=True))
                        dup_here = max(0, total - unique)
                        if dup_here > max_dup:
                            max_dup = dup_here
                    dup_count = int(max_dup)
                result['duplicate_count'] = int(dup_count or 0)

                # Provide normalized quality_dimensions and overall_quality (0..1)
                q_metrics = self.analysis_results.get('quality_metrics', {})
                quality_dimensions = {}
                for dim in ['completeness', 'consistency', 'accuracy', 'timeliness']:
                    dim_val = q_metrics.get(dim, {})
                    score = dim_val.get('overall_score', None) if isinstance(dim_val, dict) else None
                    if score is not None:
                        quality_dimensions[dim] = {'score': score / 100.0}
                result['quality_dimensions'] = quality_dimensions
                overall_pct = q_metrics.get('overall_quality_score', 0.0)
                result['overall_quality'] = {'score': overall_pct / 100.0}
            except Exception:
                # If compatibility shaping fails, fall back to raw dict
                pass

            return result
            
        except Exception as e:
            error_msg = f"Error in analyze_data_quality: {str(e)}"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
    
    def analyze_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Alias method for backward compatibility.
        Delegates to analyze_data_quality for consistent interface.
        """
        return self.analyze_data_quality(df)


    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result following defensive programming."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'dataset_overview': {},
            'quality_metrics': {},
            'trend_analysis': {},
            'pattern_analysis': {},
            'anomaly_detection': {},
            'performance_metrics': {},
            'recommendations': ['Fix critical errors before proceeding with analysis'],
            'summary': {'status': 'failed', 'error': error_message}
        }
    
    def _analyze_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset characteristics and structure."""
        try:
            overview = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'column_types': df.dtypes.value_counts().to_dict(),
                'missing_data_summary': {
                    'total_missing_values': df.isnull().sum().sum(),
                    'columns_with_missing': df.isnull().any().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                },
                'data_range': {}
            }
            
            # Analyze date ranges if date columns exist
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                if not df[col].empty and df[col].notna().any():
                    overview['data_range'][col] = {
                        'earliest': df[col].min().isoformat() if pd.notna(df[col].min()) else None,
                        'latest': df[col].max().isoformat() if pd.notna(df[col].max()) else None,
                        'span_days': (df[col].max() - df[col].min()).days if pd.notna(df[col].min()) and pd.notna(df[col].max()) else 0
                    }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error in dataset overview analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        try:
            metrics = {
                'completeness': self._calculate_completeness_metrics(df),
                'consistency': self._calculate_consistency_metrics(df),
                'accuracy': self._calculate_accuracy_metrics(df),
                'timeliness': self._calculate_timeliness_metrics(df),
                'uniqueness': self._calculate_uniqueness_metrics(df)
            }
            
            # Calculate overall quality score
            quality_scores = []
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict) and 'overall_score' in category_metrics:
                    quality_scores.append(category_metrics['overall_score'])
            
            metrics['overall_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_completeness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data completeness metrics."""
        try:
            completeness_by_column = {}
            for col in df.columns:
                non_null_count = df[col].notna().sum()
                completeness_ratio = non_null_count / len(df) if len(df) > 0 else 0.0
                completeness_by_column[col] = {
                    'completeness_percentage': completeness_ratio * 100,
                    'missing_count': len(df) - non_null_count,
                    'non_null_count': non_null_count
                }
            
            # Calculate overall completeness
            overall_completeness = np.mean([metrics['completeness_percentage'] 
                                          for metrics in completeness_by_column.values()])
            
            return {
                'overall_score': overall_completeness,
                'by_column': completeness_by_column,
                'summary': f"Overall completeness: {overall_completeness:.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating completeness metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_consistency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data consistency metrics."""
        try:
            consistency_score = 100.0
            consistency_issues = []
            
            # Check for consistent data formats
            for col in df.select_dtypes(include=['object']).columns:
                if col in df.columns:
                    # Check for mixed case inconsistencies
                    unique_values = df[col].dropna().astype(str).unique()
                    case_variations = {}
                    for value in unique_values:
                        lower_value = value.lower()
                        if lower_value in case_variations:
                            case_variations[lower_value].append(value)
                        else:
                            case_variations[lower_value] = [value]
                    
                    inconsistent_cases = {k: v for k, v in case_variations.items() if len(v) > 1}
                    if inconsistent_cases:
                        consistency_issues.append(f"Column '{col}' has case inconsistencies")
                        consistency_score -= min(10.0, len(inconsistent_cases))
            
            return {
                'overall_score': max(0.0, consistency_score),
                'issues': consistency_issues,
                'summary': f"Consistency score: {consistency_score:.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data accuracy metrics."""
        try:
            accuracy_score = 100.0
            accuracy_issues = []
            
            # Check for valid data ranges
            if 'urgency' in df.columns:
                invalid_urgency = df[~df['urgency'].isin([1, 2, 3, 4, 5])]['urgency'].count()
                if invalid_urgency > 0:
                    accuracy_issues.append(f"Invalid urgency values: {invalid_urgency}")
                    accuracy_score -= min(20.0, (invalid_urgency / len(df)) * 100)
            
            # Check for reasonable date ranges
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                if col in df.columns and df[col].notna().any():
                    future_dates = df[df[col] > datetime.now()][col].count()
                    if future_dates > 0 and col != 'resolved_at':  # Future resolution dates might be valid
                        accuracy_issues.append(f"Future dates in {col}: {future_dates}")
                        accuracy_score -= min(15.0, (future_dates / len(df)) * 100)
            
            return {
                'overall_score': max(0.0, accuracy_score),
                'issues': accuracy_issues,
                'summary': f"Accuracy score: {accuracy_score:.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_timeliness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data timeliness metrics."""
        try:
            timeliness_score = 100.0
            timeliness_analysis = {}
            
            if 'sys_created_on' in df.columns:
                df_dates = pd.to_datetime(df['sys_created_on'], errors='coerce')
                if df_dates.notna().any():
                    latest_date = df_dates.max()
                    days_since_latest = (datetime.now() - latest_date).days
                    
                    timeliness_analysis['latest_record_age_days'] = days_since_latest
                    timeliness_analysis['data_freshness'] = 'Fresh' if days_since_latest <= 7 else 'Stale'
                    
                    if days_since_latest > 30:
                        timeliness_score -= 25.0
                    elif days_since_latest > 7:
                        timeliness_score -= 10.0
            
            return {
                'overall_score': timeliness_score,
                'analysis': timeliness_analysis,
                'summary': f"Timeliness score: {timeliness_score:.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating timeliness metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_uniqueness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data uniqueness metrics."""
        try:
            uniqueness_analysis = {}
            
            # Check uniqueness of key fields
            key_fields = ['number']  # Incident number should be unique
            for field in key_fields:
                if field in df.columns:
                    total_records = len(df[df[field].notna()])
                    unique_records = df[field].nunique()
                    uniqueness_ratio = unique_records / total_records if total_records > 0 else 0.0
                    
                    uniqueness_analysis[field] = {
                        'uniqueness_percentage': uniqueness_ratio * 100,
                        'duplicate_count': total_records - unique_records,
                        'unique_count': unique_records
                    }
            
            # Calculate overall uniqueness score
            overall_uniqueness = np.mean([metrics['uniqueness_percentage'] 
                                        for metrics in uniqueness_analysis.values()]) if uniqueness_analysis else 100.0
            
            return {
                'overall_score': overall_uniqueness,
                'by_field': uniqueness_analysis,
                'summary': f"Uniqueness score: {overall_uniqueness:.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating uniqueness metrics: {str(e)}")
            return {'error': str(e)}
    
    def _perform_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis on incident data."""
        try:
            trend_analysis = {}
            
            if 'sys_created_on' in df.columns:
                df_with_dates = df.copy()
                df_with_dates['sys_created_on'] = pd.to_datetime(df_with_dates['sys_created_on'], errors='coerce')
                df_with_dates = df_with_dates.dropna(subset=['sys_created_on'])
                
                if not df_with_dates.empty:
                    # Monthly incident trends
                    monthly_counts = df_with_dates.groupby(df_with_dates['sys_created_on'].dt.to_period('M')).size()
                    trend_analysis['monthly_trends'] = {
                        'data': monthly_counts.to_dict(),
                        'average_per_month': monthly_counts.mean(),
                        'trend_direction': 'increasing' if monthly_counts.iloc[-1] > monthly_counts.iloc[0] else 'decreasing'
                    }
                    
                    # Weekly patterns
                    weekly_patterns = df_with_dates.groupby(df_with_dates['sys_created_on'].dt.day_name()).size()
                    trend_analysis['weekly_patterns'] = weekly_patterns.to_dict()
                    
                    # Hourly patterns (if time information available)
                    if df_with_dates['sys_created_on'].dt.hour.notna().any():
                        hourly_patterns = df_with_dates.groupby(df_with_dates['sys_created_on'].dt.hour).size()
                        trend_analysis['hourly_patterns'] = hourly_patterns.to_dict()
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in incident data."""
        try:
            pattern_analysis = {}
            
            # Category distribution
            if 'category' in df.columns:
                category_dist = df['category'].value_counts().head(10)
                pattern_analysis['top_categories'] = category_dist.to_dict()
            
            # State distribution
            if 'state' in df.columns:
                state_dist = df['state'].value_counts()
                pattern_analysis['state_distribution'] = state_dist.to_dict()
            
            # Urgency distribution
            if 'urgency' in df.columns:
                urgency_dist = df['urgency'].value_counts().sort_index()
                pattern_analysis['urgency_distribution'] = urgency_dist.to_dict()
            
            # Assignment group patterns
            if 'assignment_group' in df.columns:
                assignment_dist = df['assignment_group'].value_counts().head(10)
                pattern_analysis['top_assignment_groups'] = assignment_dist.to_dict()
            
            return pattern_analysis
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {str(e)}")
            return {'error': str(e)}
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in incident data."""
        try:
            anomalies = {}
            
            # Detect unusual incident volumes
            if 'sys_created_on' in df.columns:
                df_dates = df.copy()
                df_dates['sys_created_on'] = pd.to_datetime(df_dates['sys_created_on'], errors='coerce')
                daily_counts = df_dates.groupby(df_dates['sys_created_on'].dt.date).size()
                
                if len(daily_counts) > 1:
                    mean_daily = daily_counts.mean()
                    std_daily = daily_counts.std()
                    threshold = mean_daily + (2 * std_daily)
                    
                    anomalous_days = daily_counts[daily_counts > threshold]
                    if not anomalous_days.empty:
                        anomalies['high_volume_days'] = {
                            str(date): count for date, count in anomalous_days.items()
                        }
            
            # Detect unusual patterns in urgency
            if 'urgency' in df.columns:
                urgency_counts = df['urgency'].value_counts()
                if len(urgency_counts) > 0:
                    max_urgency_pct = (urgency_counts.max() / len(df)) * 100
                    if max_urgency_pct > 80:
                        anomalies['urgency_concentration'] = f"High concentration in urgency level: {max_urgency_pct:.1f}%"
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for incident management."""
        try:
            performance_metrics = {}
            
            # Resolution time analysis
            if 'sys_created_on' in df.columns and 'resolved_at' in df.columns:
                df_perf = df.copy()
                df_perf['sys_created_on'] = pd.to_datetime(df_perf['sys_created_on'], errors='coerce')
                df_perf['resolved_at'] = pd.to_datetime(df_perf['resolved_at'], errors='coerce')
                
                resolved_incidents = df_perf.dropna(subset=['sys_created_on', 'resolved_at'])
                if not resolved_incidents.empty:
                    resolution_times = (resolved_incidents['resolved_at'] - resolved_incidents['sys_created_on']).dt.total_seconds() / 3600  # Hours
                    
                    performance_metrics['resolution_time_analysis'] = {
                        'average_hours': resolution_times.mean(),
                        'median_hours': resolution_times.median(),
                        'min_hours': resolution_times.min(),
                        'max_hours': resolution_times.max(),
                        'resolved_count': len(resolved_incidents),
                        'resolution_rate': (len(resolved_incidents) / len(df)) * 100
                    }
            
            # Response time analysis
            if 'sys_created_on' in df.columns and 'u_responded_at' in df.columns:
                df_resp = df.copy()
                df_resp['sys_created_on'] = pd.to_datetime(df_resp['sys_created_on'], errors='coerce')
                df_resp['u_responded_at'] = pd.to_datetime(df_resp['u_responded_at'], errors='coerce')
                
                responded_incidents = df_resp.dropna(subset=['sys_created_on', 'u_responded_at'])
                if not responded_incidents.empty:
                    response_times = (responded_incidents['u_responded_at'] - responded_incidents['sys_created_on']).dt.total_seconds() / 3600  # Hours
                    
                    performance_metrics['response_time_analysis'] = {
                        'average_hours': response_times.mean(),
                        'median_hours': response_times.median(),
                        'response_rate': (len(responded_incidents) / len(df)) * 100
                    }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}
    
    def _generate_analysis_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        try:
            recommendations = []
            
            # Check quality metrics for recommendations
            if hasattr(self, 'quality_metrics') and self.quality_metrics:
                overall_quality = self.quality_metrics.get('overall_quality_score', 100)
                
                if overall_quality < 80:
                    recommendations.append("Overall data quality is below acceptable threshold - implement data quality improvement processes")
                
                completeness = self.quality_metrics.get('completeness', {}).get('overall_score', 100)
                if completeness < 90:
                    recommendations.append("Improve data completeness by implementing mandatory field validation")
                
                consistency = self.quality_metrics.get('consistency', {}).get('overall_score', 100)
                if consistency < 85:
                    recommendations.append("Standardize data entry processes to improve consistency")
            
            # Performance-based recommendations
            if 'performance_metrics' in self.analysis_results:
                perf_metrics = self.analysis_results['performance_metrics']
                
                if 'resolution_time_analysis' in perf_metrics:
                    avg_resolution = perf_metrics['resolution_time_analysis'].get('average_hours', 0)
                    if avg_resolution > 72:  # More than 3 days
                        recommendations.append("Average resolution time exceeds 72 hours - review incident handling processes")
                
                if 'response_time_analysis' in perf_metrics:
                    response_rate = perf_metrics['response_time_analysis'].get('response_rate', 100)
                    if response_rate < 80:
                        recommendations.append("Response rate is below 80% - improve initial response procedures")
            
            if not recommendations:
                recommendations.append("Data quality and performance metrics are within acceptable ranges")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def generate_quality_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report from analysis results.
        This method name matches the interface contract expected by main application.
        
        Args:
            analysis_results: Results from analyze_data_quality method
            
        Returns:
            Formatted quality report dictionary
        """
        try:
            if not analysis_results or 'error' in analysis_results:
                return self._create_error_report("Invalid or missing analysis results")
            
            self.logger.info("Generating comprehensive quality report")
            
            # Create structured quality report using existing analysis data
            quality_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'Data Quality Assessment',
                    'data_source': 'SAP Incident Management System',
                    'total_records': analysis_results.get('dataset_overview', {}).get('total_records', 0)
                },
                'executive_summary': {
                    'data_health_status': analysis_results.get('summary', {}).get('data_health', 'unknown'),
                    'overall_quality_score': analysis_results.get('quality_metrics', {}).get('overall_quality_score', 0),
                    'total_records_analyzed': analysis_results.get('dataset_overview', {}).get('total_records', 0),
                    'key_findings': analysis_results.get('summary', {}).get('key_findings', []),
                    'critical_issues': analysis_results.get('summary', {}).get('critical_issues', [])
                },
                'quality_metrics': analysis_results.get('quality_metrics', {}),
                'detailed_findings': {
                    'dataset_overview': analysis_results.get('dataset_overview', {}),
                    'pattern_analysis': analysis_results.get('pattern_analysis', {}),
                    'trend_analysis': analysis_results.get('trend_analysis', {}),
                    'anomaly_detection': analysis_results.get('anomaly_detection', {}),
                    'performance_metrics': analysis_results.get('performance_metrics', {})
                },
                'recommendations': analysis_results.get('recommendations', []),
                'data_health_score': {
                    'overall_score': analysis_results.get('quality_metrics', {}).get('overall_quality_score', 0),
                    'status': self._determine_health_status(analysis_results.get('quality_metrics', {}).get('overall_quality_score', 0))
                }
            }
            
            self.logger.info("Quality report generated successfully")
            return quality_report
            
        except Exception as e:
            error_msg = f"Error generating quality report: {str(e)}"
            self.logger.error(error_msg)
            return self._create_error_report(error_msg)

    def _create_error_report(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error report following defensive programming."""
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Data Quality Assessment - Error',
                'status': 'failed'
            },
            'error': error_message,
            'executive_summary': {'status': 'failed', 'message': error_message},
            'quality_metrics': {},
            'detailed_findings': {},
            'recommendations': ['Fix critical errors before generating quality report'],
            'data_health_score': {'overall_score': 0, 'status': 'critical'}
        }

    def _determine_health_status(self, score: float) -> str:
        """Determine health status based on quality score."""
        if score >= 90:
            return 'Excellent'
        elif score >= 80:
            return 'Good'
        elif score >= 70:
            return 'Fair'
        elif score >= 60:
            return 'Poor'
        else:
            return 'Critical'
        
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of analysis results."""
        try:
            summary = {
                'status': 'completed',
                'total_records_analyzed': self.analysis_results.get('dataset_overview', {}).get('total_records', 0),
                'overall_quality_score': self.analysis_results.get('quality_metrics', {}).get('overall_quality_score', 0),
                'key_findings': [],
                'critical_issues': [],
                'data_health': 'unknown'
            }
            
            # Determine data health
            quality_score = summary['overall_quality_score']
            if quality_score >= 90:
                summary['data_health'] = 'excellent'
            elif quality_score >= 80:
                summary['data_health'] = 'good'
            elif quality_score >= 70:
                summary['data_health'] = 'fair'
            else:
                summary['data_health'] = 'poor'
            
            # Extract key findings
            if 'pattern_analysis' in self.analysis_results:
                patterns = self.analysis_results['pattern_analysis']
                if 'top_categories' in patterns:
                    top_category = max(patterns['top_categories'], key=patterns['top_categories'].get)
                    summary['key_findings'].append(f"Most common incident category: {top_category}")
            
            # Identify critical issues
            if quality_score < 70:
                summary['critical_issues'].append("Data quality below acceptable threshold")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return {'status': 'error', 'error': str(e)}

# Alias for compatibility following clean code principles
QualityAnalyzer = IncidentQualityAnalyzer