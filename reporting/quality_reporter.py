#!/usr/bin/env python3
"""
Quality Reporter for SAP Incident Analysis
Generates data quality analysis and reclassification recommendations.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import logging
from typing import Dict, List, Set, Any, Optional, Union, Tuple

class QualityReporter:
    """
    Generates quality analysis reports with reclassification recommendations.
    Implements proper error handling and mathematical operations.
    """
    
    def __init__(self):
        """Initialize the Quality Reporter."""
        self.logger = logging.getLogger(__name__)
        self.quality_data = {}
    
    def generate_quality_report(self, df: pd.DataFrame, keyword_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive quality analysis report.
        
        Args:
            df: DataFrame containing incident data
            keyword_analysis: Optional keyword analysis results
            
        Returns:
            Dictionary containing quality report data
        """
        try:
            # Calculate data completeness using mathematical operations
            completeness_metrics = self._calculate_completeness(df)
            
            # Identify duplicate records
            duplicate_analysis = self._analyze_duplicates(df)
            
            # Analyze data consistency
            consistency_analysis = self._analyze_consistency(df)
            
            # Generate reclassification candidates
            reclassification_candidates = self._identify_reclassification_candidates(df, keyword_analysis)
            
            # Calculate quality scores using division and multiplication
            quality_scores = self._calculate_quality_scores(completeness_metrics, duplicate_analysis, consistency_analysis)
            
            # Generate improvement recommendations
            improvement_recommendations = self._generate_improvement_recommendations(quality_scores)
            
            self.quality_data = {
                'completeness': completeness_metrics,
                'duplicates': duplicate_analysis,
                'consistency': consistency_analysis,
                'reclassification_candidates': reclassification_candidates,
                'quality_scores': quality_scores,
                'recommendations': improvement_recommendations,
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
            return self.quality_data
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            raise
    
    def _calculate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data completeness using mathematical operations."""
        try:
            total_records = len(df)
            if total_records == 0:
                return {'error': 'No data to analyze'}
            
            # Define critical fields for incident management
            critical_fields = [
                'Number', 'Priority', 'State', 'Assignment group',
                'Short description', 'Created'
            ]
            
            # Calculate completeness for each field using division and multiplication
            field_completeness = {}
            for field in critical_fields:
                if field in df.columns:
                    non_null_count = df[field].notna().sum()  # Addition operation
                    completeness_percentage = (non_null_count / total_records) * 100  # Division and multiplication
                    field_completeness[field] = round(completeness_percentage, 1)
                else:
                    field_completeness[field] = 0.0
            
            # Calculate overall completeness score using addition and division
            total_completeness = sum(field_completeness.values())
            overall_score = total_completeness / len(critical_fields)
            
            # Identify fields with poor completeness using comparison operations
            poor_completeness_fields = [
                field for field, score in field_completeness.items() 
                if score < 90.0
            ]
            
            return {
                'total_records': total_records,
                'field_completeness': field_completeness,
                'overall_completeness_score': round(overall_score, 1),
                'poor_completeness_fields': poor_completeness_fields,
                'critical_fields_analyzed': len(critical_fields)
            }
            
        except Exception as e:
            return {'error': f'Error calculating completeness: {e}'}
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records using mathematical operations."""
        try:
            total_records = len(df)
            
            # Check for exact duplicates across all columns
            exact_duplicates = df.duplicated().sum()  # Addition operation
            
            # Check for potential duplicates based on key fields
            potential_duplicate_groups = []
            
            if 'Short description' in df.columns and 'Assignment group' in df.columns:
                # Group by short description and assignment group
                grouped = df.groupby(['Short description', 'Assignment group'])
                
                for (desc, group), group_df in grouped:
                    if len(group_df) > 1:  # More than one incident with same description and group
                        # Calculate time differences using subtraction
                        if 'Created' in group_df.columns:
                            group_df['Created'] = pd.to_datetime(group_df['Created'])
                            time_diffs = group_df['Created'].diff().dt.total_seconds() / 3600  # Division operation
                            
                            # Consider as potential duplicates if created within 1 hour
                            close_time_incidents = (time_diffs < 1).sum()
                            
                            if close_time_incidents > 0:
                                potential_duplicate_groups.append({
                                    'description': desc[:50],
                                    'assignment_group': group,
                                    'incident_count': len(group_df),
                                    'incidents': group_df['Number'].tolist() if 'Number' in group_df.columns else []
                                })
            
            # Calculate duplicate percentages using division and multiplication
            exact_duplicate_percentage = (exact_duplicates / total_records) * 100 if total_records > 0 else 0
            potential_duplicate_count = sum(len(group['incidents']) for group in potential_duplicate_groups)
            potential_duplicate_percentage = (potential_duplicate_count / total_records) * 100 if total_records > 0 else 0
            
            return {
                'exact_duplicates': int(exact_duplicates),
                'exact_duplicate_percentage': round(exact_duplicate_percentage, 1),
                'potential_duplicate_groups': potential_duplicate_groups,
                'potential_duplicate_count': potential_duplicate_count,
                'potential_duplicate_percentage': round(potential_duplicate_percentage, 1)
            }
            
        except Exception as e:
            return {'error': f'Error analyzing duplicates: {e}'}
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency using mathematical operations."""
        try:
            consistency_issues = []
            
            # Check Priority field consistency
            if 'Priority' in df.columns:
                valid_priorities = ['Critical', 'High', 'Medium', 'Low']
                invalid_priorities = df[~df['Priority'].isin(valid_priorities)]['Priority'].value_counts()
                
                if len(invalid_priorities) > 0:
                    invalid_count = invalid_priorities.sum()  # Addition operation
                    invalid_percentage = (invalid_count / len(df)) * 100  # Division and multiplication
                    
                    consistency_issues.append({
                        'field': 'Priority',
                        'issue': 'Invalid priority values',
                        'count': int(invalid_count),
                        'percentage': round(invalid_percentage, 1),
                        'examples': invalid_priorities.head(3).to_dict()
                    })
            
            # Check State field consistency
            if 'State' in df.columns:
                valid_states = ['New', 'In Progress', 'Resolved', 'Closed', 'Open']
                invalid_states = df[~df['State'].isin(valid_states)]['State'].value_counts()
                
                if len(invalid_states) > 0:
                    invalid_count = invalid_states.sum()
                    invalid_percentage = (invalid_count / len(df)) * 100
                    
                    consistency_issues.append({
                        'field': 'State',
                        'issue': 'Invalid state values',
                        'count': int(invalid_count),
                        'percentage': round(invalid_percentage, 1),
                        'examples': invalid_states.head(3).to_dict()
                    })
            
            # Check for inconsistent assignment group naming
            if 'Assignment group' in df.columns:
                group_variations = self._find_group_variations(df['Assignment group'])
                if group_variations:
                    consistency_issues.append({
                        'field': 'Assignment group',
                        'issue': 'Inconsistent group naming',
                        'count': len(group_variations),
                        'percentage': 0,  # Difficult to calculate percentage for this type of issue
                        'examples': group_variations[:3]
                    })
            
            # Calculate overall consistency score using mathematical operations
            total_issues = sum(issue['count'] for issue in consistency_issues)
            consistency_score = max(0, 100 - (total_issues / len(df) * 100)) if len(df) > 0 else 100
            
            return {
                'consistency_issues': consistency_issues,
                'total_consistency_issues': total_issues,
                'consistency_score': round(consistency_score, 1),
                'issues_found': len(consistency_issues)
            }
            
        except Exception as e:
            return {'error': f'Error analyzing consistency: {e}'}
    
    def _find_group_variations(self, group_series: pd.Series) -> List[str]:
        """Find variations in assignment group names."""
        try:
            # Get unique group names
            unique_groups = group_series.dropna().unique()
            
            # Look for similar names (simple approach)
            variations = []
            for i, group1 in enumerate(unique_groups):
                for group2 in unique_groups[i+1:]:
                    # Check if groups are similar (case-insensitive, ignoring spaces)
                    clean1 = re.sub(r'\s+', '', str(group1).lower())
                    clean2 = re.sub(r'\s+', '', str(group2).lower())
                    
                    if clean1 == clean2 and group1 != group2:
                        variations.append(f"'{group1}' vs '{group2}'")
            
            return variations
            
        except Exception as e:
            self.logger.error(f"Error finding group variations: {e}")
            return []
    
    def _identify_reclassification_candidates(self, df: pd.DataFrame, keyword_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Identify incidents that may need reclassification."""
        candidates = []
        
        try:
            # Priority-based reclassification
            if 'Priority' in df.columns and 'Short description' in df.columns:
                for _, incident in df.iterrows():
                    description = str(incident.get('Short description', '')).lower()
                    current_priority = incident.get('Priority', '')
                    
                    # Look for keywords that suggest different priority
                    urgent_keywords = ['critical', 'urgent', 'down', 'outage', 'emergency', 'production']
                    low_keywords = ['question', 'request', 'how to', 'training', 'documentation']
                    
                    urgent_score = sum(1 for keyword in urgent_keywords if keyword in description)
                    low_score = sum(1 for keyword in low_keywords if keyword in description)
                    
                    # Calculate confidence using division
                    total_words = len(description.split())
                    confidence = 0
                    suggested_priority = current_priority
                    reason = ""
                    
                    if urgent_score > 0 and current_priority in ['Low', 'Medium']:
                        confidence = min((urgent_score / total_words) * 100, 95) if total_words > 0 else 0
                        suggested_priority = 'High'
                        reason = f"Contains {urgent_score} urgent keywords"
                    elif low_score > 0 and current_priority in ['Critical', 'High']:
                        confidence = min((low_score / total_words) * 100, 95) if total_words > 0 else 0
                        suggested_priority = 'Low'
                        reason = f"Contains {low_score} informational keywords"
                    
                    if confidence > 20:  # Only suggest if confidence is reasonable
                        candidates.append({
                            'incident': incident.get('Number', 'N/A'),
                            'current_class': current_priority,
                            'suggested_class': suggested_priority,
                            'confidence': round(confidence, 1),
                            'reason': reason,
                            'field_type': 'Priority'
                        })
            
            # SAP-specific reclassification using keyword analysis
            if keyword_analysis and 'keywords_found' in keyword_analysis:
                # This would use the keyword manager results to suggest better classifications
                # Implementation would depend on the specific keyword analysis structure
                pass
            
            # Sort by confidence (descending) using comparison operations
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return top 20 candidates to avoid overwhelming the report
            return candidates[:20]
            
        except Exception as e:
            self.logger.error(f"Error identifying reclassification candidates: {e}")
            return []
    
    def _calculate_quality_scores(self, completeness: Dict[str, Any], duplicates: Dict[str, Any], consistency: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality scores using mathematical operations."""
        try:
            # Extract scores with error handling
            completeness_score = completeness.get('overall_completeness_score', 0)
            
            # Calculate duplicate score (inverse of duplicate percentage)
            duplicate_percentage = duplicates.get('exact_duplicate_percentage', 0) + duplicates.get('potential_duplicate_percentage', 0)
            duplicate_score = max(0, 100 - duplicate_percentage)  # Subtraction operation
            
            consistency_score = consistency.get('consistency_score', 0)
            
            # Calculate weighted overall score using multiplication and division
            weights = {'completeness': 0.4, 'duplicates': 0.3, 'consistency': 0.3}
            
            overall_score = (
                completeness_score * weights['completeness'] +
                duplicate_score * weights['duplicates'] +
                consistency_score * weights['consistency']
            )
            
            return {
                'completeness_score': round(completeness_score, 1),
                'duplicate_score': round(duplicate_score, 1),
                'consistency_score': round(consistency_score, 1),
                'overall_quality_score': round(overall_score, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating quality scores: {e}")
            return {
                'completeness_score': 0.0,
                'duplicate_score': 0.0,
                'consistency_score': 0.0,
                'overall_quality_score': 0.0
            }
    
    def _generate_improvement_recommendations(self, quality_scores: Dict[str, float]) -> List[str]:
        """Generate actionable improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations using comparison operations
        if quality_scores.get('completeness_score', 0) < 90:
            recommendations.append("Improve data completeness by enforcing mandatory field validation")
        
        if quality_scores.get('completeness_score', 0) < 70:
            recommendations.append("CRITICAL: Implement data quality controls - completeness below acceptable threshold")
        
        # Duplicate recommendations
        if quality_scores.get('duplicate_score', 0) < 95:
            recommendations.append("Implement duplicate detection and prevention mechanisms")
        
        # Consistency recommendations
        if quality_scores.get('consistency_score', 0) < 85:
            recommendations.append("Standardize data entry processes and implement validation rules")
        
        # Overall quality recommendations
        overall_score = quality_scores.get('overall_quality_score', 0)
        if overall_score < 80:
            recommendations.append("PRIORITY: Overall data quality requires immediate attention")
        elif overall_score < 90:
            recommendations.append("Consider implementing automated data quality monitoring")
        
        return recommendations if recommendations else ["Data quality is within acceptable parameters"]
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for quality dashboard display."""
        if not self.quality_data:
            return {'error': 'No quality data available - run generate_quality_report first'}
        
        # Format for dashboard display
        dashboard_data = {
            'quality_scores': self.quality_data.get('quality_scores', {}),
            'top_issues': self._get_top_quality_issues(),
            'reclassification_summary': self._get_reclassification_summary(),
            'improvement_actions': self.quality_data.get('recommendations', [])[:5]  # Top 5 recommendations
        }
        
        return dashboard_data
    
    def _get_top_quality_issues(self) -> List[Dict[str, Any]]:
        """Get top quality issues requiring attention."""
        issues = []
        
        # Completeness issues
        completeness = self.quality_data.get('completeness', {})
        if 'poor_completeness_fields' in completeness:
            for field in completeness['poor_completeness_fields']:
                score = completeness.get('field_completeness', {}).get(field, 0)
                issues.append({
                    'type': 'Completeness',
                    'description': f"{field} field only {score}% complete",
                    'severity': 'High' if score < 70 else 'Medium'
                })
        
        # Duplicate issues
        duplicates = self.quality_data.get('duplicates', {})
        if duplicates.get('exact_duplicates', 0) > 0:
            issues.append({
                'type': 'Duplicates',
                'description': f"{duplicates['exact_duplicates']} exact duplicate records found",
                'severity': 'High'
            })
        
        # Consistency issues
        consistency = self.quality_data.get('consistency', {})
        for issue in consistency.get('consistency_issues', []):
            issues.append({
                'type': 'Consistency',
                'description': f"{issue['field']}: {issue['issue']} ({issue['count']} records)",
                'severity': 'High' if issue['percentage'] > 10 else 'Medium'
            })
        
        return issues[:10]  # Top 10 issues
    
    def _get_reclassification_summary(self) -> Dict[str, Any]:
        """Get summary of reclassification recommendations."""
        candidates = self.quality_data.get('reclassification_candidates', [])
        
        if not candidates:
            return {'total_candidates': 0, 'high_confidence': 0}
        
        # Count high confidence recommendations using comparison operations
        high_confidence = sum(1 for candidate in candidates if candidate.get('confidence', 0) > 70)
        
        return {
            'total_candidates': len(candidates),
            'high_confidence': high_confidence,
            'top_candidate': candidates[0] if candidates else None
        }