#!/usr/bin/env python3
"""
Incident Data Validator Module
Handles comprehensive data validation and quality assessment.
Following clean code principles and defensive programming.
"""

import sys
import logging
import re
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

class IncidentDataValidator:
    """
    Professional data validator for incident management systems.
    Implements comprehensive data quality assessment following clean code principles.
    """
    
    def __init__(self):
        """Initialize the data validator with comprehensive logging and configuration."""
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize data quality thresholds for validation."""
        return {
            'completeness_threshold': 0.80,  # 80% completeness required
            'uniqueness_threshold': 0.95,    # 95% uniqueness for key fields
            'validity_threshold': 0.90,      # 90% valid data required
            'consistency_threshold': 0.85    # 85% consistency required
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize comprehensive validation rules for incident data."""
        return {
            'required_fields': ['number', 'state', 'short_description'],
            'unique_fields': ['number'],
            'date_fields': ['sys_created_on', 'resolved_at', 'u_responded_at', 'sys_updated_on'],
            'numeric_fields': ['urgency', 'sys_mod_count'],
            'text_fields': ['short_description', 'description', 'work_notes'],
            'categorical_fields': ['state', 'category', 'subcategory', 'assignment_group'],
            'valid_states': ['New', 'Open', 'In Progress', 'Resolved', 'Closed', 'Cancelled'],
            'valid_urgencies': [1, 2, 3, 4, 5],
            'max_text_length': 5000,
            'min_description_length': 5
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method for comprehensive data quality validation.
        This method name matches the interface contract expected by main application.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Comprehensive validation results dictionary
        """
        try:
            if df is None or df.empty:
                return self._create_error_result("DataFrame is empty or None")
            
            self.logger.info(f"Starting data quality validation for {len(df)} records")
            
            # Initialize validation results structure
            self.validation_results = {
                'is_valid': True,
                'overall_score': 0.0,
                'total_records': len(df),
                'total_columns': len(df.columns),
                'validation_timestamp': datetime.now().isoformat(),
                'errors': [],
                'warnings': [],
                'quality_metrics': {},
                'field_validations': {},
                'recommendations': []
            }
            
            # Run comprehensive validation pipeline
            self._validate_structure(df)
            self._validate_completeness(df)
            self._validate_uniqueness(df)
            self._validate_data_types(df)
            self._validate_business_rules(df)
            self._validate_data_consistency(df)
            self._calculate_overall_quality_score()
            self._generate_recommendations()
            
            self.logger.info(f"Data quality validation completed. Overall score: {self.validation_results['overall_score']:.2f}")
            return self.validation_results
            
        except Exception as e:
            error_msg = f"Error in validate_data_quality: {str(e)}"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Alias method for backward compatibility.
        Delegates to validate_data_quality for consistent interface.
        """
        return self.validate_data_quality(df)
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result following defensive programming."""
        return {
            'is_valid': False,
            'overall_score': 0.0,
            'total_records': 0,
            'total_columns': 0,
            'validation_timestamp': datetime.now().isoformat(),
            'errors': [error_message],
            'warnings': [],
            'quality_metrics': {},
            'field_validations': {},
            'recommendations': ['Fix critical errors before proceeding']
        }
    
    def _validate_structure(self, df: pd.DataFrame):
        """Validate basic data structure and required fields."""
        try:
            structure_score = 100.0
            
            # Check for required fields
            missing_required = []
            for field in self.validation_rules['required_fields']:
                if field not in df.columns:
                    missing_required.append(field)
                    structure_score -= 30.0
            
            if missing_required:
                error_msg = f"Missing required fields: {missing_required}"
                self.validation_results['errors'].append(error_msg)
                self.validation_results['is_valid'] = False
            
            # Check for empty DataFrame
            if len(df) == 0:
                self.validation_results['errors'].append("DataFrame contains no records")
                self.validation_results['is_valid'] = False
                structure_score = 0.0
            
            self.validation_results['quality_metrics']['structure_score'] = max(0.0, structure_score)
            
        except Exception as e:
            self.logger.error(f"Error validating structure: {str(e)}")
            self.validation_results['errors'].append(f"Structure validation error: {str(e)}")
    
    def _validate_completeness(self, df: pd.DataFrame):
        """Validate data completeness across all fields."""
        try:
            completeness_scores = {}
            overall_completeness = 0.0
            
            for column in df.columns:
                non_null_count = df[column].notna().sum()
                completeness_ratio = non_null_count / len(df) if len(df) > 0 else 0.0
                completeness_scores[column] = {
                    'completeness_ratio': completeness_ratio,
                    'missing_count': len(df) - non_null_count,
                    'score': completeness_ratio * 100
                }
                
                # Check against threshold for required fields
                if column in self.validation_rules['required_fields']:
                    if completeness_ratio < self.quality_thresholds['completeness_threshold']:
                        warning_msg = f"Field '{column}' has low completeness: {completeness_ratio:.2%}"
                        self.validation_results['warnings'].append(warning_msg)
            
            # Calculate overall completeness score
            if completeness_scores:
                overall_completeness = sum(score['score'] for score in completeness_scores.values()) / len(completeness_scores)
            
            self.validation_results['quality_metrics']['completeness_score'] = overall_completeness
            self.validation_results['field_validations']['completeness'] = completeness_scores
            
        except Exception as e:
            self.logger.error(f"Error validating completeness: {str(e)}")
            self.validation_results['errors'].append(f"Completeness validation error: {str(e)}")
    
    def _validate_uniqueness(self, df: pd.DataFrame):
        """Validate uniqueness constraints for key fields."""
        try:
            uniqueness_scores = {}
            overall_uniqueness = 100.0
            
            for field in self.validation_rules['unique_fields']:
                if field in df.columns:
                    total_count = len(df[df[field].notna()])
                    unique_count = df[field].nunique()
                    uniqueness_ratio = unique_count / total_count if total_count > 0 else 0.0
                    
                    uniqueness_scores[field] = {
                        'uniqueness_ratio': uniqueness_ratio,
                        'duplicate_count': total_count - unique_count,
                        'score': uniqueness_ratio * 100
                    }
                    
                    if uniqueness_ratio < self.quality_thresholds['uniqueness_threshold']:
                        error_msg = f"Field '{field}' has duplicate values: {total_count - unique_count} duplicates"
                        self.validation_results['errors'].append(error_msg)
                        overall_uniqueness -= 25.0
            
            self.validation_results['quality_metrics']['uniqueness_score'] = max(0.0, overall_uniqueness)
            self.validation_results['field_validations']['uniqueness'] = uniqueness_scores
            
        except Exception as e:
            self.logger.error(f"Error validating uniqueness: {str(e)}")
            self.validation_results['errors'].append(f"Uniqueness validation error: {str(e)}")
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Validate data types and format consistency."""
        try:
            type_scores = {}
            overall_type_score = 100.0
            
            # Validate date fields
            for field in self.validation_rules['date_fields']:
                if field in df.columns:
                    valid_dates = pd.to_datetime(df[field], errors='coerce').notna().sum()
                    total_non_null = df[field].notna().sum()
                    validity_ratio = valid_dates / total_non_null if total_non_null > 0 else 1.0
                    
                    type_scores[field] = {
                        'expected_type': 'datetime',
                        'validity_ratio': validity_ratio,
                        'invalid_count': total_non_null - valid_dates,
                        'score': validity_ratio * 100
                    }
                    
                    if validity_ratio < self.quality_thresholds['validity_threshold']:
                        warning_msg = f"Field '{field}' has invalid date formats: {total_non_null - valid_dates} invalid"
                        self.validation_results['warnings'].append(warning_msg)
            
            # Validate numeric fields
            for field in self.validation_rules['numeric_fields']:
                if field in df.columns:
                    valid_numbers = pd.to_numeric(df[field], errors='coerce').notna().sum()
                    total_non_null = df[field].notna().sum()
                    validity_ratio = valid_numbers / total_non_null if total_non_null > 0 else 1.0
                    
                    type_scores[field] = {
                        'expected_type': 'numeric',
                        'validity_ratio': validity_ratio,
                        'invalid_count': total_non_null - valid_numbers,
                        'score': validity_ratio * 100
                    }
                    
                    if validity_ratio < self.quality_thresholds['validity_threshold']:
                        warning_msg = f"Field '{field}' has invalid numeric values: {total_non_null - valid_numbers} invalid"
                        self.validation_results['warnings'].append(warning_msg)
            
            # Calculate overall type score
            if type_scores:
                overall_type_score = sum(score['score'] for score in type_scores.values()) / len(type_scores)
            
            self.validation_results['quality_metrics']['data_type_score'] = overall_type_score
            self.validation_results['field_validations']['data_types'] = type_scores
            
        except Exception as e:
            self.logger.error(f"Error validating data types: {str(e)}")
            self.validation_results['errors'].append(f"Data type validation error: {str(e)}")
    
    def _validate_business_rules(self, df: pd.DataFrame):
        """Validate business-specific rules and constraints."""
        try:
            business_rule_score = 100.0
            rule_violations = []
            
            # Validate state values
            if 'state' in df.columns:
                invalid_states = df[~df['state'].isin(self.validation_rules['valid_states'])]['state'].unique()
                if len(invalid_states) > 0:
                    rule_violations.append(f"Invalid state values found: {list(invalid_states)}")
                    business_rule_score -= 20.0
            
            # Validate urgency values
            if 'urgency' in df.columns:
                numeric_urgency = pd.to_numeric(df['urgency'], errors='coerce')
                invalid_urgency_count = (~numeric_urgency.isin(self.validation_rules['valid_urgencies'])).sum()
                if invalid_urgency_count > 0:
                    rule_violations.append(f"Invalid urgency values found: {invalid_urgency_count} records")
                    business_rule_score -= 15.0
            
            # Validate description length
            if 'short_description' in df.columns:
                short_descriptions = df['short_description'].astype(str)
                too_short_count = (short_descriptions.str.len() < self.validation_rules['min_description_length']).sum()
                if too_short_count > 0:
                    rule_violations.append(f"Descriptions too short: {too_short_count} records")
                    business_rule_score -= 10.0
            
            # Add rule violations to warnings
            for violation in rule_violations:
                self.validation_results['warnings'].append(violation)
            
            self.validation_results['quality_metrics']['business_rules_score'] = max(0.0, business_rule_score)
            
        except Exception as e:
            self.logger.error(f"Error validating business rules: {str(e)}")
            self.validation_results['errors'].append(f"Business rules validation error: {str(e)}")
    
    def _validate_data_consistency(self, df: pd.DataFrame):
        """Validate data consistency and logical relationships."""
        try:
            consistency_score = 100.0
            consistency_issues = []
            
            # Check date consistency (created <= updated <= resolved)
            date_fields = ['sys_created_on', 'sys_updated_on', 'resolved_at']
            available_date_fields = [field for field in date_fields if field in df.columns]
            
            if len(available_date_fields) >= 2:
                for i, df_row in df.iterrows():
                    try:
                        dates = {}
                        for field in available_date_fields:
                            if pd.notna(df_row[field]):
                                dates[field] = pd.to_datetime(df_row[field])
                        
                        # Check logical date order
                        if 'sys_created_on' in dates and 'sys_updated_on' in dates:
                            if dates['sys_created_on'] > dates['sys_updated_on']:
                                consistency_issues.append(f"Created date after updated date in record {i}")
                        
                        if 'sys_updated_on' in dates and 'resolved_at' in dates:
                            if dates['sys_updated_on'] > dates['resolved_at']:
                                consistency_issues.append(f"Updated date after resolved date in record {i}")
                                
                    except Exception:
                        continue  # Skip problematic records
            
            # Limit consistency issues reporting to avoid overwhelming output
            if len(consistency_issues) > 10:
                self.validation_results['warnings'].append(f"Date consistency issues found in {len(consistency_issues)} records")
                consistency_score -= min(50.0, len(consistency_issues) * 2)
            else:
                for issue in consistency_issues:
                    self.validation_results['warnings'].append(issue)
                consistency_score -= len(consistency_issues) * 5
            
            self.validation_results['quality_metrics']['consistency_score'] = max(0.0, consistency_score)
            
        except Exception as e:
            self.logger.error(f"Error validating consistency: {str(e)}")
            self.validation_results['errors'].append(f"Consistency validation error: {str(e)}")
    
    def _calculate_overall_quality_score(self):
        """Calculate weighted overall quality score."""
        try:
            metrics = self.validation_results['quality_metrics']
            
            # Define weights for different quality aspects
            weights = {
                'structure_score': 0.25,
                'completeness_score': 0.25,
                'uniqueness_score': 0.20,
                'data_type_score': 0.15,
                'business_rules_score': 0.10,
                'consistency_score': 0.05
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    weighted_score += metrics[metric] * weight
                    total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                self.validation_results['overall_score'] = weighted_score / total_weight
            else:
                self.validation_results['overall_score'] = 0.0
            
            # Set validation status based on score
            if self.validation_results['overall_score'] < 70.0:
                self.validation_results['is_valid'] = False
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {str(e)}")
            self.validation_results['overall_score'] = 0.0
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on validation results."""
        try:
            recommendations = []
            
            # Recommendations based on quality scores
            metrics = self.validation_results['quality_metrics']
            
            if metrics.get('completeness_score', 100) < 80:
                recommendations.append("Improve data completeness by addressing missing values in key fields")
            
            if metrics.get('uniqueness_score', 100) < 95:
                recommendations.append("Remove duplicate records to improve data uniqueness")
            
            if metrics.get('data_type_score', 100) < 90:
                recommendations.append("Standardize data formats, especially for dates and numeric fields")
            
            if metrics.get('business_rules_score', 100) < 85:
                recommendations.append("Review and correct business rule violations")
            
            if metrics.get('consistency_score', 100) < 85:
                recommendations.append("Address data consistency issues, particularly date relationships")
            
            if len(self.validation_results['errors']) > 0:
                recommendations.append("Critical errors must be resolved before data processing")
            
            if not recommendations:
                recommendations.append("Data quality is excellent - no major issues detected")
            
            self.validation_results['recommendations'] = recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            self.validation_results['recommendations'] = ["Unable to generate recommendations due to validation error"]

# Alias for compatibility following clean code principles
DataValidator = IncidentDataValidator