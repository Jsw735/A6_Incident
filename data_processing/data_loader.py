#!/usr/bin/env python3
"""
Incident Data Loader Module
Handles loading and initial processing of incident data from various sources.
Following clean code principles and defensive programming.
"""

import sys
import logging
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd

class IncidentDataLoader:
    """
    Professional data loader for incident management systems.
    Implements robust file handling and error recovery.
    """
    
    def __init__(self):
        """Initialize the data loader with comprehensive logging."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.csv', '.xlsx', '.json']
        self.encoding_attempts = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    def load_csv(self, file_path: Path) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Load incident data from CSV file with comprehensive error handling.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        try:
            if not file_path.exists():
                return False, None, f"File does not exist: {file_path}"
            
            self.logger.info(f"Loading CSV file: {file_path}")
            
            # Try multiple encodings for robust file loading
            df = None
            successful_encoding = None
            
            for encoding in self.encoding_attempts:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    successful_encoding = encoding
                    self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError as e:
                    self.logger.debug(f"Failed to load with {encoding}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Unexpected error with {encoding}: {e}")
                    continue
            
            if df is None:
                return False, None, "Could not decode file with any supported encoding"
            
            if df.empty:
                return False, None, "CSV file is empty"
            
            # Log successful loading
            self.logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            self.logger.info(f"Columns found: {list(df.columns)}")
            
            return True, df, f"Success - loaded {len(df)} records with {successful_encoding} encoding"
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            self.logger.error(error_msg)
            return False, None, error_msg
        except PermissionError:
            error_msg = f"Permission denied accessing file: {file_path}"
            self.logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error loading CSV: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary for validation and reporting.
        Implements defensive programming with comprehensive metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with detailed data summary
        """
        try:
            if df is None:
                return {
                    'total_records': 0,
                    'total_columns': 0,
                    'columns': [],
                    'memory_usage_mb': 0.0,
                    'data_types': {},
                    'missing_data': {},
                    'summary_status': 'error',
                    'error': 'DataFrame is None'
                }
            
            if df.empty:
                return {
                    'total_records': 0,
                    'total_columns': len(df.columns) if hasattr(df, 'columns') else 0,
                    'columns': list(df.columns) if hasattr(df, 'columns') else [],
                    'memory_usage_mb': 0.0,
                    'data_types': {},
                    'missing_data': {},
                    'summary_status': 'empty',
                    'error': 'DataFrame is empty'
                }
            
            # Calculate comprehensive metrics
            total_records = len(df)
            total_columns = len(df.columns)
            columns = list(df.columns)
            
            # Memory usage calculation
            memory_usage_bytes = df.memory_usage(deep=True).sum()
            memory_usage_mb = memory_usage_bytes / (1024 * 1024)
            
            # Data types analysis
            data_types = {}
            for col in df.columns:
                data_types[col] = str(df[col].dtype)
            
            # Missing data analysis
            missing_data = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_percentage = (missing_count / total_records) * 100 if total_records > 0 else 0
                missing_data[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_percentage, 2)
                }
            
            # Data quality indicators
            total_missing = df.isnull().sum().sum()
            completeness_percentage = ((total_records * total_columns - total_missing) / (total_records * total_columns)) * 100 if total_records > 0 and total_columns > 0 else 0
            
            # Sample data preview (first few values for each column)
            sample_data = {}
            for col in df.columns:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_data[col] = list(non_null_values.head(3).astype(str))
                else:
                    sample_data[col] = ['No data']
            
            summary = {
                'total_records': total_records,
                'total_columns': total_columns,
                'columns': columns,
                'memory_usage_mb': round(memory_usage_mb, 2),
                'data_types': data_types,
                'missing_data': missing_data,
                'total_missing_values': int(total_missing),
                'completeness_percentage': round(completeness_percentage, 2),
                'sample_data': sample_data,
                'summary_status': 'success',
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.logger.info(f"Generated data summary: {total_records} records, {total_columns} columns")
            return summary
            
        except Exception as e:
            error_msg = f"Error generating data summary: {str(e)}"
            self.logger.error(error_msg)
            return {
                'total_records': 0,
                'total_columns': 0,
                'columns': [],
                'memory_usage_mb': 0.0,
                'data_types': {},
                'missing_data': {},
                'summary_status': 'error',
                'error': error_msg,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
    
    def validate_data_structure(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate data structure against expected schema.
        Implements comprehensive validation following business rules.
        """
        try:
            if required_columns is None:
                required_columns = ['Number', 'Short Description', 'State', 'Priority']
            
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'validation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            if df is None or df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append('DataFrame is empty or None')
                return validation_results
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                validation_results['is_valid'] = False
            
            # Check for duplicate columns
            duplicate_columns = df.columns[df.columns.duplicated()].tolist()
            if duplicate_columns:
                validation_results['warnings'].append(f"Duplicate column names found: {duplicate_columns}")
            
            # Check for completely empty columns
            empty_columns = [col for col in df.columns if df[col].isnull().all()]
            if empty_columns:
                validation_results['warnings'].append(f"Completely empty columns: {empty_columns}")
            
            self.logger.info(f"Data structure validation completed: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Data structure validation error: {str(e)}"
            self.logger.error(error_msg)
            return {
                'is_valid': False,
                'errors': [error_msg],
                'warnings': [],
                'validation_timestamp': pd.Timestamp.now().isoformat()
            }
    
    def load_data_with_fallback(self, primary_path: Path, fallback_paths: Optional[List[Path]] = None) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Load data with intelligent fallback strategy.
        Implements robust file discovery following defensive programming.
        """
        try:
            # Try primary path first
            success, df, message = self.load_csv(primary_path)
            if success:
                return success, df, f"Loaded from primary path: {message}"
            
            # Try fallback paths if provided
            if fallback_paths:
                for fallback_path in fallback_paths:
                    self.logger.info(f"Trying fallback path: {fallback_path}")
                    success, df, message = self.load_csv(fallback_path)
                    if success:
                        return success, df, f"Loaded from fallback path {fallback_path}: {message}"
            
            return False, None, f"Failed to load from primary path and all fallback paths"
            
        except Exception as e:
            error_msg = f"Error in load_data_with_fallback: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
            
# Class alias for backward compatibility
DataLoader = IncidentDataLoader