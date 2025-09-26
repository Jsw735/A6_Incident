#!/usr/bin/env python3
"""
Enhanced Incident Data Cleaner Module
Handles comprehensive data cleaning with robust column mapping for all reporters.
"""

import sys
import logging
import re
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

class IncidentDataCleaner:
    """
    Enhanced data cleaner with comprehensive column mapping for all reporting modules.
    Implements robust data standardization following clean code principles.
    """
    
    def __init__(self):
        """Initialize the enhanced data cleaner with comprehensive mapping."""
        self.logger = logging.getLogger(__name__)
        self.cleaning_stats = {}
        self.validation_rules = self._initialize_validation_rules()
        self.column_mapping = self._initialize_comprehensive_column_mapping()
        # attempt to load Settings (optional)
        try:
            from config.settings import Settings
            self.settings = Settings()
        except Exception:
            self.settings = None
    
    def _initialize_comprehensive_column_mapping(self) -> Dict[str, List[str]]:
        """Initialize comprehensive column mapping that matches ALL reporter expectations."""
        return {
            # Date columns - CRITICAL for weekly/daily reporters
            'created_date': [
                'Created', 'Created Date', 'Date Created', 'Opened', 'Open Date',
                'sys_created_on', 'creation_date', 'incident_created', 'created_on',
                'date_opened', 'open_time', 'created_time', 'creation_time'
            ],
            'resolved_date': [
                'Resolved', 'Resolved Date', 'Date Resolved', 'Closed', 'Close Date',
                'resolved_at', 'resolution_date', 'closed_date', 'date_closed',
                'completion_date', 'end_date', 'resolved_on', 'closed_on'
            ],
            
            # Assignment columns - CRITICAL for workstream analysis
            'assignment_group': [
                'Assignment Group', 'Assigned To', 'Team', 'Support Group',
                'assigned_group', 'support_team', 'resolver_group',
                'assignment_team', 'responsible_team'
            ],
            
            # Priority columns - CRITICAL for priority analysis
            'priority': [
                'Priority', 'Incident Priority', 'Severity', 'Urgency',
                'priority_level', 'incident_severity', 'urgency_level'
            ],
            
            # Status columns - CRITICAL for status tracking
            'status': [
                'State', 'Status', 'Incident State', 'Current State',
                'incident_status', 'current_status', 'state_name'
            ],
            
            # Keep existing mappings for backward compatibility
            'number': [
                'Number', 'Incident Number', 'ID', 'Incident ID', 'Ticket Number',
                'incident_number', 'ticket_id', 'case_number', 'ref_number'
            ],
            'short_description': [
                'Short Description', 'Summary', 'Title', 'Subject',
                'short_desc', 'incident_summary', 'description_short'
            ]
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules - updated to use new column names."""
        return {
            'required_columns': ['number', 'status', 'short_description'],
            'date_columns': ['created_date', 'resolved_date'],  # Updated names
            # Removed 'urgency'/'priority' from numeric coercion to preserve label strings like '1 - High'
            'numeric_columns': [],
            'text_columns': ['short_description', 'description', 'work_notes'],
            'categorical_columns': ['status', 'category', 'subcategory', 'assignment_group']
        }
    
    def clean_incident_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced main method with comprehensive column mapping.
        """
        try:
            if df is None or df.empty:
                self.logger.warning("Received empty or None DataFrame")
                return df
            
            self.logger.info(f"Starting enhanced data cleaning for {len(df)} records")
            self.logger.info(f"Original columns: {list(df.columns)}")
            
            # Initialize cleaning statistics
            self.cleaning_stats = {
                'original_records': len(df),
                'original_columns': len(df.columns),
                'columns_mapped': 0,
                'duplicates_removed': 0,
                'missing_values_handled': 0,
                'text_standardized': 0,
                'dates_standardized': 0
            }
            
            # Create a copy to avoid modifying original data
            cleaned_df = df.copy()
            
            # STEP 1: Apply comprehensive column mapping FIRST
            cleaned_df = self._apply_comprehensive_column_mapping(cleaned_df)
            
            # STEP 2: Apply other cleaning operations
            cleaned_df = self._remove_duplicates(cleaned_df)
            cleaned_df = self._standardize_dates(cleaned_df)
            cleaned_df = self._clean_text_fields(cleaned_df)
            cleaned_df = self._handle_missing_values(cleaned_df)
            cleaned_df = self._standardize_categorical_data(cleaned_df)
            cleaned_df = self._validate_data_types(cleaned_df)
            
            # Update final statistics
            self.cleaning_stats['final_records'] = len(cleaned_df)
            self.cleaning_stats['final_columns'] = len(cleaned_df.columns)
            # Ensure canonical output columns required by downstream analyzers
            cleaned_df = self._ensure_canonical_output_columns(cleaned_df)

            self.logger.info(f"Enhanced data cleaning completed: {len(cleaned_df)} records processed")
            self.logger.info(f"Final columns: {list(cleaned_df.columns)}")
            self._log_cleaning_summary()
            
            return cleaned_df
            
        except Exception as e:
            error_msg = f"Error in clean_incident_data: {str(e)}"
            self.logger.error(error_msg)
            return df
    
    def _apply_comprehensive_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive column mapping to standardize ALL column names."""
        try:
            mapped_columns = {}
            columns_mapped = 0
            
            # Create case-insensitive lookup of current columns
            current_columns_lower = {col.lower().strip(): col for col in df.columns}
            
            self.logger.info("=== COLUMN MAPPING ANALYSIS ===")
            self.logger.info(f"Available columns: {list(df.columns)}")
            
            # Apply mapping for each standard column
            for standard_name, possible_names in self.column_mapping.items():
                mapped = False
                
                for possible_name in possible_names:
                    possible_lower = possible_name.lower().strip()
                    
                    if possible_lower in current_columns_lower:
                        original_name = current_columns_lower[possible_lower]
                        
                        if original_name != standard_name:
                            mapped_columns[original_name] = standard_name
                            columns_mapped += 1
                            self.logger.info(f"MAPPED: '{original_name}' -> '{standard_name}'")
                            mapped = True
                            break
                        else:
                            self.logger.info(f"ALREADY CORRECT: '{original_name}'")
                            mapped = True
                            break
                
                if not mapped:
                    self.logger.warning(f"NOT FOUND: No column found for '{standard_name}' in {possible_names}")
            
            # Apply the mapping
            if mapped_columns:
                df = df.rename(columns=mapped_columns)
                self.cleaning_stats['columns_mapped'] = columns_mapped
                self.logger.info(f"Successfully mapped {columns_mapped} columns")
            else:
                self.logger.warning("No columns were mapped - check your data structure")
            
            # Log final column status
            self.logger.info("=== FINAL COLUMN STATUS ===")
            for standard_name in self.column_mapping.keys():
                if standard_name in df.columns:
                    self.logger.info(f"{standard_name}: AVAILABLE")
                else:
                    self.logger.warning(f"{standard_name}: MISSING")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error applying column mapping: {str(e)}")
            return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced date standardization using the mapped column names."""
        try:
            date_columns = self.validation_rules.get('date_columns', [])
            dates_processed = 0
            
            for col in date_columns:
                if col in df.columns:
                    try:
                        original_count = df[col].notna().sum()
                        self.logger.info(f"Processing date column '{col}' with {original_count} non-null values")
                        
                        # Convert to datetime with error handling
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        
                        converted_count = df[col].notna().sum()
                        if converted_count < original_count:
                            lost_dates = original_count - converted_count
                            self.logger.warning(f"Lost {lost_dates} dates during conversion of '{col}'")
                        
                        dates_processed += 1
                        self.logger.info(f"Standardized '{col}': {converted_count} valid dates")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not convert {col} to datetime: {str(e)}")
            
            self.cleaning_stats['dates_standardized'] = dates_processed
            return df
            
        except Exception as e:
            self.logger.error(f"Error standardizing dates: {str(e)}")
            return df
    
    # Keep all your existing methods but update them to use the new column names
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records with comprehensive logging."""
        try:
            original_count = len(df)
            
            # Remove exact duplicates
            df_deduplicated = df.drop_duplicates()
            
            # Remove duplicates based on incident number if available
            if 'number' in df.columns:
                df_deduplicated = df_deduplicated.drop_duplicates(subset=['number'], keep='first')
            
            duplicates_removed = original_count - len(df_deduplicated)
            self.cleaning_stats['duplicates_removed'] = duplicates_removed
            
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate records")
            
            return df_deduplicated
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates: {str(e)}")
            return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields."""
        try:
            text_columns = df.select_dtypes(include=['object']).columns
            text_fields_processed = 0
            
            for col in text_columns:
                if col in df.columns:
                    # Remove leading/trailing whitespace
                    df[col] = df[col].astype(str).str.strip()
                    
                    # Replace multiple spaces with single space
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    
                    # Handle common text issues
                    df[col] = df[col].replace(['nan', 'NaN', 'NULL', 'null'], pd.NA)
                    
                    text_fields_processed += 1
            
            self.cleaning_stats['text_standardized'] = text_fields_processed
            
            if text_fields_processed > 0:
                self.logger.info(f"Cleaned {text_fields_processed} text fields")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning text fields: {str(e)}")
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent strategies."""
        try:
            missing_handled = 0
            
            # Handle missing values in categorical columns
            categorical_columns = self.validation_rules.get('categorical_columns', [])
            for col in categorical_columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        df[col] = df[col].fillna('Unknown')
                        missing_handled += missing_count
            
            # Handle missing values in text columns
            text_columns = self.validation_rules.get('text_columns', [])
            for col in text_columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        df[col] = df[col].fillna('No description provided')
                        missing_handled += missing_count
            
            self.cleaning_stats['missing_values_handled'] = missing_handled
            
            if missing_handled > 0:
                self.logger.info(f"Handled {missing_handled} missing values")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def _standardize_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical data for consistency."""
        try:
            categorical_columns = self.validation_rules.get('categorical_columns', [])
            
            for col in categorical_columns:
                if col in df.columns:
                    # Standardize case and remove extra spaces
                    df[col] = df[col].astype(str).str.strip().str.title()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error standardizing categorical data: {str(e)}")
            return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types where appropriate."""
        try:
            # Convert numeric columns
            numeric_columns = self.validation_rules.get('numeric_columns', [])
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating data types: {str(e)}")
            return df
    
    def _log_cleaning_summary(self):
        """Log comprehensive cleaning summary for debugging and monitoring."""
        try:
            self.logger.info("=== DATA CLEANING SUMMARY ===")
            for key, value in self.cleaning_stats.items():
                self.logger.info(f"{key}: {value}")
            self.logger.info("=== END CLEANING SUMMARY ===")
        except Exception as e:
            self.logger.error(f"Error logging cleaning summary: {str(e)}")

    def _ensure_canonical_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create canonical columns `Created`, `Resolved`, and `Priority` from the mapped fields.

        Uses the cleaned/mapped names (created_date, resolved_date, priority) and
        produces DataFrame columns with the exact names expected by analyzers.
        """
        try:
            # Work on a copy
            out = df.copy()

            # Created
            if 'Created' not in out.columns:
                for src in ['created_date', 'created', 'Created']:
                    if src in out.columns:
                        try:
                            out['Created'] = pd.to_datetime(out[src], errors='coerce')
                            self.logger.info(f"Canonical column Created created from {src}")
                            break
                        except Exception:
                            continue

            # Resolved
            if 'Resolved' not in out.columns:
                for src in ['resolved_date', 'resolved', 'Resolved']:
                    if src in out.columns:
                        try:
                            out['Resolved'] = pd.to_datetime(out[src], errors='coerce')
                            self.logger.info(f"Canonical column Resolved created from {src}")
                            break
                        except Exception:
                            continue

            # Priority (normalize names)
            if 'Priority' not in out.columns:
                for src in ['priority', 'Priority', 'urgency', 'Urgency']:
                    if src in out.columns:
                        try:
                            out['Priority'] = out[src]
                            # Normalize priority values using settings mapping when available
                            try:
                                if self.settings is not None:
                                    mapping = self.settings.get_urgency_mapping()
                                    if isinstance(out['Priority'], pd.Series):
                                        def map_val(v):
                                            if pd.isna(v):
                                                return v
                                            s = str(v).strip()
                                            # numeric strings like '1' or '1.0'
                                            try:
                                                if s.replace('.', '', 1).isdigit():
                                                    key = str(int(float(s)))
                                                    return mapping.get(key, s)
                                            except Exception:
                                                pass
                                            return mapping.get(s, s)

                                        out['Priority'] = out['Priority'].apply(map_val)

                            except Exception:
                                # if mapping fails, leave values as-is
                                pass

                            self.logger.info(f"Canonical column Priority created from {src}")
                            break
                        except Exception:
                            continue

            return out
        except Exception as e:
            self.logger.error(f"Error ensuring canonical output columns: {e}")
            return df
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics for reporting and analysis."""
        return self.cleaning_stats.copy()
    
    # Backward compatibility
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias method for backward compatibility."""
        return self.clean_incident_data(df)

# Alias for compatibility following clean code principles
DataCleaner = IncidentDataCleaner