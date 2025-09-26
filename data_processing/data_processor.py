#!/usr/bin/env python3
"""
Main data processing orchestrator for incident data pipeline.
Coordinates loading, cleaning, validation, and processing operations.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
from typing import Dict, List, Set, Any, Optional, Union, Tuple

# Import our custom classes
from .data_loader import IncidentDataLoader
from .data_cleaner import IncidentDataCleaner
from .data_validator import IncidentDataValidator


class IncidentDataProcessor:
    """
    Main orchestrator class that coordinates the entire data processing pipeline.
    Implements the facade pattern to provide a simple interface to complex operations.
    """
    
    def __init__(self):
        """Initialize the processor with component instances."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize component classes
        self.loader = IncidentDataLoader()
        self.cleaner = IncidentDataCleaner()
        self.validator = IncidentDataValidator()
        
        # Processing statistics
        self.processing_stats = {}
    
    def process_data(self, data_source: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline that handles the complete data workflow.
        
        Args:
            data_source: Input DataFrame to process
            
        Returns:
            Processed and validated DataFrame
        """
        try:
            self.logger.info("Starting comprehensive data processing pipeline")
            
            # Step 1: Initial validation
            self.logger.info("Step 1: Initial data validation")
            initial_validation = self.validator.validate_data(data_source)
            self.processing_stats['initial_validation'] = initial_validation
            
            # Step 2: Data cleaning
            self.logger.info("Step 2: Data cleaning and standardization")
            cleaned_data = self.cleaner.clean_data(data_source)
            
            # Step 3: Post-cleaning validation
            self.logger.info("Step 3: Post-cleaning validation")
            final_validation = self.validator.validate_data(cleaned_data)
            self.processing_stats['final_validation'] = final_validation
            
            # Step 4: Generate processing summary
            self.processing_stats['summary'] = self._generate_processing_summary(
                data_source, cleaned_data, initial_validation, final_validation
            )
            
            self.logger.info("Data processing pipeline completed successfully")
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Data processing pipeline failed: {str(e)}")
            raise
    
    def process_file(self, file_path: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Process data from a file through the complete pipeline.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (success, processed_dataframe, message)
        """
        try:
            # Load data from file
            success, raw_data, error_msg = self.loader.load_csv(Path(file_path))
            
            if not success:
                return False, None, f"Failed to load file: {error_msg}"
            
            # Process the loaded data
            processed_data = self.process_data(raw_data)
            
            return True, processed_data, "File processed successfully"
            
        except Exception as e:
            error_msg = f"File processing failed: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return comprehensive results.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        return self.validator.validate_data(df)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics from the last processing operation.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.processing_stats.copy()
    
    def export_processing_report(self, output_path: str) -> bool:
        """
        Export processing statistics to a JSON report file.
        
        Args:
            output_path: Path where to save the report
            
        Returns:
            Success status
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.processing_stats, f, indent=2, default=str)
            
            self.logger.info(f"Processing report exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {str(e)}")
            return False
    
    def generate_executive_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Orchestrate executive summary generation by calling metrics calculator.
        Maintains separation of concerns - processor orchestrates, calculator computes.
        """
        try:
            # Call metrics calculator for business logic
            executive_metrics = self.metrics_calculator.calculate_executive_summary_metrics(df)
            
            # Add to processing stats for tracking
            self.processing_stats['executive_summary'] = executive_metrics
            
            return executive_metrics
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return {}
    
    def _generate_processing_summary(self, original_df: pd.DataFrame, 
                                   processed_df: pd.DataFrame,
                                   initial_validation: Dict[str, Any],
                                   final_validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive processing summary.
        
        Args:
            original_df: Original DataFrame
            processed_df: Processed DataFrame
            initial_validation: Initial validation results
            final_validation: Final validation results
            
        Returns:
            Processing summary dictionary
        """
        summary = {
            'records_processed': {
                'original_count': len(original_df),
                'final_count': len(processed_df),
                'records_removed': len(original_df) - len(processed_df)
            },
            'quality_improvement': {
                'initial_quality_score': initial_validation.get('quality_score', 0),
                'final_quality_score': final_validation.get('quality_score', 0),
                'improvement': final_validation.get('quality_score', 0) - initial_validation.get('quality_score', 0)
            },
            'data_completeness': {
                'initial_completeness': initial_validation.get('completeness', {}).get('overall_completeness', 0),
                'final_completeness': final_validation.get('completeness', {}).get('overall_completeness', 0)
            },
            'processing_success': final_validation.get('is_valid', False)
        }
        
        return summary


# Alias for backward compatibility and test integration
DataProcessor = IncidentDataProcessor