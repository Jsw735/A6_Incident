"""
Data processing package for incident reporting system.
Provides comprehensive data loading, cleaning, validation, and processing capabilities.

This package follows clean architecture principles with proper encapsulation,
interface design, and modular components for robust data processing workflows.
"""

# Import main classes for easy access
from .data_loader import IncidentDataLoader, DataLoader
from .data_cleaner import IncidentDataCleaner, DataCleaner
from .data_validator import IncidentDataValidator, DataValidator
from .data_processor import IncidentDataProcessor, DataProcessor

# Package metadata
__version__ = "1.0.0"
__author__ = "Incident Reporting System"

# Define what gets imported with "from data_processing import *"
__all__ = [
    'IncidentDataLoader',
    'DataLoader', 
    'IncidentDataCleaner',
    'DataCleaner',
    'IncidentDataValidator',    
    'DataValidator',
    'IncidentDataProcessor',
    'DataProcessor'
]

# Package-level configuration
import logging

# Configure package-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Convenience functions for quick access
def get_loader():
    """Get a configured data loader instance."""
    return IncidentDataLoader()

def get_cleaner():
    """Get a configured data cleaner instance."""
    return IncidentDataCleaner()

def get_validator():
    """Get a configured data validator instance."""
    return IncidentDataValidator()

def get_processor():
    """Get a configured data processor instance."""
    return IncidentDataProcessor()