#!/usr/bin/env python3
"""
Enhanced Settings module for incident reporting system.
Loads configuration from INI files and provides centralized settings management.
Follows PEP 8 standards and implements comprehensive error handling.
"""

import configparser
import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Settings:
    """
    Centralized settings management for incident reporting system.
    
    This class loads configuration from INI files and provides type-safe
    access to all system settings. It follows the singleton pattern to
    ensure consistent configuration across the application.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_file: str = None):
        """Implement singleton pattern for consistent configuration."""
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = None):
        """
        Initialize settings from configuration file.
        
        Args:
            config_file: Path to configuration file, defaults to executive_config.ini
        """
        if self._initialized:
            return
        
        self.logger = logging.getLogger(__name__)
        
        # Default configuration file path
        if config_file is None:
            config_dir = Path(__file__).parent
            config_file = config_dir / 'executive_config.ini'
        
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize color schemes
        self._initialize_color_schemes()
        
        # Mark as initialized
        self._initialized = True
    
    def _load_configuration(self) -> None:
        """Load configuration from INI file with proper error handling."""
        try:
            if self.config_file.exists():
                self.config.read(self.config_file)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.warning(f"Configuration file not found: {self.config_file}")
                self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration if file doesn't exist."""
        try:
            # Executive Dashboard defaults
            self.config['EXECUTIVE_DASHBOARD'] = {
                'health_score_weights_backlog': '0.4',
                'health_score_weights_closure': '0.3',
                'health_score_weights_accuracy': '0.2',
                'health_score_weights_training': '0.1',
                'target_max_backlog': '300',
                'target_closure_rate': '85.0',
                'min_accuracy_threshold': '80.0',
                'optimal_training_min': '10.0',
                'optimal_training_max': '15.0'
            }
            
            # Trend Analysis defaults
            self.config['TREND_ANALYSIS'] = {
                'comparison_period_weeks': '2',
                'lookback_period_weeks': '4',
                'daily_average_calculation': 'true',
                'trend_smoothing_enabled': 'true'
            }
            
            # Visualization defaults
            self.config['VISUALIZATION'] = {
                'kpi_tile_layout': '5x2',
                'color_scheme': 'professional',
                'print_ready': 'true',
                'powerpoint_ready': 'true'
            }
            
            # SLA Thresholds
            self.config['SLA_THRESHOLDS'] = {
                'P0_hours': '4',
                'P1_hours': '24',
                'P2_hours': '72',
                'P3_hours': '168'
            }
            
            # Urgency Mapping
            self.config['URGENCY_MAPPING'] = {
                'urgency_1': 'High',
                'urgency_2': 'Medium',
                'urgency_3': 'Medium',
                'urgency_4': 'Low'
            }
            
            # Export Settings
            self.config['EXPORT_SETTINGS'] = {
                'default_output_directory': 'output/',
                'excel_template_enabled': 'true',
                'auto_timestamp': 'true',
                'include_metadata': 'true'
            }
            
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save default configuration
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            
            self.logger.info(f"Default configuration created: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating default configuration: {str(e)}")
    
    def _initialize_color_schemes(self) -> None:
        """Initialize professional color schemes for dashboards."""
        self.color_schemes = {
            'professional': {
                'excellent': '#28a745',    # Bootstrap success green
                'good': '#17a2b8',         # Bootstrap info blue
                'warning': '#ffc107',      # Bootstrap warning yellow
                'danger': '#dc3545',       # Bootstrap danger red
                'primary': '#007bff',      # Bootstrap primary blue
                'secondary': '#6c757d',    # Bootstrap secondary gray
                'light': '#f8f9fa',        # Bootstrap light gray
                'dark': '#343a40'          # Bootstrap dark gray
            },
            'executive': {
                'excellent': '#2e7d32',    # Material Design green
                'good': '#1976d2',         # Material Design blue
                'warning': '#f57c00',      # Material Design orange
                'danger': '#d32f2f',       # Material Design red
                'primary': '#1565c0',      # Material Design primary
                'secondary': '#616161'     # Material Design gray
            },
            'corporate': {
                'excellent': '#4caf50',    # Corporate green
                'good': '#2196f3',         # Corporate blue
                'warning': '#ff9800',      # Corporate orange
                'danger': '#f44336',       # Corporate red
                'primary': '#3f51b5',      # Corporate indigo
                'secondary': '#9e9e9e'     # Corporate gray
            }
        }
    
    def get_health_score_weights(self) -> Dict[str, float]:
        """Get health score weights from configuration."""
        try:
            weights = {
                'backlog': self.config.getfloat('EXECUTIVE_DASHBOARD', 'health_score_weights_backlog', fallback=0.4),
                'closure': self.config.getfloat('EXECUTIVE_DASHBOARD', 'health_score_weights_closure', fallback=0.3),
                'accuracy': self.config.getfloat('EXECUTIVE_DASHBOARD', 'health_score_weights_accuracy', fallback=0.2),
                'training': self.config.getfloat('EXECUTIVE_DASHBOARD', 'health_score_weights_training', fallback=0.1)
            }
            
            # Validate weights sum to 1.0
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.logger.warning(f"Health score weights sum to {total_weight}, not 1.0")
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error getting health score weights: {str(e)}")
            return {'backlog': 0.4, 'closure': 0.3, 'accuracy': 0.2, 'training': 0.1}
    
    def get_target_metrics(self) -> Dict[str, Union[float, tuple]]:
        """Get target metrics from configuration."""
        try:
            return {
                'max_backlog': self.config.getfloat('EXECUTIVE_DASHBOARD', 'target_max_backlog', fallback=300.0),
                'closure_rate': self.config.getfloat('EXECUTIVE_DASHBOARD', 'target_closure_rate', fallback=85.0),
                'min_accuracy': self.config.getfloat('EXECUTIVE_DASHBOARD', 'min_accuracy_threshold', fallback=80.0),
                'training_range': (
                    self.config.getfloat('EXECUTIVE_DASHBOARD', 'optimal_training_min', fallback=10.0),
                    self.config.getfloat('EXECUTIVE_DASHBOARD', 'optimal_training_max', fallback=15.0)
                )
            }
        except Exception as e:
            self.logger.error(f"Error getting target metrics: {str(e)}")
            return {'max_backlog': 300.0, 'closure_rate': 85.0, 'min_accuracy': 80.0, 'training_range': (10.0, 15.0)}
    
    def get_trend_analysis_config(self) -> Dict[str, Any]:
        """Get trend analysis configuration."""
        try:
            return {
                'comparison_weeks': self.config.getint('TREND_ANALYSIS', 'comparison_period_weeks', fallback=2),
                'lookback_weeks': self.config.getint('TREND_ANALYSIS', 'lookback_period_weeks', fallback=4),
                'daily_average': self.config.getboolean('TREND_ANALYSIS', 'daily_average_calculation', fallback=True),
                'smoothing_enabled': self.config.getboolean('TREND_ANALYSIS', 'trend_smoothing_enabled', fallback=True),
                'min_data_points': self.config.getint('TREND_ANALYSIS', 'minimum_data_points', fallback=7)
            }
        except Exception as e:
            self.logger.error(f"Error getting trend analysis config: {str(e)}")
            return {'comparison_weeks': 2, 'lookback_weeks': 4, 'daily_average': True}
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        try:
            return {
                'tile_layout': self.config.get('VISUALIZATION', 'kpi_tile_layout', fallback='5x2'),
                'color_scheme': self.config.get('VISUALIZATION', 'color_scheme', fallback='professional'),
                'print_ready': self.config.getboolean('VISUALIZATION', 'print_ready', fallback=True),
                'powerpoint_ready': self.config.getboolean('VISUALIZATION', 'powerpoint_ready', fallback=True),
                'chart_types': self.config.get('VISUALIZATION', 'chart_types', fallback='line,pie,stacked_bar').split(','),
                'export_format': self.config.get('VISUALIZATION', 'export_format', fallback='xlsx')
            }
        except Exception as e:
            self.logger.error(f"Error getting visualization config: {str(e)}")
            return {'tile_layout': '5x2', 'color_scheme': 'professional', 'print_ready': True}
    
    def get_color_scheme(self, scheme_name: str = None) -> Dict[str, str]:
        """Get color scheme for dashboards."""
        if scheme_name is None:
            scheme_name = self.config.get('VISUALIZATION', 'color_scheme', fallback='professional')
        
        return self.color_schemes.get(scheme_name, self.color_schemes['professional'])
    
    def get_sla_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get SLA thresholds from configuration."""
        try:
            sla_data = {}
            for priority in ['P0', 'P1', 'P2', 'P3']:
                hours = self.config.getint('SLA_THRESHOLDS', f'{priority}_hours', fallback=72)
                description = self.config.get('SLA_THRESHOLDS', f'{priority}_description', fallback=f'{priority} Incident')
                
                sla_data[priority] = {
                    'hours': hours,
                    'description': description
                }
            
            return sla_data
            
        except Exception as e:
            self.logger.error(f"Error getting SLA thresholds: {str(e)}")
            return {
                'P0': {'hours': 4, 'description': 'Critical Incident'},
                'P1': {'hours': 24, 'description': 'High Incident'},
                'P2': {'hours': 72, 'description': 'Medium Incident'},
                'P3': {'hours': 168, 'description': 'Low Incident'}
            }
    
    def get_urgency_mapping(self) -> Dict[str, str]:
        """Get urgency level mapping."""
        try:
            return {
                '1': self.config.get('URGENCY_MAPPING', 'urgency_1', fallback='High'),
                '2': self.config.get('URGENCY_MAPPING', 'urgency_2', fallback='Medium'),
                '3': self.config.get('URGENCY_MAPPING', 'urgency_3', fallback='Medium'),
                '4': self.config.get('URGENCY_MAPPING', 'urgency_4', fallback='Low')
            }
        except Exception as e:
            self.logger.error(f"Error getting urgency mapping: {str(e)}")
            return {'1': 'High', '2': 'Medium', '3': 'Medium', '4': 'Low'}
    
    def get_export_settings(self) -> Dict[str, Any]:
        """Get export settings."""
        try:
            return {
                'output_directory': self.config.get('EXPORT_SETTINGS', 'default_output_directory', fallback='output/'),
                'excel_template': self.config.getboolean('EXPORT_SETTINGS', 'excel_template_enabled', fallback=True),
                'auto_timestamp': self.config.getboolean('EXPORT_SETTINGS', 'auto_timestamp', fallback=True),
                'include_metadata': self.config.getboolean('EXPORT_SETTINGS', 'include_metadata', fallback=True),
                'filename_prefix': self.config.get('EXPORT_SETTINGS', 'filename_prefix', fallback='executive_summary')
            }
        except Exception as e:
            self.logger.error(f"Error getting export settings: {str(e)}")
            return {'output_directory': 'output/', 'excel_template': True, 'auto_timestamp': True}
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        try:
            return {
                'log_level': self.config.get('LOGGING', 'log_level', fallback='INFO'),
                'log_file': self.config.get('LOGGING', 'log_file', fallback='logs/executive_summary.log'),
                'max_size_mb': self.config.getint('LOGGING', 'max_log_size_mb', fallback=10),
                'backup_count': self.config.getint('LOGGING', 'backup_count', fallback=5),
                'console_logging': self.config.getboolean('LOGGING', 'console_logging', fallback=True)
            }
        except Exception as e:
            self.logger.error(f"Error getting logging config: {str(e)}")
            return {'log_level': 'INFO', 'log_file': 'logs/executive_summary.log'}
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization settings."""
        try:
            return {
                'cache_enabled': self.config.getboolean('PERFORMANCE', 'cache_enabled', fallback=True),
                'cache_timeout': self.config.getint('PERFORMANCE', 'cache_timeout_minutes', fallback=30),
                'parallel_processing': self.config.getboolean('PERFORMANCE', 'parallel_processing', fallback=False),
                'max_workers': self.config.getint('PERFORMANCE', 'max_workers', fallback=4)
            }
        except Exception as e:
            self.logger.error(f"Error getting performance config: {str(e)}")
            return {'cache_enabled': True, 'cache_timeout': 30}
    
    def get_keyword_config(self) -> Dict[str, Any]:
        """Get keyword enhancement settings."""
        try:
            return {
                'enhance_from_recent': self.config.getboolean('KEYWORDS', 'enhance_from_recent_data', fallback=True),
                'recent_data_weeks': self.config.getint('KEYWORDS', 'recent_data_weeks', fallback=4),
                'min_frequency': self.config.getint('KEYWORDS', 'minimum_keyword_frequency', fallback=3),
                'max_keywords': self.config.getint('KEYWORDS', 'max_enhanced_keywords', fallback=10)
            }
        except Exception as e:
            self.logger.error(f"Error getting keyword config: {str(e)}")
            return {'enhance_from_recent': True, 'recent_data_weeks': 4}
    
    def update_setting(self, section: str, key: str, value: str) -> bool:
        """
        Update a configuration setting and save to file.
        
        Args:
            section: Configuration section name
            key: Setting key
            value: New value as string
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if section not in self.config:
                self.config.add_section(section)
            
            self.config.set(section, key, value)
            
            # Save to file
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            
            self.logger.info(f"Updated setting: {section}.{key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating setting: {str(e)}")
            return False
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration settings and return validation results.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Validate health score weights
            weights = self.get_health_score_weights()
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                validation_results['warnings'].append(f"Health score weights sum to {total_weight}, not 1.0")
            
            # Validate target metrics
            targets = self.get_target_metrics()
            if targets['max_backlog'] <= 0:
                validation_results['errors'].append("Max backlog must be positive")
            
            if not (0 <= targets['closure_rate'] <= 100):
                validation_results['errors'].append("Closure rate must be between 0 and 100")
            
            # Validate SLA thresholds
            sla_thresholds = self.get_sla_thresholds()
            for priority, data in sla_thresholds.items():
                if data['hours'] <= 0:
                    validation_results['errors'].append(f"SLA hours for {priority} must be positive")
            
            # Set overall validity
            validation_results['is_valid'] = len(validation_results['errors']) == 0
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Configuration validation error: {str(e)}")
        
        return validation_results
    
    def reload_configuration(self) -> bool:
        """Reload configuration from file."""
        try:
            self.config.clear()
            self._load_configuration()
            self.logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {str(e)}")
            return False