def __init__(self):
    """Initialize the tester with comprehensive module coverage."""
    self.setup_logging()
    self.test_results = {}
    self.modules_to_test = [
        # Data processing modules
        'data.data_processor',
        'data.data_validator', 
        'data.file_handler',
        
        # Analysis modules
        'analysis.incident_analyzer',
        'analysis.keyword_analyzer',
        'analysis.metrics_calculator',
        'analysis.quality_analyzer',
        
        # Reporting modules
        'reporting.excel_generator',
        'reporting.daily_reporter',
        'reporting.weekly_reporter',
        'reporting.executive_summary',
        'reporting.quality_reporter',
        
        # Utility modules (ADD THESE)
        'utils.config_manager',
        'utils.logger_setup',
        'utils.date_utils',
        'utils.file_utils',
        'utils.validation_utils',
        'utils.format_utils',
        'utils.constants',
        'utils.helpers'
    ]