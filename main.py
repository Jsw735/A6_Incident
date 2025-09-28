import logging
from pathlib import Path
from typing import Optional

#!/usr/bin/env python3
"""
SAP S/4HANA Incident Reporting System
Main entry point with modular architecture and comprehensive error handling
Following PEP 8 guidelines and clean code principles
"""

def safe_input(message: str = "Press Enter to close...") -> None:
    """
    Always pause before closing, even on errors.
    Implements graceful error handling as shown in the context documents.
    """
    try:
        input(f"\n{message}")
    except (KeyboardInterrupt, EOFError):
        print("\nClosing...")
    except Exception:
        print("\nForced close...")


def import_modules_safely():
    """
    Import all modules with comprehensive error handling.
    Following proper Python import patterns and error handling.
    """
    try:
        import sys
        import logging
        import argparse
        import pandas as pd
        from pathlib import Path
        from datetime import datetime
        from typing import Dict, List, Set, Any, Optional, Union, Tuple

        # Data Processing Components
        from data_processing.data_loader import DataLoader
        from data_processing.data_cleaner import IncidentDataCleaner
        from data_processing.data_validator import DataValidator

        # Analysis Components
        from analysis.quality_analyzer import QualityAnalyzer
        from analysis.trend_analyzer import TrendAnalyzer
        from analysis.metrics_calculator import MetricsCalculator
        from analysis.keyword_manager import KeywordManager

        # Reporting Components
        from reporting.executive_summary import EnhancedExecutiveSummary
        from reporting.weekly_reporter import WeeklyReporter
        from reporting.daily_reporter import DailyReporter

        # Utility Components
        from utils.excel_utils import ExcelUtils
        from utils.date_utils import DateUtils
        from utils.text_analyzer import TextAnalyzer

        # Configuration - CORRECTED IMPORTS
        from config.settings import Settings  # Import the CLASS (capitalized)
        from config.business_rules import BusinessRules
        
        return True, None, {
            'sys': sys, 'logging': logging, 'argparse': argparse, 'pd': pd,
            'Path': Path, 'datetime': datetime, 'Optional': Optional, 'Dict': Dict, 'Any': Any,
            'DataLoader': DataLoader, 'IncidentDataCleaner': IncidentDataCleaner,
            'DataValidator': DataValidator, 'QualityAnalyzer': QualityAnalyzer,
            'TrendAnalyzer': TrendAnalyzer, 'MetricsCalculator': MetricsCalculator,
            'KeywordManager': KeywordManager, 'EnhancedExecutiveSummary': EnhancedExecutiveSummary,
            'WeeklyReporter': WeeklyReporter, 'DailyReporter': DailyReporter,
            'ExcelUtils': ExcelUtils,
            'DateUtils': DateUtils, 'TextAnalyzer': TextAnalyzer,
            'Settings': Settings, 'BusinessRules': BusinessRules  # Use capitalized class names
        }
        
    except ImportError as e:
        error_msg = f"Import Error: {str(e)}"
        print(f"ERROR: {error_msg}")
        print("\nThis usually means:")
        print("1. A required module file is missing")
        print("2. A module has syntax errors") 
        print("3. Required packages aren't installed (try: pip install pandas openpyxl)")
        print("4. Python path issues")
        print("5. Incorrect import names (check class vs instance names)")
        print(f"\nSpecific import that failed: {e}")
        return False, error_msg, None
        
    except Exception as e:
        error_msg = f"Unexpected import error: {str(e)}"
        print(f"ERROR: {error_msg}")
        return False, error_msg, None

def setup_logging(modules) -> 'logging.Logger':
    """
    Configure comprehensive logging system with file and console output.
    Creates timestamped log files for debugging and audit trails.
    
    Returns:
        Configured logger instance
    """
    try:
        logging = modules['logging']
        Path = modules['Path']
        datetime = modules['datetime']
        sys = modules['sys']
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_dir / f"incident_reporting_{timestamp}.log"
        
        # Configure logging format following best practices
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )
        
        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("SAP S/4HANA Incident Reporting System Started")
        logger.info(f"Log file: {log_filename.absolute()}")
        logger.info("=" * 60)
        
        return logger
        
    except Exception as e:
        print(f"ERROR: Failed to setup logging: {e}")
        return None


def find_incidents_csv(modules) -> 'Optional[Path]':
    """
    Automatically locate incidents CSV file in common locations.
    Implements clean file discovery pattern following context guidance.
    
    Returns:
        Path object or None if not found
    """
    try:
        Path = modules['Path']
        logging = modules['logging']
        logger = logging.getLogger(__name__)
        
        # Common locations to check for incidents CSV
        possible_locations = [
            Path('data/incidents.csv'),
            Path('incidents.csv'),
            Path('data/incident_data.csv'),
            Path('data/servicenow_export.csv'),
            Path('ServiceNow_incidents.csv'),
            Path('data/sample_data.csv')  # Added for testing
        ]
        
        for csv_path in possible_locations:
            if csv_path.exists():
                logger.info(f"Found incidents CSV: {csv_path.absolute()}")
                return csv_path
        
        logger.warning("No incidents CSV file found in standard locations")
        return None
        
    except Exception as e:
        print(f"ERROR: Error finding CSV file: {e}")
        return None


def process_incident_data(csv_path, modules):
    """
    Process incident data using optimized consolidated pipeline.
    PRESERVES existing functionality while improving performance.
    """
    logging = modules.get('logging')
    Path = modules.get('Path')
    logger = logging.getLogger(__name__) if logging else None

    try:
        # Validate file exists before processing
        if not csv_path.exists():
            error_msg = f"CSV file does not exist: {csv_path}"
            if logger:
                logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return None

        # NEW: Use consolidated pipeline for improved performance
        try:
            from data_processing.consolidated_pipeline import ConsolidatedDataPipeline
            pipeline = ConsolidatedDataPipeline()
            
            print(f"Loading incident data using optimized pipeline: {csv_path}")
            processed_data = pipeline.process_incident_file_optimized(csv_path, modules)
            
            if processed_data is not None:
                # Show performance stats
                stats = pipeline.get_performance_stats()
                improvements = pipeline.benchmark_improvements()
                
                print(f"✅ Optimized processing completed:")
                print(f"  - {stats.get('records_processed', 0):,} records in {stats.get('total_time', 0):.3f}s")
                print(f"  - Processing rate: {stats.get('processing_rate', 0):.0f} records/sec")
                print(f"  - Memory usage: {stats.get('memory_usage_mb', 0):.2f} MB")
                
                return processed_data
            else:
                print("⚠️  Optimized pipeline failed, falling back to original method")
                # Fall through to original code below
        except ImportError:
            logger and logger.info("Consolidated pipeline not available, using original method")
        except Exception as e:
            logger and logger.warning(f"Optimized pipeline failed: {e}, using fallback")

        # FALLBACK: Original loading method (preserved for compatibility)
        print(f"Loading incident data from: {csv_path}")
        DataLoader = modules['DataLoader']
        loader = DataLoader()

        # Use the load_csv method which returns a tuple (success, dataframe, message)
        try:
            success, raw_df, message = loader.load_csv(Path(csv_path))
        except Exception as load_error:
            if logger:
                logger.error(f"Failed to load incidents: {load_error}")
            print(f"ERROR: Data Loading Error: {load_error}")
            return None

        if not success or raw_df is None:
            err_msg = message or "Failed to load data"
            if logger:
                logger.error(f"Failed to load incidents: {err_msg}")
            print(f"ERROR: Data Loading Error: {err_msg}")
            return None

        logger and logger.info(f"Data loading successful: {message}")

        # Validate raw_df is not empty
        if raw_df.empty:
            error_msg = "Loaded DataFrame is empty"
            if logger:
                logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return None

        # Display data summary for raw input
        print(f"Loaded {len(raw_df):,} records with {len(raw_df.columns)} columns")
        print(f"\nAvailable Columns (raw):")
        for i, col in enumerate(raw_df.columns, 1):
            print(f"  {i:2d}. {col}")

        # Run data cleaner (if available) to map columns to canonical names expected by validators/analyzers
        DataCleanerClass = modules.get('IncidentDataCleaner') or modules.get('DataCleaner')
        if DataCleanerClass:
            try:
                cleaner = DataCleanerClass()
                cleaned_df = cleaner.clean_incident_data(raw_df)
                print(f"\nData cleaned. Records: {len(cleaned_df):,}, Columns: {len(cleaned_df.columns)}")
                print(f"\nAvailable Columns (cleaned):")
                for i, col in enumerate(cleaned_df.columns, 1):
                    print(f"  {i:2d}. {col}")
            except Exception as e:
                if logger:
                    logger.warning(f"Data cleaner failed: {e}")
                print(f"WARNING: Data cleaner failed: {e}")
                cleaned_df = raw_df
        else:
            cleaned_df = raw_df

        # Ensure validator-required column names exist (normalize synonyms)
        # Validator expects fields like 'state' while cleaner may produce 'status'
        if 'state' not in cleaned_df.columns and 'status' in cleaned_df.columns:
            cleaned_df['state'] = cleaned_df['status']
        if 'number' not in cleaned_df.columns and 'Number' in cleaned_df.columns:
            cleaned_df['number'] = cleaned_df['Number']
        if 'short_description' not in cleaned_df.columns and 'Short description' in cleaned_df.columns:
            cleaned_df['short_description'] = cleaned_df['Short description']

        # Validate data using dedicated validator on the cleaned dataframe
        print(f"\nValidating data quality...")
        DataValidator = modules['DataValidator']
        validator = DataValidator()
        validation_results = validator.validate_data(cleaned_df)

        if validation_results.get('is_valid', False):
            print(f"Data validation passed")
        else:
            warnings = validation_results.get('warnings', [])
            errors = validation_results.get('errors', [])
            if errors:
                print(f"ERROR: Data validation errors: {len(errors)} critical issues found")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"   - {error}")
                return None
            elif warnings:
                print(f"WARNING: Data validation warnings: {len(warnings)} issues found")
                for warning in warnings[:3]:  # Show first 3 warnings
                    print(f"   - {warning}")

        print(f"Data processing completed")
        if logger:
            logger.info("Data processing completed successfully")

        return cleaned_df

    except Exception as e:
        error_msg = f"Unexpected error in data processing: {str(e)}"
        print(f"ERROR: {error_msg}")
        if 'logging' in modules:
            logger = modules['logging'].getLogger(__name__)
            logger.exception(error_msg)
        return None


def generate_executive_summary(data, modules, args, analysis_results=None, health_score=None):
    """
    Generate enhanced executive summary using the new EnhancedExecutiveSummary class.
    Implements the CIO-level dashboard with weighted health scoring.
    """
    try:
        logging = modules['logging']
        logger = logging.getLogger(__name__)

        if data is None or data.empty:
            logger.error("Cannot generate executive summary: No data available")
            print("ERROR: Cannot generate executive summary: No data available")
            return None

        print("Generating Enhanced Executive Summary...")
        logger.info("Starting executive summary generation")

        # Initialize the enhanced executive summary
        EnhancedExecutiveSummary = modules['EnhancedExecutiveSummary']
        executive_generator = EnhancedExecutiveSummary()

        # Generate the dashboard using the data already loaded
        dashboard_data = executive_generator.generate_executive_dashboard(
            data, analysis_results or {}, health_score
        )

        if not dashboard_data:
            logger.error("Executive dashboard generation returned empty results")
            print("ERROR: Executive summary generation failed")
            return None

        # The executive generator returns a dict with status and output_file keys
        output_file = None
        if isinstance(dashboard_data, dict):
            output_file = dashboard_data.get('output_file') or dashboard_data.get('file')
        elif isinstance(dashboard_data, str):
            output_file = dashboard_data

        if output_file:
            print(f"Executive Summary generated: {output_file}")
            logger.info(f"Executive summary exported to: {output_file}")

            # Display key metrics from the dashboard if available
            try:
                metrics = dashboard_data.get('metrics', {}) if isinstance(dashboard_data, dict) else {}
                overall_health = metrics.get('overall_health_score', None) if isinstance(metrics, dict) else None
                health_status = metrics.get('health_status', 'Unknown') if isinstance(metrics, dict) else 'Unknown'
                if overall_health is not None:
                    print(f"Overall Health Score: {overall_health:.1f}% ({health_status})")
            except Exception:
                pass

            return output_file
        else:
            print("ERROR: Failed to export executive summary")
            return None

    except Exception as e:
        error_msg = f"Error generating executive summary: {str(e)}"
        print(f"ERROR: {error_msg}")
        if 'logging' in modules:
            logger = modules['logging'].getLogger(__name__)
            logger.exception(error_msg)
        return None


def run_comprehensive_analysis(data, args, modules):
    """Execute comprehensive incident analysis using available analyzers."""
    logging = modules.get('logging')
    logger = logging.getLogger(__name__) if logging else None
    results = {}

    if data is None or data.empty:
        if logger:
            logger.error("Cannot run analysis on empty or None DataFrame")
        print("ERROR: Cannot run analysis: No data available")
        return results

    # Quality Analysis
    print("Running data quality analysis...")
    try:
        QualityAnalyzer = modules['QualityAnalyzer']
        quality_analyzer = QualityAnalyzer()
        quality_results = quality_analyzer.analyze_data_quality(data)
        results['quality'] = quality_results
        accuracy = quality_results.get('accuracy_percentage', 0)
        print(f"Quality analysis complete - Accuracy: {accuracy:.1f}%")
    except Exception as e:
        if logger:
            logger.error(f"Quality analysis failed: {e}")
        print(f"WARNING: Quality analysis failed: {e}")

    if not getattr(args, 'no_pause', False):
        interactive_pause("Quality analysis complete. Continue with trend analysis? ")

    # Trend Analysis
    print("Running trend analysis...")
    try:
        TrendAnalyzer = modules['TrendAnalyzer']
        trend_analyzer = TrendAnalyzer()
        trend_results = trend_analyzer.analyze_trends(data)
        results['trends'] = trend_results
        print("Trend analysis complete")
    except Exception as e:
        if logger:
            logger.error(f"Trend analysis failed: {e}")
        print(f"WARNING: Trend analysis failed: {e}")

    if not getattr(args, 'no_pause', False):
        interactive_pause("Trend analysis complete. Continue with metrics calculation? ")

    # Metrics Calculation
    print("Calculating key metrics...")
    try:
        MetricsCalculator = modules['MetricsCalculator']
        metrics_calculator = MetricsCalculator()
        metrics_results = metrics_calculator.calculate_all_metrics(data)
        results['metrics'] = metrics_results
        
        # Extract metrics from the structured results
        core_metrics = metrics_results.get('core_metrics', {})
        total_incidents = core_metrics.get('total_incidents', metrics_results.get('total_records_analyzed', 0))
        open_incidents = core_metrics.get('open_incidents', 0)
        print(f"Metrics calculated - Total: {total_incidents:,}, Open: {open_incidents:,}")
    except Exception as e:
        if logger:
            logger.error(f"Metrics calculation failed: {e}")
        print(f"WARNING: Metrics calculation failed: {e}")

    # Keyword Analysis
    print("Analyzing keywords and patterns...")
    try:
        KeywordManager = modules['KeywordManager']
        keyword_manager = KeywordManager()
        keyword_results = keyword_manager.learn_from_data(data)
        results['keywords'] = keyword_results
        training_keywords = keyword_results.get('training_keywords', [])
        print(f"Keyword analysis complete - {len(training_keywords)} training keywords identified")
    except Exception as e:
        if logger:
            logger.error(f"Keyword analysis failed: {e}")
        print(f"WARNING: Keyword analysis failed: {e}")

    if logger:
        logger.info("Comprehensive analysis completed successfully")
    return results


def generate_all_reports(data, analysis_results, args, modules):
    """
    Generate all types of reports using the reporting modules.
    Implements coordinated reporting workflow with enhanced executive summary.
    """
    try:
        logging = modules['logging']
        Path = modules['Path']
        logger = logging.getLogger(__name__)
        report_files = {}

        # Validate inputs
        if data is None or data.empty:
            logger.error("Cannot generate reports: No data available")
            print("ERROR: Cannot generate reports: No data available")
            return report_files

        # Ensure output directory exists
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Generate Enhanced Executive Summary (highest priority)
        print("Generating Enhanced Executive Summary...")
        exec_file = generate_executive_summary(data, modules, args, analysis_results, None)
        if exec_file:
            report_files['executive'] = str(exec_file)

        # Generate Daily Report
        print("Generating daily report...")
        try:
            DailyReporter = modules.get('DailyReporter')
            if DailyReporter:
                daily_reporter = DailyReporter()
                daily_file = daily_reporter.generate_daily_report(data, analysis_results)
                if daily_file:
                    report_files['daily'] = str(daily_file)
                    print(f"Daily report: {daily_file}")
        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")
            print(f"WARNING: Daily report generation failed: {e}")

        # Generate Weekly Report
        print("Generating weekly report...")
        try:
            WeeklyReporter = modules.get('WeeklyReporter')
            if WeeklyReporter:
                weekly_reporter = WeeklyReporter()
                weekly_file = weekly_reporter.generate_weekly_report(data, analysis_results)
                if weekly_file:
                    report_files['weekly'] = str(weekly_file)
                    print(f"Weekly report: {weekly_file}")
        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")
            print(f"WARNING: Weekly report generation failed: {e}")

        # Archive old incidents_kpi_report files (keep only 2 most recent)
        try:
            print("\nArchiving old report files...")
            EnhancedExecutiveSummary = modules.get('EnhancedExecutiveSummary')
            if EnhancedExecutiveSummary and 'executive' in report_files:
                # Create a temporary instance just for archiving
                archiver = EnhancedExecutiveSummary()
                archiver.archive_kpi_reports_only()
        except Exception as e:
            logger.error(f"Report archiving failed: {e}")
            print(f"WARNING: Report archiving failed: {e}")

        logger.info(f"Generated {len(report_files)} reports successfully")
        return report_files

    except Exception as e:
        error_msg = f"Error in report generation: {str(e)}"
        print(f"ERROR: Report generation error: {error_msg}")
        if 'logging' in modules:
            logger = modules['logging'].getLogger(__name__)
            logger.exception(error_msg)
        return {}


def validate_environment(modules):
    """
    Validate that all required directories and dependencies exist.
    Implements comprehensive environment checking following best practices.
    """
    try:
        logging = modules['logging']
        Path = modules['Path']
        logger = logging.getLogger(__name__)
        issues = []
        
        # Check required directories
        required_dirs = [
            "config", "data_processing", "analysis", 
            "reporting", "utils", "tests", "data", "output"
        ]
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                issues.append(f"Missing directory: {directory}")
                logger.error(f"Directory not found: {dir_path.absolute()}")
            else:
                logger.info(f"Directory validated: {directory}")
        
        # Check for critical configuration files
        critical_files = [
            "config/settings.py",
            "config/business_rules.py",
        ]
        
        for file_path in critical_files:
            file_obj = Path(file_path)
            if not file_obj.exists():
                issues.append(f"Missing critical file: {file_path}")
                logger.error(f"Critical file not found: {file_obj.absolute()}")
            else:
                logger.info(f"Critical file validated: {file_path}")
        
        success = len(issues) == 0
        if success:
            logger.info("Environment validation completed successfully")
        else:
            logger.warning(f"Environment validation failed with {len(issues)} issues")
        
        return success, issues
        
    except Exception as e:
        print(f"❌ Error in environment validation: {e}")
        return False, [f"Validation error: {e}"]


def interactive_pause(message: str = "Press Enter to continue, 'q' to quit: ") -> None:
    """
    Pause execution for user review with clean exit option.
    Follows clean code principles with clear user interaction as shown in context.
    """
    try:
        user_input = input(f"\n{message}").strip().lower()
        if user_input in ['q', 'quit', 'exit']:
            print("Exiting application...")
            safe_input("Press Enter to close...")
            exit(0)
    except KeyboardInterrupt:
        print("\nExiting application...")
        safe_input("Press Enter to close...")
        exit(0)
    except Exception:
        # Continue if input fails
        pass


def display_system_status(validation_success: bool, issues: list, modules) -> None:
    """Display comprehensive system status with clear formatting."""
    try:
        Path = modules['Path']

        print("\n" + "=" * 60)
        print("SYSTEM STATUS REPORT")
        print("=" * 60)

        if validation_success:
            print("Environment Validation: PASSED")
        else:
            print("Environment Validation: FAILED")
            print("\nIssues Found:")
            for issue in issues:
                print(f"  - {issue}")

        print(f"\nLog Directory: {Path('logs').absolute()}")
        print(f"Project Root: {Path('.').absolute()}")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Error displaying system status: {e}")


def display_final_summary(data, analysis_results, report_files, modules) -> None:
    """Display comprehensive final summary of all processing results."""
    try:
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)

        # Data Summary - with safe access
        if data is not None and not data.empty:
            print("Data Processing:")
            print(f"   - Records processed: {len(data):,}")
            print(f"   - Columns analyzed: {len(data.columns)}")
        else:
            print("Data Processing: No data processed")

        # Analysis Summary - with safe dictionary access
        print("\nAnalysis Results:")
        if 'quality' in analysis_results and analysis_results['quality']:
            quality_score = analysis_results['quality'].get('accuracy_percentage', 0)
            print(f"   - Data quality score: {quality_score:.1f}%")

        if 'trends' in analysis_results and analysis_results['trends']:
            print(f"   - Trend analysis completed")

        if 'metrics' in analysis_results and analysis_results['metrics']:
            metrics_count = len(analysis_results['metrics'])
            print(f"   - Metrics calculated: {metrics_count}")

        # Reports Summary
        print("\nGenerated Reports:")
        if report_files:
            for report_type, file_path in report_files.items():
                print(f"   - {report_type.title()}: {file_path}")
        else:
            print("   - No reports generated")

        print("\nProcessing completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Error displaying final summary: {e}")


def create_argument_parser(modules):
    """
    Create and configure command line argument parser.
    Following context guidance on sys.argv usage and command line handling.
    """
    try:
        argparse = modules['argparse']
        
        parser = argparse.ArgumentParser(
            description='SAP S/4HANA Incident Reporting System with Enhanced Executive Summary',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py                          # Auto-detect incidents.csv
  python main.py data/incidents.csv       # Use specific CSV file
  python main.py --skip-validation        # Skip environment checks
  python main.py --no-pause              # Run without user interaction
  python main.py --executive-only         # Generate only executive summary
            """
        )
        
        parser.add_argument(
            'csv_file', 
            nargs='?',
            help='Path to incidents CSV file (default: auto-detect incidents.csv)'
        )
        parser.add_argument(
            '--skip-validation', 
            action='store_true',
            help='Skip environment validation checks'
        )
        parser.add_argument(
            '--no-pause', 
            action='store_true',
            help='Run without interactive pauses'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging output'
        )
        parser.add_argument(
            '--executive-only',
            action='store_true',
            help='Generate only the enhanced executive summary'
        )
        
        return parser
        
    except Exception as e:
        print(f"❌ Error creating argument parser: {e}")
        return None


def main() -> None:
    """Main application entry point with comprehensive error handling."""
    print("SAP S/4HANA Incident Reporting System")
    print("Enhanced with Executive Dashboard")
    print("=" * 40)
    print("Starting up...")

    try:
        # First, try to import all modules
        print("Loading required modules...")
        import_success, import_error, modules = import_modules_safely()

        if not import_success:
            print("ERROR: Cannot start application due to import errors:")
            print(f"   {import_error}")
            safe_input("Fix the import issues and try again. Press Enter to close...")
            return

        print("All modules loaded successfully")

        # Set up logging
        logger = setup_logging(modules)
        if not logger:
            print("WARNING: Continuing without logging...")

        # Parse command line arguments
        parser = create_argument_parser(modules)
        if not parser:
            safe_input("Failed to create argument parser. Press Enter to close...")
            return

        args = parser.parse_args()
        if logger:
            logger.info(f"Command line arguments: {vars(args)}")

        # Adjust logging level if verbose
        if getattr(args, 'verbose', False) and logger:
            modules['logging'].getLogger().setLevel(modules['logging'].DEBUG)
            logger.debug("Verbose logging enabled")

        # Display startup banner
        print("Initializing system components...")

        # Validate environment unless skipped
        if not getattr(args, 'skip_validation', False):
            if logger:
                logger.info("Starting environment validation")
            validation_success, issues = validate_environment(modules)
            display_system_status(validation_success, issues, modules)

            if not getattr(args, 'no_pause', False):
                interactive_pause("Review system status. Continue? ")

            if not validation_success:
                if logger:
                    logger.error("Cannot proceed with validation failures")
                if not getattr(args, 'no_pause', False):
                    interactive_pause("Fix issues and restart. Press Enter to exit: ")
                return
        else:
            if logger:
                logger.info("Environment validation skipped by user")

        # Handle CSV file detection and processing
        csv_path = None
        Path = modules['Path']

        if getattr(args, 'csv_file', None):
            # User specified CSV file
            csv_path = Path(args.csv_file)
            if logger:
                logger.info(f"User specified CSV file: {csv_path}")

            # Validate user-specified file exists
            if not csv_path.exists():
                print(f"ERROR: Specified CSV file does not exist: {csv_path}")
                if logger:
                    logger.error(f"User-specified CSV file not found: {csv_path}")
                if not getattr(args, 'no_pause', False):
                    interactive_pause("Press Enter to exit: ")
                return
        else:
            # Auto-detect incidents.csv
            if logger:
                logger.info("Auto-detecting incidents CSV file")
            csv_path = find_incidents_csv(modules)

            if not csv_path:
                print("ERROR: No incidents.csv file found!")
                print("Please place your CSV file in one of these locations:")
                print("  - data/incidents.csv")
                print("  - data/sample_data.csv")
                print("  - incidents.csv")
                if not getattr(args, 'no_pause', False):
                    interactive_pause("Press Enter to exit: ")
                return

        # Process incident data
        processed_data = process_incident_data(csv_path, modules)

        if processed_data is None:
            if logger:
                logger.error("Data processing failed")
            if not getattr(args, 'no_pause', False):
                interactive_pause("Press Enter to exit: ")
            return

        if not getattr(args, 'no_pause', False):
            interactive_pause("Data loaded successfully. Continue with analysis? ")

        # Check if user wants executive summary only
        if getattr(args, 'executive_only', False):
            print("Generating Executive Summary Only...")
            exec_file = generate_executive_summary(processed_data, modules, args)
            if exec_file:
                print(f"\nExecutive Summary completed: {exec_file}")
            else:
                print("\nERROR: Executive Summary generation failed")

            if not getattr(args, 'no_pause', False):
                interactive_pause("Executive summary complete. Press Enter to finish: ")
            return

        # Run comprehensive analysis
        if logger:
            logger.info("Starting comprehensive analysis workflow")
        analysis_results = run_comprehensive_analysis(processed_data, args, modules)

        if not getattr(args, 'no_pause', False):
            interactive_pause("Analysis complete. Continue with report generation? ")

        # Generate all reports
        if logger:
            logger.info("Starting report generation workflow")
        report_files = generate_all_reports(processed_data, analysis_results, args, modules)

        # Display final summary
        display_final_summary(processed_data, analysis_results, report_files, modules)

        if logger:
            logger.info("Main processing completed successfully")

        if not getattr(args, 'no_pause', False):
            interactive_pause("Review results. Press Enter to finish: ")

    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user")
        safe_input("Press Enter to close...")

    except Exception as e:
        print(f"\nERROR: Unexpected error in main execution: {str(e)}")
        print("Check log file for detailed error information")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        safe_input("Press Enter to close...")


if __name__ == "__main__":
    # Wrap EVERYTHING in a try-catch to prevent any window closing
    try:
        main()
    except SystemExit:
        # Handle sys.exit() calls
        safe_input("Application exited. Press Enter to close...")
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        safe_input("Press Enter to close...")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        safe_input("Press Enter to close...")
    finally:
        # This ALWAYS runs, no matter what
        print("\nShutdown complete.")
        safe_input("Final pause - Press Enter to close...")