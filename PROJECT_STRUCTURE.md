# SAP Incident Reporting System - Project Structure

Generated: 2025-09-27 14:01:21

## Core System Structure

```
incident_reporting_system/
├── main.py                     # Main application entry point
├── README.md                   # Project documentation
├── PROJECT_STRUCTURE.md        # This file
│
├── analysis/                   # Data analysis modules
│   ├── __init__.py
│   ├── keyword_manager.py
│   ├── metrics_calculator.py
│   ├── quality_analyzer.py
│   ├── trend_analyzer.py
│   └── test_quality_metrics.py
│
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── business_rules.py
│   ├── executive_config.ini
│   ├── executive_config.py
│   └── settings.py
│
├── data/                       # Data files
│   ├── incidents.csv           # Main incident data
│   ├── keyword_seeds.json      # Keyword configuration
│   └── sample_data.csv
│
├── data_processing/            # Data processing modules
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── data_processor.py
│   └── data_validator.py
│
├── reporting/                  # Report generation
│   ├── __init__.py
│   ├── daily_reporter.py
│   ├── excel_generator.py
│   ├── executive_summary.py
│   └── weekly_reporter.py
│
├── tests/                      # Unit tests
├── utils/                      # Utility modules
│
├── output/                     # Current output files
│   └── (recent Excel reports)
│
├── reports/                    # Generated reports
├── logs/                       # Current log files
│
├── archive/                    # Archived files
│   ├── outputs/               # Old Excel files
│   ├── logs/                  # Old log files
│   └── temp_files/            # Archived temp files
│
├── development/               # Development tools
│   ├── analysis_scripts/      # Analysis tools
│   ├── test_scripts/          # Testing tools
│   └── batch_files/           # Automation scripts
│
├── documentation/             # Project documentation
├── backup/                    # Important file backups
│
└── .venv/                     # Python virtual environment
```

## Cleanup Summary

This structure was created by the cleanup script which:
- Archived old output files (keeping last 7 days)
- Organized development scripts into appropriate folders
- Cleaned temporary and lock files
- Created backups of important configuration files
- Added documentation to explain folder purposes

## Usage

To run the main system:
```bash
python main.py
```

To run cleanup again:
```bash
python cleanup_project.py --help
```
