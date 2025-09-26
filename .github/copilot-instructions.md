## Incident Reporting System — Copilot Instructions

This repository implements a modular SAP S/4HANA incident reporting system (data loading → cleaning → validation → analysis → reporting).
Keep changes small and focused. Below are the key patterns, workflow commands, and file pointers to help you be productive.

1) Big-picture architecture
   - Data pipeline: `data_processing/` orchestrates loading (`data_loader.py` → `DataLoader`), cleaning (`data_cleaner.py` → `DataCleaner`), validation (`data_validator.py`), and processing (`data_processor.py` → `DataProcessor`).
   - Analysis: `analysis/` contains domain analyzers used by `main.py`: `quality_analyzer.py` (`QualityAnalyzer`), `trend_analyzer.py` (`TrendAnalyzer`), `metrics_calculator.py` (`MetricsCalculator`), `keyword_manager.py` (`KeywordManager`).
   - Reporting: `reporting/` (imported by `main.py`) produces Excel and executive summaries — prefer using `ExcelGenerator` / `EnhancedExecutiveSummary` classes.
   - Config: `config/settings.py` (`Settings`) centralizes INI-driven configuration (`executive_config.ini`) used across modules.

2) Naming & import conventions
   - Many modules provide class aliases for backward compatibility (e.g., `DataLoader = IncidentDataLoader`, `MetricsCalculator = IncidentMetricsCalculator`). Use the top-level class names seen in `main.py` when adding imports.
   - Config exposes a singleton `Settings` class — instantiate with `Settings()` (it will create defaults if `executive_config.ini` is missing).
   - Column mapping: cleaners and analyzers standardize columns (e.g., `created_date`, `resolved_date`, `number`, `priority`, `assignment_group`, `short_description`). Prefer these canonical names in new code.

3) Tests & quick checks
   - Unit tests live under `analysis/test_quality_metrics.py` (example test harness). Run tests with your Python test runner (no framework file found; run specific files directly).
   - Quick interactive run: open a PowerShell terminal and run the repo's `main.py`. Ensure dependencies are installed (pandas, numpy, openpyxl, matplotlib, seaborn). Example:

```powershell
python .\main.py
```

If imports fail, inspect `main.py`'s safe import block for the expected class names and required packages.

4) Common patterns and pitfalls
   - Defensive loading: `DataLoader.load_csv()` returns (success, df, message). Always check the success flag before using the DataFrame.
   - Validators return dicts with keys like `is_valid`, `warnings`, `errors`. Follow this structure when writing new validators or handling results.
   - Column mapping happens early in `DataCleaner`; add new canonical column names to `column_mapping` in `data_cleaner.py` if you introduce new fields.
   - Date handling: analyzers expect datetime columns named `created_date` / `resolved_date` (trend/metrics code uses `Created` and mapping layers). Use `pd.to_datetime(..., errors='coerce')` and filter out NaT.

5) Code style & behavior
   - Follow existing style: tidy, defensive functions, explicit logging (`logging.getLogger(__name__)`) and informative log messages (see `main.py` logging setup).
   - Prefer small, testable helper methods (see `IncidentTrendAnalyzer._prepare_data_pipeline`) and method chaining for pandas transformations.

6) Examples to copy/use
   - Load → validate → process flow (from `main.py`): call `DataLoader().load_csv(path)` → `DataValidator().validate_data(df)` → `DataProcessor().process_data(df)`.
   - Trend analysis example: use `TrendAnalyzer().analyze_trends(df)` which returns a `TrendMetrics` dataclass with `.to_dict()` for reporting.

7) Integration & outputs
   - Outputs: `output/` directory and Excel exports configured via `Settings().get_export_settings()`; log files go to `logs/` and use timestamped filenames (see `main.py`).
   - Seed files: `data/keyword_seeds.json` is read by `KeywordManager` if provided.

8) When to ask the maintainer
   - Missing expected column mappings for a new data source.
   - Changes that alter export formats or executive dashboard shape.
   - Adding external dependencies beyond typical data stack (pandas, numpy, openpyxl, matplotlib).

If anything in this file seems unclear or incomplete, tell me which area (pipeline, config, reporting, or tests) and I will expand or adapt the instructions.
