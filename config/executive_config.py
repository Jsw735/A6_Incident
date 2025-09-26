[EXECUTIVE_DASHBOARD]
# Executive Summary Configuration
# Health score weights (must sum to 1.0)
health_score_weights_backlog = 0.4
health_score_weights_closure = 0.3
health_score_weights_accuracy = 0.2
health_score_weights_training = 0.1

# Target Metrics for October Backlog Clearance
target_max_backlog = 300
target_closure_rate = 85.0
min_accuracy_threshold = 80.0
optimal_training_min = 10.0
optimal_training_max = 15.0

# Health Status Thresholds
health_excellent_threshold = 85.0
health_good_threshold = 75.0
health_fair_threshold = 65.0

[TREND_ANALYSIS]
# Trend Analysis Configuration
comparison_period_weeks = 2
lookback_period_weeks = 4
daily_average_calculation = true
trend_smoothing_enabled = true
minimum_data_points = 7

[VISUALIZATION]
# Dashboard Visualization Settings
kpi_tile_layout = 5x2
color_scheme = professional
print_ready = true
powerpoint_ready = true
chart_types = line,pie,stacked_bar
export_format = xlsx

[SLA_THRESHOLDS]
# Priority Level SLA Hours (based on requirements table)
P0_hours = 4
P1_hours = 24
P2_hours = 72
P3_hours = 168

# SLA Descriptions
P0_description = Critical Incident - Immediate service disruption (4 hrs)
P1_description = High Incident - Manual workaround available (1 business day)
P2_description = Medium Incident - Normal resolution SLAs (3 business days)
P3_description = Low Incident - Lowest resolution SLAs acceptable

[URGENCY_MAPPING]
# Urgency Display Mapping
urgency_1 = High
urgency_2 = Medium
urgency_3 = Medium
urgency_4 = Low

[EXPORT_SETTINGS]
# Export Configuration
default_output_directory = output/
excel_template_enabled = true
auto_timestamp = true
include_metadata = true
filename_prefix = executive_summary

[LOGGING]
# Logging Configuration
log_level = INFO
log_file = logs/executive_summary.log
max_log_size_mb = 10
backup_count = 5
console_logging = true

[PERFORMANCE]
# Performance Optimization Settings
cache_enabled = true
cache_timeout_minutes = 30
parallel_processing = false
max_workers = 4

[KEYWORDS]
# Keyword Enhancement Settings
enhance_from_recent_data = true
recent_data_weeks = 4
minimum_keyword_frequency = 3
max_enhanced_keywords = 10