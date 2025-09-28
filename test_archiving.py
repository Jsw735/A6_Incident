#!/usr/bin/env python3
"""
Test script to fix the archiving issue
"""
import sys
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(str(Path.cwd()))

try:
    from reporting.executive_summary import EnhancedExecutiveSummary
    
    print("Creating archiver instance...")
    archiver = EnhancedExecutiveSummary()
    
    print("Running KPI report archiving...")
    archiver.archive_kpi_reports_only()
    
    print("Archiving test completed!")
    
except Exception as e:
    print(f"Error during archiving test: {e}")
    import traceback
    traceback.print_exc()