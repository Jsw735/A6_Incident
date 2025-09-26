@echo off
echo SAP Incident Reporting System - Main Test Suite
echo Current directory: %CD%
echo.

echo Running comprehensive system tests...
python executive_summary.py

echo.
echo incidents_report completed.
pause