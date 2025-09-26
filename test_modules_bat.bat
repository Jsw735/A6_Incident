@echo off
echo SAP Incident Reporting System - Main Test Suite
echo Current directory: %CD%
echo.

echo Running comprehensive system tests...
python modules_to_test.py

echo.
echo Test execution completed.
pause