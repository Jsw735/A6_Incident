@echo off
setlocal enabledelayedexpansion

:: Set console colors and formatting
color 0A
title SAP Incident Reporting System - Comprehensive Test Suite

echo ===============================================
echo SAP INCIDENT REPORTING SYSTEM - MODULE TESTS
echo ===============================================
echo.
echo Testing modules individually to isolate issues
echo Following clean code testing principles
echo Press Ctrl+C at any time to stop testing
echo.

:: Initialize test tracking variables
set TOTAL_TESTS=4
set PASSED_TESTS=0
set FAILED_TESTS=0
set START_TIME=%TIME%

echo Starting comprehensive test suite at %START_TIME%
echo ===============================================
echo.

:: Test 1: Data Processing Module
echo [1/%TOTAL_TESTS%] Testing Data Processing Module...
echo ===============================================
echo Running: python test_data_processing.py
python test_data_processing.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Data Processing test failed with exit code %ERRORLEVEL%
    set /a FAILED_TESTS+=1
    echo.
    echo Debugging Information:
    echo - Check data_processing/__init__.py imports
    echo - Verify all module dependencies are installed
    echo - Review data_cleaner.py for missing methods
    echo.
    pause
    goto :test_summary
) else (
    echo ✅ Data Processing test completed successfully!
    set /a PASSED_TESTS+=1
)
echo.
pause

:: Test 2: Analysis Module
echo [2/%TOTAL_TESTS%] Testing Analysis Module...
echo ===============================================
echo Running: python test_analysis.py
python test_analysis.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Analysis test failed with exit code %ERRORLEVEL%
    set /a FAILED_TESTS+=1
    echo.
    echo Debugging Information:
    echo - Check analysis/__init__.py imports
    echo - Verify quality_analyzer.py implementation
    echo - Review trend_analyzer.py and metrics_calculator.py
    echo - Ensure keyword_manager.py is properly configured
    echo.
    pause
    goto :test_summary
) else (
    echo ✅ Analysis test completed successfully!
    set /a PASSED_TESTS+=1
)
echo.
pause

:: Test 3: Reporting Module
echo [3/%TOTAL_TESTS%] Testing Reporting Module...
echo ===============================================
echo Running: python test_reporting.py
python test_reporting.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Reporting test failed with exit code %ERRORLEVEL%
    set /a FAILED_TESTS+=1
    echo.
    echo Debugging Information:
    echo - Check reporting module structure
    echo - Verify report generation functionality
    echo - Review template and output configurations
    echo.
    pause
    goto :test_summary
) else (
    echo ✅ Reporting test completed successfully!
    set /a PASSED_TESTS+=1
)
echo.
pause

:: Test 4: Integration Testing
echo [4/%TOTAL_TESTS%] Testing Integration...
echo ===============================================
echo Running: python test_integration.py
python test_integration.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Integration test failed with exit code %ERRORLEVEL%
    set /a FAILED_TESTS+=1
    echo.
    echo Debugging Information:
    echo - Check end-to-end workflow integration
    echo - Verify module communication interfaces
    echo - Review data flow between components
    echo.
    pause
    goto :test_summary
) else (
    echo ✅ Integration test completed successfully!
    set /a PASSED_TESTS+=1
)
echo.

:test_summary
set END_TIME=%TIME%
echo ===============================================
echo TEST EXECUTION SUMMARY
echo ===============================================
echo Start Time: %START_TIME%
echo End Time:   %END_TIME%
echo.
echo Total Tests:  %TOTAL_TESTS%
echo Passed:       %PASSED_TESTS%
echo Failed:       %FAILED_TESTS%
echo.

if %FAILED_TESTS% EQU 0 (
    echo 🎉 ALL TESTS PASSED! System is ready for deployment.
    echo ===============================================
) else (
    echo ⚠️  %FAILED_TESTS% test(s) failed. Review errors above.
    echo ===============================================
    echo Recommended next steps:
    echo 1. Fix the failing module(s)
    echo 2. Run individual tests to isolate issues
    echo 3. Check dependencies and imports
    echo 4. Review error logs for specific issues
)

echo.
echo Press any key to exit...
pause > nul
exit /b %FAILED_TESTS%