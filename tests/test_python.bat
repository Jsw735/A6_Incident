@echo off
echo Testing Python with project structure...
echo.

echo Current directory: %CD%
echo.

if exist test_basic.py (
    echo Running basic test...
    python test_basic.py
) else (
    echo test_basic.py not found, creating it...
    echo print("Basic test created and running") > test_basic.py
    echo input("Press Enter...") >> test_basic.py
    python test_basic.py
)

echo.
echo Now testing the main system test...
if exist modules_to_test.py (
    python modules_to_test.py
) else (
    echo modules_to_test.py not found!
    echo Please check file location.
)

pause