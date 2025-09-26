#!/usr/bin/env python3
"""
Automatic Import Fixer for SAP Incident Reporting System
Fixes missing Tuple imports following PEP 8 and clean code principles.
Implements proper path resolution as per Python best practices.
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """
    Get the project root directory reliably.
    Implements proper path resolution following clean code principles.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Change working directory to script location
    os.chdir(script_dir)
    
    print(f"[OK] Working directory set to: {script_dir}")
    return script_dir

def verify_project_structure():
    """
    Verify we're in the correct project directory.
    Implements defensive programming with comprehensive validation.
    """
    required_dirs = ['analysis', 'data_processing', 'reporting', 'utils']
    current_dir = Path('.')
    
    print("[SEARCH] Verifying project structure...")
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"   [OK] Found: {dir_name}/")
        else:
            missing_dirs.append(dir_name)
            print(f"   [ERROR] Missing: {dir_name}/")
    
    if missing_dirs:
        print(f"\n[WARN] Warning: Missing directories {missing_dirs}")
        print("This script should be run from the project root directory.")
        return False
    
    print("[OK] Project structure verified")
    return True

def fix_tuple_import(file_path):
    """
    Fix missing Tuple import in a Python file.
    Implements defensive programming with comprehensive error handling.
    """
    try:
        # Convert to absolute path for better error reporting
        abs_path = file_path.absolute()
        
        # Check if file exists
        if not file_path.exists():
            return False, f"File not found: {abs_path}"
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file uses Tuple but doesn't import it properly
        if 'Tuple' not in content:
            return False, "No Tuple usage found"
        
        lines = content.split('\n')
        fixed = False
        
        # Look for existing typing import line
        for i, line in enumerate(lines):
            if line.strip().startswith('from typing import'):
                # Check if Tuple is already in the import
                if 'Tuple' not in line:
                    # Add Tuple to the existing import
                    # Handle various import formats cleanly
                    if line.endswith(','):
                        lines[i] = line + ' Tuple'
                    else:
                        lines[i] = line + ', Tuple'
                    fixed = True
                    break
        
        # If no typing import found, add one after other imports
        if not fixed:
            # Find the best place to insert the import (after other imports)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_index = i + 1
                elif line.strip() == '' and insert_index > 0:
                    break
            
            # Insert the typing import
            typing_import = 'from typing import Dict, List, Set, Any, Optional, Union, Tuple'
            lines.insert(insert_index, typing_import)
            fixed = True
        
        if fixed:
            # Write the fixed content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True, "Fixed successfully"
        else:
            return False, "Tuple already imported"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """
    Main function with proper path setup and comprehensive error handling.
    Follows clean code principles with systematic execution.
    """
    print("SAP Incident Reporting System - Import Fixer")
    print("=" * 60)
    
    try:
        # Step 1: Set up proper working directory
        project_root = get_project_root()
        
        # Step 2: Verify we're in the right place
        if not verify_project_structure():
            print("\n[ERROR] Please run this script from the project root directory:")
            print(f"   cd C:\\Users\\jsw73\\Downloads\\incident_reporting_system")
            print(f"   python fix_imports.py")
            input("Press Enter to close...")
            return
        
        print("\n[FIX] Fixing missing Tuple imports...")
        
        # Files identified from findstr output - using relative paths
        files_to_fix = [
            'analysis/metrics_calculator.py',
            'analysis/quality_analyzer.py',
            'analysis/trend_analyzer.py',
            'data_processing/data_loader.py',
            'data_processing/data_processor.py',
            'data_processing/data_validator.py',
            'reporting/quality_reporter.py',
            'reporting/weekly_reporter.py',
            'utils/date_utils.py',
            'utils/text_analyzer.py'
        ]
        
        fixed_count = 0
        error_count = 0
        not_found_count = 0
        
        for file_path_str in files_to_fix:
            # Use Path for proper cross-platform path handling
            file_path = Path(file_path_str)
            
            print(f"\n[SEARCH] Processing: {file_path}")
            
            if not file_path.exists():
                print(f"   [WARN] File not found: {file_path.absolute()}")
                not_found_count += 1
                continue
            
            success, message = fix_tuple_import(file_path)
            
            if success:
                print(f"   [OK] Fixed: {file_path}")
                fixed_count += 1
            else:
                if "already imported" in message or "No Tuple usage found" in message:
                    print(f"   [INFO] Already OK: {file_path}")
                else:
                    print(f"   [ERROR] Error: {message}")
                    error_count += 1
        
        # Summary report
        print("\n" + "=" * 60)
        print("IMPORT FIXING SUMMARY")
        print("=" * 60)
        print(f"[OK] Files fixed: {fixed_count}")
        print(f"[WARN] Files not found: {not_found_count}")
        if error_count > 0:
            print(f"[ERROR] Files with errors: {error_count}")
        else:
            print("[DONE] All existing files processed successfully!")
        
        if not_found_count > 0:
            print(f"\n[INFO] Note: {not_found_count} files were not found.")
            print("This is normal if those modules haven't been created yet.")
        
        print(f"\n[OK] Import fixing completed!")
        print("You can now run: python main.py")
        
    except Exception as e:
        print(f"\n[ERROR] Critical error: {str(e)}")
        print("Make sure you're running this from the correct directory.")
    
    finally:
        input("\nPress Enter to close...")

if __name__ == "__main__":
    main()