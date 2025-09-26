#!/usr/bin/env python3
"""
SAP Incident Analyzer Test Runner - Bulletproof Version
This script implements proper Python script patterns to prevent disappearing.
"""

import sys
import os
import traceback
import time
from pathlib import Path


def safe_input(prompt="Press ENTER to continue..."):
    """Safe input function that handles all edge cases."""
    try:
        print(f"\n{prompt}")
        return input()
    except (KeyboardInterrupt, EOFError):
        print("\n\nUser interrupted. Exiting safely...")
        return None
    except Exception as e:
        print(f"Input error: {e}")
        return ""


def display_header():
    """Display application header - always visible."""
    print("\n" + "="*60)
    print("SAP INCIDENT ANALYZER - TEST RUNNER")
    print("="*60)
    print("Status: RUNNING (Script is active)")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("="*60)


def test_basic_functionality():
    """Test basic Python functionality to ensure environment works."""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        # Test 1: Basic Python operations
        result = 2 + 2
        assert result == 4
        print("✅ Basic math operations: WORKING")
        
        # Test 2: File system access
        current_dir = Path.cwd()
        print(f"✅ File system access: WORKING ({current_dir})")
        
        # Test 3: Import capabilities
        import pandas as pd
        print("✅ Pandas import: WORKING")
        
        # Test 4: Analysis module check
        analysis_dir = Path("analysis")
        if analysis_dir.exists():
            print("✅ Analysis directory: FOUND")
            
            # Check for required files
            required_files = ["__init__.py", "quality_analyzer.py", "trend_analyzer.py"]
            for file in required_files:
                file_path = analysis_dir / file
                if file_path.exists():
                    print(f"✅ {file}: FOUND")
                else:
                    print(f"❌ {file}: MISSING")
        else:
            print("❌ Analysis directory: NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        print("Full error details:")
        traceback.print_exc()
        return False


def test_analyzer_imports():
    """Test analyzer imports with detailed error reporting."""
    print("\n📦 Testing Analyzer Imports...")
    
    results = {}
    
    # Test Quality Analyzer
    try:
        from analysis.quality_analyzer import QualityAnalyzer
        print("✅ QualityAnalyzer: IMPORTED SUCCESSFULLY")
        results['quality'] = True
    except Exception as e:
        print(f"❌ QualityAnalyzer: IMPORT FAILED - {e}")
        results['quality'] = False
    
    # Test Trend Analyzer
    try:
        from analysis.trend_analyzer import TrendAnalyzer
        print("✅ TrendAnalyzer: IMPORTED SUCCESSFULLY")
        results['trend'] = True
    except Exception as e:
        print(f"❌ TrendAnalyzer: IMPORT FAILED - {e}")
        results['trend'] = False
    
    return results


def run_safe_analyzer_test():
    """Run analyzer tests with maximum safety."""
    print("\n🧪 Running Safe Analyzer Tests...")
    
    try:
        import pandas as pd
        
        # Create simple test data
        test_data = pd.DataFrame({
            'Number': ['INC001', 'INC002', 'INC003'],
            'Priority': ['High', 'Medium', 'Low'],
            'State': ['Open', 'Closed', 'Open']
        })
        
        print(f"✅ Test data created: {len(test_data)} records")
        
        # Test Quality Analyzer if available
        try:
            from analysis.quality_analyzer import QualityAnalyzer
            qa = QualityAnalyzer()
            results = qa.analyze_data_quality(test_data)
            print(f"✅ Quality Analysis: {results.total_records} records analyzed")
        except Exception as e:
            print(f"⚠️  Quality Analysis skipped: {e}")
        
        # Test Trend Analyzer if available
        try:
            from analysis.trend_analyzer import TrendAnalyzer
            ta = TrendAnalyzer()
            results = ta.analyze_trends(test_data)
            print(f"✅ Trend Analysis: {results.total_incidents} incidents analyzed")
        except Exception as e:
            print(f"⚠️  Trend Analysis skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        traceback.print_exc()
        return False


def show_menu():
    """Display menu options."""
    menu = """
┌─────────────────────────────────────────┐
│              TEST OPTIONS               │
├─────────────────────────────────────────┤
│  1. Test Basic Functionality           │
│  2. Test Analyzer Imports              │
│  3. Run Safe Analyzer Tests            │
│  4. Show System Information            │
│  5. Exit (Safe)                        │
└─────────────────────────────────────────┘
"""
    print(menu)


def show_system_info():
    """Display comprehensive system information."""
    print("\n💻 System Information:")
    print("="*40)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Script Path: {__file__}")
    print(f"Python Path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Check for required packages
    packages = ['pandas', 'logging', 'pathlib']
    print("\nPackage Status:")
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: Missing")


def main():
    """
    Main function implementing proper Python script patterns.
    This function will NOT disappear and always provides feedback.
    """
    print("🚀 Starting Test Runner...")
    print("This script will NOT disappear - you'll always see output!")
    
    try:
        display_header()
        
        # Main execution loop
        while True:
            show_menu()
            
            choice = safe_input("Select option (1-5): ")
            
            # Handle user interruption
            if choice is None:
                print("User interrupted. Exiting safely...")
                break
            
            choice = choice.strip()
            
            if choice == '1':
                print("\n" + "="*50)
                test_basic_functionality()
                safe_input()
                
            elif choice == '2':
                print("\n" + "="*50)
                test_analyzer_imports()
                safe_input()
                
            elif choice == '3':
                print("\n" + "="*50)
                run_safe_analyzer_test()
                safe_input()
                
            elif choice == '4':
                show_system_info()
                safe_input()
                
            elif choice == '5':
                print("\n👋 Exiting safely...")
                print("Script completed successfully!")
                break
                
            else:
                print(f"❌ Invalid choice: '{choice}'. Please select 1-5.")
                time.sleep(1)
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Full error traceback:")
        traceback.print_exc()
        print("\nScript encountered an error but did NOT disappear!")
        safe_input("Press ENTER to exit...")
    
    finally:
        print("\n" + "="*60)
        print("SCRIPT EXECUTION COMPLETED")
        print("The script has finished running and will now exit.")
        print("="*60)
    
    return 0


# Proper Python script execution pattern
if __name__ == '__main__':
    """
    Script entry point - implements proper Python conventions.
    This ensures the script runs correctly when executed directly.
    """
    try:
        print("Script starting...")
        exit_code = main()
        print(f"Script finished with exit code: {exit_code}")
        
        # Final pause to prevent window from closing
        safe_input("Press ENTER to close this window...")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user (Ctrl+C)")
        print("Script did NOT crash - this was a controlled exit.")
        safe_input("Press ENTER to close...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("Full error details:")
        traceback.print_exc()
        print("\nEven with a fatal error, the script did NOT disappear!")
        safe_input("Press ENTER to close...")
        sys.exit(1)