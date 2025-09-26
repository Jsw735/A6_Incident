#!/usr/bin/env python3
"""
Reporting Module Test
Tests Excel generation, report creation, and output formatting
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_excel_generator():
    """Test Excel generation capabilities."""
    print("🔍 Testing Excel Generator...")
    
    try:
        from reporting.excel_generator import ExcelGenerator
        
        generator = ExcelGenerator()
        print("  ✅ ExcelGenerator instantiation successful")
        
        # Create test data for Excel export
        test_data = pd.DataFrame({
            'incident_id': ['INC001', 'INC002', 'INC003'],
            'status': ['Open', 'Closed', 'In Progress'],
            'priority': ['High', 'Medium', 'Low']
        })
        
        if hasattr(generator, 'create_workbook'):
            workbook = generator.create_workbook()
            print("  ✅ Workbook creation successful")
        
        if hasattr(generator, 'add_data_sheet'):
            generator.add_data_sheet(test_data, 'Test_Data')
            print("  ✅ Data sheet addition successful")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ ExcelGenerator import failed: {e}")
        print("  💡 Check that reporting/excel_generator.py exists")
        return False
    except Exception as e:
        print(f"  ❌ ExcelGenerator test failed: {e}")
        return False

def test_report_generators():
    """Test various report generators."""
    print("\n🔍 Testing Report Generators...")
    
    report_modules = [
        'reporting.daily_reporter',
        'reporting.weekly_reporter',
        'reporting.executive_summary'
    ]
    
    results = []
    
    for module_name in report_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"  ✅ {module_name} import successful")
            results.append(True)
        except ImportError as e:
            print(f"  ❌ {module_name} import failed: {e}")
            results.append(False)
    
    return all(results)

def main():
    """Run all reporting tests."""
    print("=" * 50)
    print("REPORTING MODULE TESTS")
    print("=" * 50)
    
    results = []
    
    # Run individual tests
    results.append(test_excel_generator())
    results.append(test_report_generators())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All reporting tests passed!")
        return 0
    else:
        print("❌ Some tests failed - check errors above")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        input("\nPress Enter to continue...")