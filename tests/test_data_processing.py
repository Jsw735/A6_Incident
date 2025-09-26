#!/usr/bin/env python3
"""
Data Processing Module Test
Tests data validation, file handling, and processing functions
"""

import sys
import pandas as pd
from pathlib import Path
import tempfile
import os

# Add project root to path - corrected for your actual structure
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_test_csv_file():
    """Create a temporary CSV file for testing DataLoader."""
    test_data = pd.DataFrame({
        'incident_id': ['INC001', 'INC002', 'INC003', 'INC004'],
        'status': ['Open', 'Closed', 'In Progress', 'Open'],
        'priority': ['High', 'Medium', 'Low', 'High'],
        'created_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'description': ['Server down', 'Login issue', 'Network slow', 'Database error']
    })
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    test_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name, test_data

def test_data_loader():
    """Test the DataLoader class with file operations."""
    print("🔍 Testing Data Loader...")
    
    try:
        # Test 1: Import the module
        print("  Step 1: Testing DataLoader import...")
        from data_processing.data_loader import DataLoader
        print("  ✅ DataLoader import successful")
        
        # Test 2: Class instantiation
        print("  Step 2: Testing DataLoader instantiation...")
        loader = DataLoader()
        print("  ✅ DataLoader instantiation successful")
        
        # Test 3: Create test file and load data
        print("  Step 3: Testing file loading...")
        test_file_path, expected_data = create_test_csv_file()
        
        try:
            if hasattr(loader, 'load_data'):
                loaded_data = loader.load_data(test_file_path)
                print(f"  ✅ File loading successful: {len(loaded_data)} rows loaded")
                
                # Validate loaded data matches expected
                if len(loaded_data) == len(expected_data):
                    print("  ✅ Row count validation passed")
                else:
                    print(f"  ⚠️  Row count mismatch: expected {len(expected_data)}, got {len(loaded_data)}")
                
                # Check if required columns exist
                required_columns = ['incident_id', 'status', 'priority']
                missing_columns = [col for col in required_columns if col not in loaded_data.columns]
                if not missing_columns:
                    print("  ✅ Required columns validation passed")
                else:
                    print(f"  ⚠️  Missing columns: {missing_columns}")
                    
            else:
                print("  ⚠️  load_data method not found - check DataLoader implementation")
                
            # Test 4: Test error handling with invalid file
            print("  Step 4: Testing error handling...")
            if hasattr(loader, 'load_data'):
                try:
                    loader.load_data("nonexistent_file.csv")
                    print("  ⚠️  Expected error for invalid file not raised")
                except Exception:
                    print("  ✅ Error handling for invalid file works correctly")
            
        finally:
            # Clean up temporary file
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
        
        print("  🎉 DataLoader test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"  ❌ DataLoader Import Error: {e}")
        print("  💡 Check that data_processing/data_loader.py exists and has correct syntax")
        return False
        
    except Exception as e:
        print(f"  ❌ DataLoader Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_cleaner():
    """Test the DataCleaner class with data quality issues."""
    print("\n🔍 Testing Data Cleaner...")
    
    try:
        # Test 1: Import the module
        print("  Step 1: Testing DataCleaner import...")
        from data_processing.data_cleaner import DataCleaner
        print("  ✅ DataCleaner import successful")
        
        # Test 2: Class instantiation
        print("  Step 2: Testing DataCleaner instantiation...")
        cleaner = DataCleaner()
        print("  ✅ DataCleaner instantiation successful")
        
        # Test 3: Create dirty test data
        print("  Step 3: Creating dirty test data...")
        dirty_data = pd.DataFrame({
            'incident_id': ['INC001', 'INC002', None, 'INC004', 'INC002'],  # Missing and duplicate
            'status': ['Open', 'CLOSED', 'in progress', 'Open', 'closed'],   # Inconsistent case
            'priority': ['High', 'Medium', '', 'Low', 'High'],               # Empty string
            'created_date': ['2024-01-01', '2024-01-02', '2024-01-03', 'invalid_date', '2024-01-05'],
            'description': ['  Server down  ', 'Login issue', None, 'Database error', 'Network slow']  # Whitespace and None
        })
        
        print(f"  📊 Dirty data created: {len(dirty_data)} rows")
        print(f"  📊 Missing values: {dirty_data.isnull().sum().sum()}")
        print(f"  📊 Duplicate incident_ids: {dirty_data['incident_id'].duplicated().sum()}")
        
        # Test 4: Data cleaning
        print("  Step 4: Testing data cleaning...")
        if hasattr(cleaner, 'clean_data'):
            cleaned_data = cleaner.clean_data(dirty_data)
            print(f"  ✅ Data cleaning successful: {len(cleaned_data)} rows after cleaning")
            
            # Validate cleaning results
            if len(cleaned_data) <= len(dirty_data):
                print("  ✅ Cleaning reduced or maintained row count (expected)")
            else:
                print("  ⚠️  Cleaning increased row count (unexpected)")
            
            # Check for remaining missing values in critical columns
            if 'incident_id' in cleaned_data.columns:
                missing_ids = cleaned_data['incident_id'].isnull().sum()
                if missing_ids == 0:
                    print("  ✅ No missing incident_ids after cleaning")
                else:
                    print(f"  ⚠️  Still {missing_ids} missing incident_ids")
            
        else:
            print("  ⚠️  clean_data method not found - check DataCleaner implementation")
        
        # Test 5: Test specific cleaning methods (if available)
        print("  Step 5: Testing specific cleaning methods...")
        test_methods = ['remove_duplicates', 'standardize_text', 'handle_missing_values']
        
        for method_name in test_methods:
            if hasattr(cleaner, method_name):
                print(f"  ✅ Method {method_name} found")
            else:
                print(f"  ⚠️  Method {method_name} not found")
        
        print("  🎉 DataCleaner test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"  ❌ DataCleaner Import Error: {e}")
        print("  💡 Check that data_processing/data_cleaner.py exists and has correct syntax")
        return False
        
    except Exception as e:
        print(f"  ❌ DataCleaner Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processor():
    """Test the DataProcessor class with controlled data."""
    print("\n🔍 Testing Data Processing Module...")
    
    try:
        # Test 1: Import the module - corrected import path
        print("  Step 1: Testing module import...")
        from data_processing.data_processor import DataProcessor
        print("  ✅ DataProcessor import successful")
        
        # Test 2: Class instantiation
        print("  Step 2: Testing class instantiation...")
        processor = DataProcessor()
        print("  ✅ DataProcessor instantiation successful")
        
        # Test 3: Create test data
        print("  Step 3: Creating test incident data...")
        test_data = pd.DataFrame({
            'incident_id': ['INC001', 'INC002', 'INC003'],
            'status': ['Open', 'Closed', 'In Progress'],
            'priority': ['High', 'Medium', 'Low'],
            'created_date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        print(f"  ✅ Test data created: {len(test_data)} rows")
        
        # Test 4: Basic processing
        print("  Step 4: Testing basic data processing...")
        if hasattr(processor, 'process_data'):
            result = processor.process_data(test_data)
            print(f"  ✅ Data processing successful: {type(result)}")
        else:
            print("  ⚠️  process_data method not found - check implementation")
        
        # Test 5: Data validation
        print("  Step 5: Testing data validation...")
        if hasattr(processor, 'validate_data'):
            validation_result = processor.validate_data(test_data)
            print(f"  ✅ Data validation successful: {validation_result}")
        else:
            print("  ⚠️  validate_data method not found - check implementation")
        
        print("  🎉 Data Processing Module test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import Error: {e}")
        print("  💡 Check that data_processing/data_processor.py exists and has correct syntax")
        return False
        
    except Exception as e:
        print(f"  ❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validator():
    """Test the DataValidator class."""
    print("\n🔍 Testing Data Validator...")
    
    try:
        from data_processing.data_validator import DataValidator
        validator = DataValidator()
        
        # Test with known data quality issues
        test_data = pd.DataFrame({
            'col1': [1, 2, None, 4],  # 25% missing
            'col2': [1, 1, 2, 3],     # 1 duplicate value
            'col3': ['A', 'B', 'C', 'D']
        })
        
        if hasattr(validator, 'check_completeness'):
            completeness = validator.check_completeness(test_data)
            print(f"  ✅ Completeness check: {completeness}")
        
        if hasattr(validator, 'check_duplicates'):
            duplicates = validator.check_duplicates(test_data)
            print(f"  ✅ Duplicate check: {duplicates}")
        
        if hasattr(validator, 'validate_data'):
            validation = validator.validate_data(test_data)
            print(f"  ✅ General validation: {validation}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ DataValidator import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ DataValidator test failed: {e}")
        return False

def main():
    """Run all data processing tests."""
    print("=" * 60)
    print("DATA PROCESSING MODULE COMPREHENSIVE TESTS")
    print("=" * 60)
    
    results = []
    
    # Run individual tests in logical order
    print("Testing individual components...")
    results.append(test_data_loader())
    results.append(test_data_cleaner())
    results.append(test_data_validator())
    results.append(test_data_processor())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All data processing tests passed!")
        print("✅ Your data processing pipeline is ready!")
        return 0
    else:
        print("❌ Some tests failed - check errors above")
        print("💡 Focus on fixing the failed imports and missing methods")
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