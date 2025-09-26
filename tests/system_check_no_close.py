#!/usr/bin/env python3
"""
Test script that will NOT close automatically.
Implements proper Python script execution patterns.
"""

import sys
import os
import time

def display_system_info():
    """Display comprehensive system information."""
    print("=" * 60)
    print("PYTHON SCRIPT EXECUTION TEST")
    print("=" * 60)
    print(f"✅ Python Version: {sys.version}")
    print(f"✅ Current Directory: {os.getcwd()}")
    print(f"✅ Script Location: {__file__}")
    print(f"✅ Platform: {sys.platform}")
    print("=" * 60)

def test_basic_functionality():
    """Test basic Python operations."""
    print("\n🔧 Testing Basic Functionality...")
    
    # Test 1: Math operations
    result = 2 + 2
    print(f"✅ Math test: 2 + 2 = {result}")
    
    # Test 2: String operations
    message = "Hello from Python!"
    print(f"✅ String test: {message}")
    
    # Test 3: List operations
    test_list = [1, 2, 3, 4, 5]
    print(f"✅ List test: {test_list}")
    
    # Test 4: File system check
    files = os.listdir('.')
    print(f"✅ Directory contents: {len(files)} items found")
    
    return True

def interactive_menu():
    """Interactive menu that keeps script running."""
    while True:
        print("\n" + "=" * 40)
        print("INTERACTIVE TEST MENU")
        print("=" * 40)
        print("1. Show System Information")
        print("2. Test Basic Functionality")
        print("3. List Current Directory")
        print("4. Test Python Imports")
        print("5. Exit")
        print("=" * 40)
        
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                display_system_info()
                
            elif choice == '2':
                test_basic_functionality()
                
            elif choice == '3':
                print("\n📁 Current Directory Contents:")
                for item in os.listdir('.'):
                    print(f"   {item}")
                    
            elif choice == '4':
                print("\n📦 Testing Python Imports...")
                test_imports()
                
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-5.")
            
            # Pause after each operation
            input("\nPress ENTER to continue...")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Press ENTER to continue...")

def test_imports():
    """Test common Python imports."""
    imports_to_test = [
        ('sys', 'System functions'),
        ('os', 'Operating system interface'),
        ('time', 'Time functions'),
        ('pathlib', 'Path handling'),
        ('pandas', 'Data analysis (if installed)'),
        ('logging', 'Logging functionality')
    ]
    
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: {description} - Available")
        except ImportError:
            print(f"❌ {module_name}: {description} - Not available")

def main():
    """
    Main function implementing proper Python script execution.
    This function ensures the script stays open and provides feedback.
    """
    print("🚀 Starting Python Test Script...")
    print("This script will NOT close automatically!")
    
    try:
        # Initial system check
        display_system_info()
        
        # Show that script is running
        print("\n✅ Script is running successfully!")
        print("The window will stay open until you choose to exit.")
        
        # Start interactive menu
        interactive_menu()
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        print("Script encountered an error but will NOT close automatically.")
        
    finally:
        print("\n" + "=" * 60)
        print("SCRIPT EXECUTION COMPLETED")
        print("=" * 60)
        
        # Final pause to prevent closing
        input("Press ENTER to close this window...")
    
    return 0

# Proper Python script execution pattern
if __name__ == '__main__':
    """
    Script entry point - this is the standard Python convention.
    This ensures the script runs when executed directly.
    """
    try:
        print("Initializing script...")
        exit_code = main()
        print(f"Script completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user (Ctrl+C)")
        input("Press ENTER to close...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("Even with an error, the script will NOT disappear!")
        input("Press ENTER to close...")
        sys.exit(1)