#!/usr/bin/env python3
"""
Persistent test script that will stay open.
Demonstrates proper Python script execution patterns.
"""

import sys
import os

def main():
    """Main function that keeps the script running."""
    print("=" * 50)
    print("PYTHON SCRIPT EXECUTION TEST")
    print("=" * 50)
    print("✅ Script is running successfully!")
    print(f"✅ Python Version: {sys.version}")
    print(f"✅ Current Directory: {os.getcwd()}")
    print("=" * 50)
    
    # Test basic functionality
    print("\n🔧 Testing Basic Operations...")
    
    # Math test
    result = 10 + 5
    print(f"Math test: 10 + 5 = {result}")
    
    # String test
    message = "Hello from Python script!"
    print(f"String test: {message}")
    
    # List directory contents
    print(f"\nDirectory contents:")
    try:
        files = os.listdir('.')
        for i, file in enumerate(files[:5], 1):  # Show first 5 files
            print(f"  {i}. {file}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    print("\n" + "=" * 50)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("The script will now wait for your input...")
    print("=" * 50)
    
    # This is the KEY - force the script to wait
    input("Press ENTER to close this window...")

# This is the proper Python script execution pattern
if __name__ == '__main__':
    """
    Script entry point - ensures script runs when executed directly.
    This is the standard Python convention for executable scripts.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        input("Press ENTER to close...")
    except Exception as e:
        print(f"Error occurred: {e}")
        input("Press ENTER to close...")