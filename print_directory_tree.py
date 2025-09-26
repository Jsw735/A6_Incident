#!/usr/bin/env python3
"""
Directory Tree Explorer Script
Recursively explores directories starting from script location
Following PEP 8 guidelines and clean code principles
"""

import os
import sys
from pathlib import Path
from typing import Optional

def print_directory_tree(root_path: str, max_depth: Optional[int] = None) -> None:
    """
    Print a tree structure of directories and files.
    
    Args:
        root_path: The root directory to start exploration
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"❌ Error: Directory '{root_path}' does not exist")
        return
    
    if not root.is_dir():
        print(f"❌ Error: '{root_path}' is not a directory")
        return
    
    print(f"📁 Directory Structure for: {root.absolute()}")
    print("=" * 60)
    
    _explore_directory(root, "", max_depth, 0)

def _explore_directory(directory: Path, prefix: str, max_depth: Optional[int], current_depth: int) -> None:
    """
    Recursively explore directory structure.
    
    Args:
        directory: Current directory to explore
        prefix: String prefix for tree formatting
        max_depth: Maximum depth to traverse
        current_depth: Current traversal depth
    """
    try:
        # Check depth limit
        if max_depth is not None and current_depth >= max_depth:
            return
        
        # Get all items in directory
        items = list(directory.iterdir())
        
        # Separate directories and files
        directories = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # Sort for consistent output
        directories.sort(key=lambda x: x.name.lower())
        files.sort(key=lambda x: x.name.lower())
        
        # Print current directory name
        if current_depth > 0:
            print(f"{prefix}📁 {directory.name}/")
        
        # Calculate new prefix for children
        child_prefix = prefix + "│   " if current_depth > 0 else ""
        
        # Print subdirectories first
        for i, subdir in enumerate(directories):
            is_last_dir = (i == len(directories) - 1) and len(files) == 0
            
            if is_last_dir:
                print(f"{child_prefix}└── ", end="")
                next_prefix = child_prefix + "    "
            else:
                print(f"{child_prefix}├── ", end="")
                next_prefix = child_prefix + "│   "
            
            _explore_directory(subdir, next_prefix, max_depth, current_depth + 1)
        
        # Print files
        for i, file in enumerate(files):
            is_last_item = i == len(files) - 1
            
            if is_last_item:
                print(f"{child_prefix}└── 📄 {file.name}")
            else:
                print(f"{child_prefix}├── 📄 {file.name}")
                
    except PermissionError:
        print(f"{prefix}❌ Permission denied: {directory.name}")
    except Exception as e:
        print(f"{prefix}❌ Error accessing {directory.name}: {e}")

def get_script_directory() -> str:
    """
    Get the directory where the script is located.
    This follows the pattern shown in the context for script execution.
    
    Returns:
        Absolute path to the script's directory
    """
    # Get the script's file path
    script_path = os.path.abspath(__file__)
    
    # Get the directory containing the script
    script_directory = os.path.dirname(script_path)
    
    return script_directory

def print_simple_structure(root_path: str) -> None:
    """
    Print a simple directory structure without tree formatting.
    
    Args:
        root_path: The root directory to explore
    """
    root = Path(root_path)
    
    if not root.exists() or not root.is_dir():
        print(f"❌ Invalid directory: {root_path}")
        return
    
    print(f"📁 Simple Structure for: {root.absolute()}")
    print("=" * 60)
    
    for item in root.rglob("*"):
        try:
            # Calculate relative path from root
            relative_path = item.relative_to(root)
            
            # Calculate indentation based on depth
            depth = len(relative_path.parts) - 1
            indent = "  " * depth
            
            if item.is_dir():
                print(f"{indent}📁 {relative_path}/")
            else:
                print(f"{indent}📄 {relative_path}")
                
        except Exception as e:
            print(f"❌ Error processing {item}: {e}")

def get_directory_stats(root_path: str) -> dict:
    """
    Calculate statistics about the directory structure.
    
    Args:
        root_path: The root directory to analyze
        
    Returns:
        Dictionary containing directory statistics
    """
    root = Path(root_path)
    stats = {
        'total_directories': 0,
        'total_files': 0,
        'total_size': 0,
        'file_types': {}
    }
    
    try:
        for item in root.rglob("*"):
            if item.is_dir():
                stats['total_directories'] += 1
            elif item.is_file():
                stats['total_files'] += 1
                
                # Calculate file size
                try:
                    file_size = item.stat().st_size
                    stats['total_size'] += file_size
                except:
                    pass
                
                # Track file extensions
                extension = item.suffix.lower() or 'no_extension'
                stats['file_types'][extension] = stats['file_types'].get(extension, 0) + 1
                
    except Exception as e:
        print(f"❌ Error calculating stats: {e}")
    
    return stats

def print_directory_stats(stats: dict) -> None:
    """
    Print formatted directory statistics.
    
    Args:
        stats: Statistics dictionary from get_directory_stats
    """
    print("\n📊 Directory Statistics:")
    print("-" * 30)
    print(f"Total Directories: {stats['total_directories']:,}")
    print(f"Total Files: {stats['total_files']:,}")
    print(f"Total Size: {stats['total_size']:,} bytes")
    
    if stats['file_types']:
        print("\nFile Types:")
        for ext, count in sorted(stats['file_types'].items()):
            print(f"  {ext}: {count:,} files")

def main():
    """
    Main function following best practices for script structure.
    Automatically uses the script's current directory as the starting point.
    """
    try:
        print("Directory Tree Explorer")
        print("=" * 40)
        
        # Get the directory where this script is located
        script_directory = get_script_directory()
        
        print(f"📂 Script location: {script_directory}")
        print(f"🎯 Exploring from script's directory")
        
        # Use script directory as the root path
        directory_path = script_directory
        
        # Ask user for display preference
        print("\nChoose display option:")
        print("1. Tree structure (formatted)")
        print("2. Simple list")
        print("3. Both + statistics")
        
        choice = input("Enter choice (1-3, default=1): ").strip() or "1"
        
        # Ask for depth limit
        depth_input = input("Enter maximum depth (or press Enter for unlimited): ").strip()
        max_depth = None
        if depth_input.isdigit():
            max_depth = int(depth_input)
        
        print("\n")
        
        # Execute based on user choice
        if choice == "1":
            print_directory_tree(directory_path, max_depth)
        elif choice == "2":
            print_simple_structure(directory_path)
        elif choice == "3":
            print_directory_tree(directory_path, max_depth)
            print("\n" + "=" * 60)
            print_simple_structure(directory_path)
            
            # Calculate and display statistics
            stats = get_directory_stats(directory_path)
            print_directory_stats(stats)
        else:
            print("Invalid choice, using tree structure")
            print_directory_tree(directory_path, max_depth)
        
        print("\n✅ Directory exploration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the directory permissions and try again.")
    
    finally:
        # Keep window open for user review
        print("\n" + "=" * 50)
        try:
            input("Press Enter to close the window...")
        except (KeyboardInterrupt, EOFError):
            print("\nClosing...")

if __name__ == "__main__":
    main()