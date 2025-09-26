#!/usr/bin/env python3
"""
Comprehensive Method Usage Analysis for Incident Reporting System
Identifies used vs unused methods across the entire project.
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict

def analyze_method_usage():
    """Analyze all methods defined and used across the project."""
    
    project_root = Path('.')
    skip_dirs = {'.venv', '__pycache__', '.git', 'node_modules'}
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"📁 Found {len(python_files)} Python files to analyze\n")
    
    # Store method definitions and calls
    method_definitions = defaultdict(list)  # method_name -> [(file, class, line)]
    method_calls = defaultdict(set)         # method_name -> {files_that_call_it}
    import_usage = defaultdict(set)         # module -> {importing_files}
    
    # Track public methods vs private methods
    public_methods = set()
    private_methods = set()
    
    # Analyze each file
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            relative_path = str(py_file.relative_to(project_root))
            
            # Parse AST for method definitions
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    # Class methods
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_name = item.name
                                method_definitions[method_name].append((relative_path, node.name, getattr(item, 'lineno', 0)))
                                
                                if method_name.startswith('_'):
                                    private_methods.add(method_name)
                                else:
                                    public_methods.add(method_name)
                    
                    # Module-level functions
                    elif isinstance(node, ast.FunctionDef):
                        method_name = node.name
                        method_definitions[method_name].append((relative_path, 'module', getattr(node, 'lineno', 0)))
                        
                        if method_name.startswith('_'):
                            private_methods.add(method_name)
                        else:
                            public_methods.add(method_name)
                    
                    # Import tracking
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_usage[node.module].add(relative_path)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            import_usage[alias.name].add(relative_path)
                            
            except SyntaxError:
                print(f"⚠️  Syntax error in {relative_path}, skipping AST analysis")
            except Exception as e:
                print(f"⚠️  Error parsing {relative_path}: {e}")
            
            # Find method calls using pattern matching
            # Look for method calls: word followed by opening parenthesis
            call_patterns = [
                r'\b([a-zA-Z_]\w*)\s*\(',  # Direct calls
                r'\.([a-zA-Z_]\w*)\s*\(',  # Method calls on objects
                r'self\.([a-zA-Z_]\w*)\s*\(',  # Self method calls
            ]
            
            for pattern in call_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    method_calls[match].add(relative_path)
            
            # Also look for references without parentheses (e.g., in hasattr, getattr)
            ref_patterns = [
                r"hasattr\([^,]+,\s*['\"]([a-zA-Z_]\w*)['\"]",
                r"getattr\([^,]+,\s*['\"]([a-zA-Z_]\w*)['\"]",
                r"setattr\([^,]+,\s*['\"]([a-zA-Z_]\w*)['\"]",
            ]
            
            for pattern in ref_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    method_calls[match].add(relative_path)
                    
        except Exception as e:
            print(f"❌ Error processing {py_file}: {e}")
    
    return method_definitions, method_calls, public_methods, private_methods, import_usage

def generate_report(method_definitions, method_calls, public_methods, private_methods, import_usage):
    """Generate comprehensive usage report."""
    
    print("=" * 80)
    print("📊 COMPREHENSIVE METHOD USAGE ANALYSIS")
    print("=" * 80)
    
    all_defined_methods = set(method_definitions.keys())
    all_called_methods = set(method_calls.keys())
    
    # Calculate usage statistics
    used_methods = all_defined_methods & all_called_methods
    unused_methods = all_defined_methods - all_called_methods
    undefined_calls = all_called_methods - all_defined_methods
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Total methods defined: {len(all_defined_methods)}")
    print(f"   Total method calls found: {len(all_called_methods)}")
    print(f"   Used methods: {len(used_methods)} ({len(used_methods)/len(all_defined_methods)*100:.1f}%)")
    print(f"   Unused methods: {len(unused_methods)} ({len(unused_methods)/len(all_defined_methods)*100:.1f}%)")
    print(f"   External/undefined calls: {len(undefined_calls)}")
    
    print(f"\n🔓 PUBLIC vs 🔒 PRIVATE BREAKDOWN:")
    public_used = public_methods & used_methods
    public_unused = public_methods & unused_methods
    private_used = private_methods & used_methods
    private_unused = private_methods & unused_methods
    
    print(f"   Public methods: {len(public_methods)} total")
    print(f"     - Used: {len(public_used)} ({len(public_used)/len(public_methods)*100:.1f}%)")
    print(f"     - Unused: {len(public_unused)} ({len(public_unused)/len(public_methods)*100:.1f}%)")
    print(f"   Private methods: {len(private_methods)} total")
    print(f"     - Used: {len(private_used)} ({len(private_used)/len(private_methods)*100:.1f}%)")
    print(f"     - Unused: {len(private_unused)} ({len(private_unused)/len(private_methods)*100:.1f}%)")
    
    # Key modules analysis
    print(f"\n🏗️ KEY MODULE ANALYSIS:")
    key_modules = ['main', 'analysis.metrics_calculator', 'reporting.executive_summary', 'data_processing']
    
    for module_pattern in key_modules:
        relevant_methods = [m for m in method_definitions if any(module_pattern in location[0] for location in method_definitions[m])]
        used_in_module = [m for m in relevant_methods if m in used_methods]
        unused_in_module = [m for m in relevant_methods if m in unused_methods]
        
        print(f"   {module_pattern}:")
        print(f"     Methods: {len(relevant_methods)}, Used: {len(used_in_module)}, Unused: {len(unused_in_module)}")
    
    # Most called methods
    print(f"\n🔥 MOST FREQUENTLY CALLED METHODS:")
    method_call_counts = [(method, len(files)) for method, files in method_calls.items() if method in all_defined_methods]
    method_call_counts.sort(key=lambda x: x[1], reverse=True)
    
    for method, count in method_call_counts[:15]:
        locations = method_definitions.get(method, [])
        location_str = f" (defined in {len(locations)} place{'s' if len(locations) != 1 else ''})"
        print(f"     {method}: called from {count} files{location_str}")
    
    # Unused methods that might be safe to remove
    print(f"\n🗑️ POTENTIALLY SAFE TO REMOVE (unused methods):")
    
    # Filter out special methods and common patterns that are often unused but needed
    special_methods = {'__init__', '__new__', '__str__', '__repr__', '__getitem__', '__setitem__', '__contains__'}
    test_methods = [m for m in unused_methods if 'test_' in m or any('test' in loc[0] for loc in method_definitions[m])]
    
    safe_to_remove = unused_methods - special_methods - set(test_methods)
    
    print(f"   Total unused (excluding special methods & tests): {len(safe_to_remove)}")
    
    # Group by file for easier cleanup
    by_file = defaultdict(list)
    for method in safe_to_remove:
        for location in method_definitions[method]:
            by_file[location[0]].append((method, location[1], location[2]))
    
    for file_path in sorted(by_file.keys())[:10]:  # Show first 10 files
        methods = by_file[file_path]
        print(f"\n   📁 {file_path}: {len(methods)} unused methods")
        for method, class_name, line in sorted(methods)[:8]:  # Show first 8 per file
            class_info = f"in {class_name}" if class_name != 'module' else "module-level"
            print(f"      - {method} ({class_info}, line {line})")
        if len(methods) > 8:
            print(f"      ... and {len(methods)-8} more")
    
    # Specific analysis for metrics_calculator.py
    print(f"\n🎯 METRICS_CALCULATOR.PY SPECIFIC ANALYSIS:")
    metrics_methods = [m for m in method_definitions if any('metrics_calculator' in loc[0] for loc in method_definitions[m])]
    metrics_used = [m for m in metrics_methods if m in used_methods]
    metrics_unused = [m for m in metrics_methods if m in unused_methods]
    
    print(f"   Total methods in metrics_calculator: {len(metrics_methods)}")
    print(f"   Used: {len(metrics_used)}")
    print(f"   Unused: {len(metrics_unused)}")
    
    # Show fallback methods specifically
    fallback_methods = [m for m in metrics_methods if 'fallback' in m.lower()]
    fallback_used = [m for m in fallback_methods if m in used_methods]
    fallback_unused = [m for m in fallback_methods if m in unused_methods]
    
    print(f"\n   Fallback methods: {len(fallback_methods)}")
    print(f"     Used: {fallback_used}")
    print(f"     Unused: {fallback_unused}")
    
    # Show all unused methods in metrics_calculator
    if metrics_unused:
        print(f"\n   Unused methods in metrics_calculator:")
        for method in sorted(metrics_unused):
            locations = method_definitions[method]
            for location in locations:
                if 'metrics_calculator' in location[0]:
                    class_info = f"in {location[1]}" if location[1] != 'module' else "module-level"
                    print(f"      - {method} ({class_info}, line {location[2]})")

if __name__ == "__main__":
    try:
        method_defs, method_calls, public_methods, private_methods, import_usage = analyze_method_usage()
        generate_report(method_defs, method_calls, public_methods, private_methods, import_usage)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()