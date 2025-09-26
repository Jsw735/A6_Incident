#!/usr/bin/env python3
"""
Targeted analysis of fallback methods and duplicates in metrics_calculator.py
"""

import ast
import re
from pathlib import Path

def analyze_metrics_calculator_specifics():
    """Analyze metrics_calculator.py for specific redundancy patterns."""
    
    metrics_file = Path('analysis/metrics_calculator.py')
    
    if not metrics_file.exists():
        print("❌ metrics_calculator.py not found")
        return
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🎯 TARGETED METRICS_CALCULATOR ANALYSIS")
    print("=" * 60)
    
    # Parse AST to get method definitions with line numbers
    tree = ast.parse(content)
    method_info = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info[item.name] = {
                        'line': item.lineno,
                        'class': node.name,
                        'args': [arg.arg for arg in item.args.args],
                        'returns_dict': 'Dict' in ast.unparse(item.returns) if item.returns else False
                    }
    
    print(f"\n📊 Found {len(method_info)} methods in IncidentMetricsCalculator")
    
    # Identify fallback methods
    fallback_methods = {name: info for name, info in method_info.items() if 'fallback' in name.lower()}
    print(f"\n🔄 FALLBACK METHODS: {len(fallback_methods)}")
    for name, info in fallback_methods.items():
        print(f"   - {name} (line {info['line']})")
    
    # Look for hasattr patterns that use fallback methods
    hasattr_pattern = r"if hasattr\(self, ['\"]([^'\"]+)['\"].*?:\s*\n\s*(.+?)else:\s*\n\s*(.+)"
    hasattr_matches = re.findall(hasattr_pattern, content, re.MULTILINE | re.DOTALL)
    
    print(f"\n🔍 HASATTR CONDITIONALS FOUND: {len(hasattr_matches)}")
    for i, (method_name, if_branch, else_branch) in enumerate(hasattr_matches):
        print(f"   {i+1}. Checking for: {method_name}")
        if_method = re.search(r'self\.([a-zA-Z_]\w*)', if_branch)
        else_method = re.search(r'self\.([a-zA-Z_]\w*)', else_branch)
        
        if if_method and else_method:
            print(f"      If exists: {if_method.group(1)}")
            print(f"      Fallback: {else_method.group(1)}")
    
    # Look for methods that have similar names (potential duplicates)
    method_names = list(method_info.keys())
    similar_groups = {}
    
    for method in method_names:
        base_name = method.replace('_fallback_', '').replace('_calculate_', '').replace('calculate_', '')
        if base_name not in similar_groups:
            similar_groups[base_name] = []
        similar_groups[base_name].append(method)
    
    print(f"\n🔄 POTENTIAL DUPLICATES (similar base names):")
    for base_name, methods in similar_groups.items():
        if len(methods) > 1:
            print(f"   {base_name}: {methods}")
    
    # Analyze method call patterns within the file
    method_call_pattern = r'self\.([a-zA-Z_]\w*)\s*\('
    internal_calls = re.findall(method_call_pattern, content)
    
    call_counts = {}
    for call in internal_calls:
        call_counts[call] = call_counts.get(call, 0) + 1
    
    print(f"\n📞 INTERNAL METHOD CALLS (within metrics_calculator):")
    for method, count in sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        if method in method_info:
            print(f"   {method}: called {count} times internally")
    
    # Check for unused internal methods
    defined_but_not_called_internally = set(method_info.keys()) - set(call_counts.keys())
    
    print(f"\n🤔 METHODS NOT CALLED INTERNALLY: {len(defined_but_not_called_internally)}")
    for method in sorted(defined_but_not_called_internally):
        if not method.startswith('__'):  # Skip special methods
            info = method_info[method]
            print(f"   - {method} (line {info['line']})")
    
    # Look for specific patterns we discussed
    print(f"\n🎯 SPECIFIC CLEANUP CANDIDATES:")
    
    # Check if _calculate_velocity_metrics exists (it doesn't, so fallback is always used)
    if '_calculate_velocity_metrics' not in method_info and '_fallback_velocity_metrics' in method_info:
        print("   ✅ _fallback_velocity_metrics can be renamed to _calculate_velocity_metrics")
    
    if '_calculate_weekly_comparison' not in method_info and '_fallback_weekly_comparison' in method_info:
        print("   ✅ _fallback_weekly_comparison can be renamed to _calculate_weekly_comparison")
    
    # Check for process methods
    process_methods = [m for m in method_info if 'process' in m.lower()]
    print(f"   📋 Process methods: {process_methods}")
    
    # Check which fallback methods are only used in hasattr checks
    print(f"\n💡 RECOMMENDED ACTIONS:")
    print("   1. Rename _fallback_velocity_metrics → _calculate_velocity_metrics")
    print("   2. Rename _fallback_weekly_comparison → _calculate_weekly_comparison") 
    print("   3. Remove hasattr checks in calculate_executive_summary_metrics")
    print("   4. Simplify _fallback_process_* methods or inline them")
    
def analyze_executive_summary_usage():
    """Check what executive_summary.py actually uses from metrics_calculator."""
    
    exec_file = Path('reporting/executive_summary.py')
    
    if not exec_file.exists():
        print("❌ executive_summary.py not found")
        return
        
    with open(exec_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\n📋 EXECUTIVE_SUMMARY.PY USAGE OF METRICS_CALCULATOR:")
    
    # Find references to metrics calculator methods
    metrics_refs = re.findall(r'calc\.([a-zA-Z_]\w*)', content)
    metrics_calls = {}
    for ref in metrics_refs:
        metrics_calls[ref] = metrics_calls.get(ref, 0) + 1
    
    print("   Methods called on metrics calculator:")
    for method, count in sorted(metrics_calls.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {method}: {count} times")
    
    # Check for any hardcoded references to fallback methods  
    fallback_refs = re.findall(r'fallback[a-zA-Z_]*', content, re.IGNORECASE)
    if fallback_refs:
        print(f"   Fallback references: {set(fallback_refs)}")
    else:
        print("   ✅ No direct fallback method references found")

if __name__ == "__main__":
    analyze_metrics_calculator_specifics()
    analyze_executive_summary_usage()