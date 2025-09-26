# Simple test to check if we can import the module
import sys
sys.path.append('.')

try:
    from reporting.executive_summary import EnhancedExecutiveSummary
    print("✅ Import successful")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
except IndentationError as e:
    print(f"❌ Indentation error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")