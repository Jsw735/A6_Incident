import sys
from pathlib import Path

# Move up one directory to the project root
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")

# Add the project root to Python path for imports
sys.path.insert(0, str(project_root))

# Check for directories in the correct location
print("\nLooking for project directories in parent folder:")
expected_dirs = ['data', 'analysis', 'reporting', 'utils']
for dir_name in expected_dirs:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"  [OK] {dir_name}/ found at {dir_path}")
    else:
        print(f"  [MISSING] {dir_name}/ missing at {dir_path}")
        # Do not block in automated test runs; just report status
        continue
