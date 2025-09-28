#!/usr/bin/env python3
"""
Comprehensive Test Suite for SAP Incident Reporting System
Tests all data processing, analyzer, and reporting modules
Following Python testing best practices with clear error reporting
"""

import sys
import traceback
import importlib
import inspect
from pathlib import Path
from datetime import datetime
import logging

class SystemTester:
    """
    Comprehensive testing class that validates all system components.
    Implements systematic testing with clear progress reporting.
    """
    
    def __init__(self):
        """Initialize the tester with logging and progress tracking."""
        
        # Fix path issue - handle running from tests directory
        current_path = Path(__file__).parent
        if current_path.name == 'tests':
            self.project_root = current_path.parent
            # Change to project root for proper imports
            import os
            os.chdir(self.project_root)
        else:
            self.project_root = Path.cwd()
        
        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))
        
        print(f"🔧 Project root set to: {self.project_root}")
        print(f"🔧 Working directory: {Path.cwd()}")
    
    self.setup_logging()
    self.test_results = {}
        self.setup_logging()
        self.test_results = {}
        self.project_root = Path.cwd()
        self.modules_to_test = [
            'data.data_processor',
            'data.data_validator', 
            'data.file_handler',
            'analysis.incident_analyzer',
            'analysis.keyword_analyzer',
            'analysis.metrics_calculator',
            'analysis.quality_analyzer',
            'reporting.excel_generator',
            'reporting.daily_reporter',
            'reporting.weekly_reporter',
            'reporting.executive_summary',
            'reporting.quality_reporter',
            'utils.date_utils',
            'utils.file_utils',
            'utils.validation_utils',
            'utils.format_utils'
        ]
        
    def setup_logging(self):
        """Configure logging for test execution."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('test_results.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_tests(self):
        """
        Run essential tests focused on file existence, code correctness, and dependencies.
        Following the context guidance about testing what's needed to get to the actual work.
        """
        print("=" * 60)
        print("SAP INCIDENT REPORTING SYSTEM - ESSENTIAL VALIDATION TESTS")
        print("=" * 60)
        print()
        
        # Step 1: Environment Check - ESSENTIAL for dependencies
        self.pause_and_continue("Step 1: Environment and Dependencies Check")
        self.test_environment()
        
        # Step 2: Configuration Files Check - ESSENTIAL for ini files and config
        self.pause_and_continue("Step 2: Configuration Files Check")
        self.test_configuration_files()
        
        # Step 3: Package Structure Validation - ESSENTIAL for file existence
        self.pause_and_continue("Step 3: Package Structure Validation")
        self.test_package_structure()
        
        # Step 4: Module Import Tests - ESSENTIAL for code correctness
        self.pause_and_continue("Step 4: Module Import Tests")
        self.test_module_imports()
        
        # Step 5: Class and Function Discovery - ESSENTIAL for dependency validation
        self.pause_and_continue("Step 5: Class and Function Discovery")
        self.discover_components()
        
        # Step 6: Basic Functionality Tests - ESSENTIAL for code error checking
        self.pause_and_continue("Step 6: Basic Functionality Tests")
        self.test_basic_functionality()
        
        # Step 7: Integration Tests - ESSENTIAL for dependency checking
        self.pause_and_continue("Step 7: Integration Tests")
        self.test_integration()
        
        # COMMENTED OUT - Not essential for getting to Excel work
        # # Step X: Utility Function Tests
        # self.pause_and_continue("Step X: Utility Function Tests")
        # self.test_utility_functions()
        
        # # Step X: Code Quality Checks
        # self.pause_and_continue("Step X: Code Quality and Documentation Checks")
        # self.test_code_quality()
        
        # # Step X: Calculator Functionality Test
        # self.pause_and_continue("Step X: Calculator Functionality Test")
        # self.test_calculator_functionality()
        
        # # Step X: File Processing Test
        # self.pause_and_continue("Step X: File Processing Test")
        # self.test_file_processing()
        
        # # Step X: Script Executability Check
        # self.pause_and_continue("Step X: Script Executability Check")
        # self.test_script_executability()
        
        # # Step X: Data Files and Resources
        # self.pause_and_continue("Step X: Data Files and Resources")
        # self.test_data_files()
        
        # # Step X: Documentation Check
        # self.pause_and_continue("Step X: Documentation Check")
        # self.test_documentation_files()
        
        # # Step X: Existing Tests Discovery
        # self.pause_and_continue("Step X: Existing Tests Discovery")
        # self.test_existing_tests()
        
        # FINAL Step: Generate Report
        self.pause_and_continue("Final Step: Generate Essential Validation Report")
        self.generate_final_report()
    
    def pause_and_continue(self, step_description):
        """Pause execution and wait for user input before continuing."""
        print(f"\n🔍 {step_description}")
        print("-" * len(step_description))
        
        try:
            input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping pause due to input error...")
        
        print()
    
    def test_environment(self):
        """Test the Python environment and required dependencies."""
        print("Testing Python environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check critical dependencies for Excel work
            dependencies = ['pandas', 'openpyxl', 'pathlib', 'datetime', 'logging', 'json']
            missing_deps = []
            
            for dep in dependencies:
                try:
                    importlib.import_module(dep)
                    print(f"✅ {dep} - Available")
                except ImportError:
                    print(f"❌ {dep} - Missing")
                    missing_deps.append(dep)
            
            # Check Excel-specific dependencies
            excel_deps = ['xlsxwriter', 'numpy']
            for dep in excel_deps:
                try:
                    importlib.import_module(dep)
                    print(f"✅ {dep} - Available (Excel support)")
                except ImportError:
                    print(f"⚠️  {dep} - Missing (Excel functionality may be limited)")
            
            self.test_results['environment'] = {
                'python_version': f"{python_version.major}.{python_version.minor}",
                'missing_dependencies': missing_deps,
                'status': 'PASS' if not missing_deps else 'FAIL'
            }
            
        except Exception as e:
            print(f"❌ Environment test failed: {str(e)}")
            self.test_results['environment'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_configuration_files(self):
        """Test for required configuration files and project structure."""
        print("Checking configuration and project files...")
        
        # Focus on essential config files for the system
        config_files_to_check = [
            'config.json',
            'settings.ini', 
            'config.yaml',
            '.env',
            'requirements.txt',
            'setup.py'
        ]
        
        found_configs = []
        missing_configs = []
        
        for config_file in config_files_to_check:
            if Path(config_file).exists():
                found_configs.append(config_file)
                print(f"  ✅ {config_file} found")
                
                # Validate ini files specifically
                if config_file.endswith('.ini'):
                    self._validate_ini_file(config_file)
                    
            else:
                missing_configs.append(config_file)
                print(f"  ⚠️  {config_file} not found")
        
        self.test_results['configuration_files'] = {
            'found_files': found_configs,
            'missing_files': missing_configs,
            'total_found': len(found_configs),
            'coverage': len(found_configs) / len(config_files_to_check)
        }
    
    def _validate_ini_file(self, ini_file):
        """Validate INI file format and content."""
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(ini_file)
            print(f"    ✅ {ini_file} - Valid INI format with {len(config.sections())} sections")
        except Exception as e:
            print(f"    ❌ {ini_file} - INI validation failed: {str(e)}")
        
    def test_package_structure(self):
        """Validate Python package structure with __init__.py files."""
        print("Validating package structure...")
        
        package_dirs = ['data', 'analysis', 'reporting', 'utils']
        structure_results = {}
        
        for pkg_dir in package_dirs:
            dir_path = self.project_root / pkg_dir
            init_file = dir_path / '__init__.py'
            
            if dir_path.exists():
                print(f"  ✅ {pkg_dir}/ directory exists")
                if init_file.exists():
                    print(f"  ✅ {pkg_dir}/__init__.py exists")
                    structure_results[pkg_dir] = 'COMPLETE'
                else:
                    print(f"  ❌ {pkg_dir}/__init__.py missing - WILL CAUSE IMPORT ERRORS")
                    structure_results[pkg_dir] = 'MISSING_INIT'
            else:
                print(f"  ❌ {pkg_dir}/ directory not found - CRITICAL ERROR")
                structure_results[pkg_dir] = 'MISSING_DIR'
        
        # Check for essential Python files
        essential_files = ['main.py', 'run.py', 'app.py']
        found_main = False
        for main_file in essential_files:
            if (self.project_root / main_file).exists():
                print(f"  ✅ {main_file} found")
                found_main = True
                break
        
        if not found_main:
            print(f"  ⚠️  No main entry point found ({', '.join(essential_files)})")
        
        self.test_results['package_structure'] = structure_results
    
    def test_module_imports(self):
        """Test importing all modules systematically - CRITICAL for code correctness."""
        print("Testing module imports...")
        
        import_results = {}
        
        for module_name in self.modules_to_test:
            try:
                print(f"  Testing import: {module_name}")
                module = importlib.import_module(module_name)
                print(f"  ✅ {module_name} - Import successful")
                import_results[module_name] = {
                    'status': 'PASS',
                    'module': module,
                    'file_path': getattr(module, '__file__', 'Unknown')
                }
                
            except ImportError as e:
                print(f"  ❌ {module_name} - Import failed: {str(e)}")
                print(f"      This indicates missing files or syntax errors!")
                import_results[module_name] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'error_type': 'ImportError'
                }
                
            except SyntaxError as e:
                print(f"  ❌ {module_name} - Syntax error: {str(e)}")
                print(f"      Code has syntax errors that must be fixed!")
                import_results[module_name] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'error_type': 'SyntaxError'
                }
                
            except Exception as e:
                print(f"  ❌ {module_name} - Unexpected error: {str(e)}")
                import_results[module_name] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        self.test_results['imports'] = import_results
    
    def discover_components(self):
        """Discover classes and functions in successfully imported modules."""
        print("Discovering classes and functions...")
        
        discovery_results = {}
        
        for module_name, result in self.test_results.get('imports', {}).items():
            if result['status'] == 'PASS':
                try:
                    module = result['module']
                    
                    # Find classes
                    classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)
                             if obj.__module__ == module.__name__]
                    
                    # Find functions
                    functions = [name for name, obj in inspect.getmembers(module, inspect.isfunction)
                               if obj.__module__ == module.__name__]
                    
                    print(f"  📦 {module_name}:")
                    print(f"    Classes: {classes}")
                    print(f"    Functions: {functions}")
                    
                    discovery_results[module_name] = {
                        'classes': classes,
                        'functions': functions,
                        'total_components': len(classes) + len(functions)
                    }
                    
                except Exception as e:
                    print(f"  ❌ {module_name} - Discovery failed: {str(e)}")
                    discovery_results[module_name] = {'error': str(e)}
        
        self.test_results['discovery'] = discovery_results
    
    def test_basic_functionality(self):
        """Test basic functionality of key classes - ESSENTIAL for code correctness."""
        print("Testing basic functionality...")
        
        functionality_results = {}
        
        # Test key classes that should be instantiable
        test_classes = {
            'reporting.excel_generator': 'ExcelGenerator',
            'data.data_processor': 'DataProcessor',
            'analysis.incident_analyzer': 'IncidentAnalyzer'
        }
        
        for module_name, class_name in test_classes.items():
            try:
                if (module_name in self.test_results.get('imports', {}) and 
                    self.test_results['imports'][module_name]['status'] == 'PASS'):
                    
                    module = self.test_results['imports'][module_name]['module']
                    
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        
                        # Try to instantiate - this will catch __init__ errors
                        try:
                            instance = cls()
                            print(f"  ✅ {module_name}.{class_name} - Instantiation successful")
                            
                            functionality_results[f"{module_name}.{class_name}"] = {
                                'instantiation': 'PASS',
                                'class_found': True
                            }
                        except Exception as init_error:
                            print(f"  ❌ {module_name}.{class_name} - Instantiation failed: {str(init_error)}")
                            functionality_results[f"{module_name}.{class_name}"] = {
                                'instantiation': 'FAIL',
                                'class_found': True,
                                'error': str(init_error)
                            }
                        
                    else:
                        print(f"  ❌ {module_name} - Class {class_name} not found")
                        functionality_results[f"{module_name}.{class_name}"] = {
                            'instantiation': 'FAIL',
                            'class_found': False,
                            'error': f'Class {class_name} not found'
                        }
                        
            except Exception as e:
                print(f"  ❌ {module_name}.{class_name} - Functionality test failed: {str(e)}")
                functionality_results[f"{module_name}.{class_name}"] = {
                    'instantiation': 'FAIL',
                    'error': str(e)
                }
        
        self.test_results['functionality'] = functionality_results
    
    def test_integration(self):
        """Test integration between modules - ESSENTIAL for dependency validation."""
        print("Testing integration scenarios...")
        
        integration_results = {}
        
        try:
            # Test 1: Data flow simulation
            print("  Testing data flow simulation...")
            workflow_test = self.simulate_basic_workflow()
            integration_results['workflow'] = workflow_test
            
            # Test 2: Cross-module dependencies
            print("  Testing cross-module dependencies...")
            dependency_test = self.test_module_dependencies()
            integration_results['dependencies'] = dependency_test
            
        except Exception as e:
            print(f"  ❌ Integration test failed: {str(e)}")
            integration_results['error'] = str(e)
        
        self.test_results['integration'] = integration_results
    
    def simulate_basic_workflow(self):
        """Simulate a basic data processing workflow."""
        try:
            # Test pandas availability for Excel work
            try:
                import pandas as pd
            except ImportError:
                print("    ❌ Pandas not available - CRITICAL for Excel processing")
                return {'status': 'FAIL', 'error': 'Pandas not available'}
            
            # Create sample data
            sample_data = pd.DataFrame({
                'incident_id': ['INC001', 'INC002'],
                'status': ['Open', 'Closed'],
                'priority': ['High', 'Medium']
            })
            
            print("    ✅ Sample data created")
            
            # Test if we can pass data between components
            workflow_steps = []
            
            # Try data processing
            if 'data.data_processor' in self.test_results.get('imports', {}):
                if self.test_results['imports']['data.data_processor']['status'] == 'PASS':
                    workflow_steps.append('data_processor_available')
                    print("    ✅ Data processor available")
            
            # Try analysis
            if 'analysis.incident_analyzer' in self.test_results.get('imports', {}):
                if self.test_results['imports']['analysis.incident_analyzer']['status'] == 'PASS':
                    workflow_steps.append('analyzer_available')
                    print("    ✅ Incident analyzer available")
            
            # Try reporting - CRITICAL for Excel work
            if 'reporting.excel_generator' in self.test_results.get('imports', {}):
                if self.test_results['imports']['reporting.excel_generator']['status'] == 'PASS':
                    workflow_steps.append('reporting_available')
                    print("    ✅ Excel generator available")
            else:
                print("    ❌ Excel generator not available - CRITICAL for Excel work")
            
            return {
                'status': 'PASS',
                'workflow_steps': workflow_steps,
                'sample_data_rows': len(sample_data),
                'workflow_completeness': len(workflow_steps) / 3
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_module_dependencies(self):
        """Test dependencies between modules."""
        dependency_issues = []
        
        try:
            # Check for circular imports
            for module_name in self.modules_to_test:
                if module_name in self.test_results.get('imports', {}):
                    result = self.test_results['imports'][module_name]
                    if result['status'] == 'FAIL' and 'circular' in result.get('error', '').lower():
                        dependency_issues.append(f"Circular import in {module_name}")
            
            # Check for missing dependencies
            import_results = self.test_results.get('imports', {})
            failed_imports = [name for name, result in import_results.items() 
                             if result['status'] == 'FAIL']
            
            if failed_imports:
                dependency_issues.extend([f"Failed import: {name}" for name in failed_imports])
            
            return {
                'status': 'PASS' if not dependency_issues else 'FAIL',
                'issues': dependency_issues,
                'total_issues': len(dependency_issues)
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def generate_final_report(self):
        """Generate and display the final test report."""
        print("\n" + "=" * 60)
        print("ESSENTIAL VALIDATION REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_modules = len(self.modules_to_test)
        successful_imports = sum(1 for result in self.test_results.get('imports', {}).values() 
                               if result['status'] == 'PASS')
        
        print(f"\n📊 CRITICAL VALIDATION SUMMARY:")
        print(f"  Total modules tested: {total_modules}")
        print(f"  Successful imports: {successful_imports}")
        print(f"  Failed imports: {total_modules - successful_imports}")
        
        # Environment status
        env_status = self.test_results.get('environment', {}).get('status', 'UNKNOWN')
        print(f"  Environment status: {env_status}")
        
        # Configuration files
        config_files = self.test_results.get('configuration_files', {})
        if config_files:
            print(f"\n📁 CONFIGURATION FILES:")
            print(f"  Found: {config_files.get('total_found', 0)} files")
            print(f"  Coverage: {config_files.get('coverage', 0):.1%}")
        
        # Package structure
        package_structure = self.test_results.get('package_structure', {})
        if package_structure:
            print(f"\n📦 PACKAGE STRUCTURE:")
            complete_packages = sum(1 for status in package_structure.values() if status == 'COMPLETE')
            missing_packages = sum(1 for status in package_structure.values() if status in ['MISSING_DIR', 'MISSING_INIT'])
            print(f"  Complete packages: {complete_packages}/{len(package_structure)}")
            if missing_packages > 0:
                print(f"  ❌ Missing packages/init files: {missing_packages} - MUST BE FIXED")
        
        # Integration test results
        integration = self.test_results.get('integration', {})
        if integration:
            print(f"\n🔗 INTEGRATION STATUS:")
            workflow = integration.get('workflow', {})
            if workflow.get('status') == 'PASS':
                print(f"  ✅ Workflow simulation passed")
                print(f"  📊 Workflow completeness: {workflow.get('workflow_completeness', 0):.1%}")
            else:
                print(f"  ❌ Workflow simulation failed - CRITICAL")
            
            dependencies = integration.get('dependencies', {})
            if dependencies.get('total_issues', 0) > 0:
                print(f"  ❌ Dependency issues found: {dependencies.get('total_issues', 0)}")

        # Critical errors that must be fixed
        print(f"\n🚨 CRITICAL ISSUES THAT MUST BE FIXED:")
        critical_issues = []
        
        if env_status == 'FAIL':
            critical_issues.append("Missing required dependencies")
        
        failed_imports = [name for name, result in self.test_results.get('imports', {}).items() 
                         if result['status'] == 'FAIL']
        if failed_imports:
            critical_issues.append(f"Failed imports: {', '.join(failed_imports)}")
        
        missing_packages = [pkg for pkg, status in package_structure.items() 
                           if status in ['MISSING_DIR', 'MISSING_INIT']]
        if missing_packages:
            critical_issues.append(f"Missing packages/init files: {', '.join(missing_packages)}")
        
        if critical_issues:
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("  🎉 No critical issues found! Ready to proceed with Excel work.")
        
        # Save detailed report
        self.save_detailed_report()
        
        print(f"\n📄 Detailed report saved to: test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def save_detailed_report(self):
        """Save detailed test results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"test_results_{timestamp}.log"
        
        try:
            with open(report_file, 'w') as f:
                f.write("SAP INCIDENT REPORTING SYSTEM - ESSENTIAL VALIDATION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Test executed: {datetime.now()}\n\n")
                
                # Write all test results
                for section, results in self.test_results.items():
                    f.write(f"\n{section.upper()} RESULTS:\n")
                    f.write("-" * 30 + "\n")
                    f.write(str(results) + "\n")
                    
        except Exception as e:
            print(f"Warning: Could not save detailed report: {e}")

def main():
    """Main function to run the essential validation test suite."""
    tester = SystemTester()
    
    try:
        tester.run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        
     # ADD THIS - Critical for preventing immediate closure
    finally:
        input("\nPress any key to close the test window...")

if __name__ == "__main__":
    main()