"""
Tests for the Project Loader module
"""

import unittest
import tempfile
import os
import json
import shutil
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from project_loader import ProjectLoader

class TestProjectLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.loader = ProjectLoader()
        
        # Create a test project folder
        self.test_project_path = os.path.join(self.test_dir, 'test_project')
        os.makedirs(self.test_project_path)
        
        # Create test files
        with open(os.path.join(self.test_project_path, 'config.json'), 'w') as f:
            json.dump({'name': 'test_project', 'version': '1.0.0'}, f)
        
        with open(os.path.join(self.test_project_path, 'README.md'), 'w') as f:
            f.write('# Test Project\nThis is a test project.')
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_load_project_folder(self):
        """Test loading a project folder"""
        result = self.loader.load_project_folder(self.test_project_path)
        
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['project_name'], 'test_project')
        self.assertGreater(result['total_files'], 0)
        self.assertIn('load_id', result)
    
    def test_load_nonexistent_folder(self):
        """Test loading a non-existent folder"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_project_folder('/nonexistent/path')
    
    def test_load_file_instead_of_folder(self):
        """Test loading a file instead of a folder"""
        test_file = os.path.join(self.test_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        
        with self.assertRaises(ValueError):
            self.loader.load_project_folder(test_file)
    
    def test_list_loaded_projects(self):
        """Test listing loaded projects"""
        # Load a project first
        self.loader.load_project_folder(self.test_project_path)
        
        projects = self.loader.list_loaded_projects()
        self.assertGreater(len(projects), 0)
        self.assertEqual(projects[0]['project_name'], 'test_project')

if __name__ == '__main__':
    unittest.main()