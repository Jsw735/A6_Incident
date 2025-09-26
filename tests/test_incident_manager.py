"""
Tests for the Incident Manager module
"""

import unittest
import tempfile
import os

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from incident_manager import IncidentManager

class TestIncidentManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_incidents.db')
        self.manager = IncidentManager(self.db_path)
        self.manager.initialize_database()
    
    def test_create_incident(self):
        """Test creating an incident"""
        incident_id = self.manager.create_incident(
            title="Test Incident",
            description="This is a test incident",
            severity="high"
        )
        
        self.assertIsInstance(incident_id, int)
        self.assertGreater(incident_id, 0)
    
    def test_get_all_incidents(self):
        """Test getting all incidents"""
        # Create test incidents
        self.manager.create_incident("Incident 1", "Description 1")
        self.manager.create_incident("Incident 2", "Description 2")
        
        incidents = self.manager.get_all_incidents()
        self.assertEqual(len(incidents), 2)
    
    def test_get_incident_by_id(self):
        """Test getting a specific incident"""
        incident_id = self.manager.create_incident(
            title="Test Incident",
            description="Test Description",
            severity="medium"
        )
        
        incident = self.manager.get_incident_by_id(incident_id)
        self.assertIsNotNone(incident)
        self.assertEqual(incident['title'], "Test Incident")
        self.assertEqual(incident['severity'], "medium")
    
    def test_update_incident_status(self):
        """Test updating incident status"""
        incident_id = self.manager.create_incident("Test Incident", "Description")
        
        success = self.manager.update_incident_status(incident_id, "resolved")
        self.assertTrue(success)
        
        incident = self.manager.get_incident_by_id(incident_id)
        self.assertEqual(incident['status'], "resolved")
        self.assertIsNotNone(incident['resolved_at'])
    
    def test_link_incident_to_project(self):
        """Test linking incident to project"""
        incident_id = self.manager.create_incident("Test Incident", "Description")
        
        # This should not raise an exception
        self.manager.link_incident_to_project(incident_id, "test_project_123", "/test/file.json")

if __name__ == '__main__':
    unittest.main()