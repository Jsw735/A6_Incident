"""
Incident Manager Module for A6 Incident Reporting System

This module handles incident creation, tracking, and management.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class IncidentManager:
    """Manages incident reporting and tracking"""
    
    def __init__(self, db_path: str = "data/incidents.db"):
        """Initialize the incident manager"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def initialize_database(self):
        """Initialize the incidents database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'open',
                    project_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Create project_incidents table for linking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_incidents (
                    incident_id INTEGER,
                    project_load_id TEXT,
                    file_path TEXT,
                    FOREIGN KEY (incident_id) REFERENCES incidents (id)
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def create_incident(self, title: str, description: str = "", 
                       severity: str = "medium", project_id: str = None,
                       metadata: Dict[str, Any] = None) -> int:
        """Create a new incident"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO incidents (title, description, severity, project_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (title, description, severity, project_id, 
                 json.dumps(metadata) if metadata else None))
            
            incident_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created incident {incident_id}: {title}")
            return incident_id
    
    def get_all_incidents(self) -> List[Dict[str, Any]]:
        """Get all incidents"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM incidents ORDER BY created_at DESC
            ''')
            
            incidents = []
            for row in cursor.fetchall():
                incident = dict(row)
                if incident['metadata']:
                    incident['metadata'] = json.loads(incident['metadata'])
                incidents.append(incident)
            
            return incidents
    
    def get_incident_by_id(self, incident_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific incident by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM incidents WHERE id = ?', (incident_id,))
            row = cursor.fetchone()
            
            if row:
                incident = dict(row)
                if incident['metadata']:
                    incident['metadata'] = json.loads(incident['metadata'])
                return incident
            
            return None
    
    def update_incident_status(self, incident_id: int, status: str) -> bool:
        """Update incident status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            resolved_at = datetime.now() if status == 'resolved' else None
            
            cursor.execute('''
                UPDATE incidents 
                SET status = ?, updated_at = CURRENT_TIMESTAMP, resolved_at = ?
                WHERE id = ?
            ''', (status, resolved_at, incident_id))
            
            success = cursor.rowcount > 0
            conn.commit()
            
            if success:
                logger.info(f"Updated incident {incident_id} status to {status}")
            
            return success
    
    def link_incident_to_project(self, incident_id: int, project_load_id: str, 
                                file_path: str = None):
        """Link an incident to a loaded project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO project_incidents (incident_id, project_load_id, file_path)
                VALUES (?, ?, ?)
            ''', (incident_id, project_load_id, file_path))
            
            conn.commit()
            logger.info(f"Linked incident {incident_id} to project {project_load_id}")
    
    def get_incidents_for_project(self, project_load_id: str) -> List[Dict[str, Any]]:
        """Get all incidents linked to a specific project"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT i.*, pi.file_path
                FROM incidents i
                JOIN project_incidents pi ON i.id = pi.incident_id
                WHERE pi.project_load_id = ?
                ORDER BY i.created_at DESC
            ''', (project_load_id,))
            
            incidents = []
            for row in cursor.fetchall():
                incident = dict(row)
                if incident['metadata']:
                    incident['metadata'] = json.loads(incident['metadata'])
                incidents.append(incident)
            
            return incidents