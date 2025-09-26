#!/usr/bin/env python3
"""
A6 Incident Reporting System - Command Line Interface
This provides a simple CLI interface to the incident reporting system
when Flask is not available.
"""

import os
import sys
import json
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from project_loader import ProjectLoader
from incident_manager import IncidentManager

def main():
    """Main CLI interface"""
    print("=" * 60)
    print("A6 Incident Reporting System - Command Line Interface")
    print("=" * 60)
    
    # Initialize components
    project_loader = ProjectLoader()
    incident_manager = IncidentManager()
    incident_manager.initialize_database()
    
    while True:
        print("\nOptions:")
        print("1. Load a project folder")
        print("2. List loaded projects")
        print("3. Create an incident")
        print("4. List incidents")
        print("5. Load sample project")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            load_project(project_loader)
        elif choice == '2':
            list_projects(project_loader)
        elif choice == '3':
            create_incident(incident_manager, project_loader)
        elif choice == '4':
            list_incidents(incident_manager)
        elif choice == '5':
            load_sample_project(project_loader)
        else:
            print("Invalid choice. Please try again.")

def load_project(project_loader):
    """Load a project folder"""
    print("\n--- Load Project Folder ---")
    
    project_path = input("Enter the full path to the project folder: ").strip()
    
    if not project_path:
        print("No path provided.")
        return
    
    if not os.path.exists(project_path):
        print(f"Path does not exist: {project_path}")
        return
    
    if not os.path.isdir(project_path):
        print(f"Path is not a directory: {project_path}")
        return
    
    try:
        print("Loading project... Please wait.")
        result = project_loader.load_project_folder(project_path)
        
        print("\n✓ Project loaded successfully!")
        print(f"Project Name: {result['project_name']}")
        print(f"Load ID: {result['load_id']}")
        print(f"Files Processed: {result['total_files']}")
        print(f"Destination: {result.get('destination_path', 'N/A')}")
        
        if result.get('errors'):
            print(f"Errors encountered: {len(result['errors'])}")
            for i, error in enumerate(result['errors'][:3]):
                print(f"  {i+1}. {error}")
            if len(result['errors']) > 3:
                print(f"  ... and {len(result['errors']) - 3} more")
        
    except Exception as e:
        print(f"\n✗ Failed to load project: {str(e)}")

def load_sample_project(project_loader):
    """Load the sample project"""
    print("\n--- Load Sample Project ---")
    
    sample_path = os.path.join(os.path.dirname(__file__), 'examples', 'sample_project')
    
    if not os.path.exists(sample_path):
        print(f"Sample project not found at: {sample_path}")
        return
    
    try:
        print("Loading sample project... Please wait.")
        result = project_loader.load_project_folder(sample_path)
        
        print("\n✓ Sample project loaded successfully!")
        print(f"Project Name: {result['project_name']}")
        print(f"Load ID: {result['load_id']}")
        print(f"Files Processed: {result['total_files']}")
        
        # Show file details
        print("\nFiles processed:")
        for file_info in result.get('files_processed', []):
            status = "✓" if file_info.get('processed') else "✗"
            print(f"  {status} {file_info['path']} ({file_info['type']})")
        
    except Exception as e:
        print(f"\n✗ Failed to load sample project: {str(e)}")

def list_projects(project_loader):
    """List all loaded projects"""
    print("\n--- Loaded Projects ---")
    
    projects = project_loader.list_loaded_projects()
    
    if not projects:
        print("No projects loaded yet.")
        return
    
    for i, project in enumerate(projects, 1):
        print(f"\n{i}. {project['project_name']}")
        print(f"   Load ID: {project['load_id']}")
        print(f"   Loaded: {project['loaded_at']}")
        print(f"   Files: {project.get('total_files', 0)}")
        print(f"   Status: {project['status']}")

def create_incident(incident_manager, project_loader):
    """Create a new incident"""
    print("\n--- Create Incident ---")
    
    title = input("Enter incident title: ").strip()
    if not title:
        print("Title is required.")
        return
    
    description = input("Enter description (optional): ").strip()
    
    print("Select severity:")
    print("1. Low")
    print("2. Medium")
    print("3. High")
    print("4. Critical")
    
    severity_map = {'1': 'low', '2': 'medium', '3': 'high', '4': 'critical'}
    severity_choice = input("Enter choice (1-4, default=2): ").strip() or '2'
    severity = severity_map.get(severity_choice, 'medium')
    
    # Optionally link to a project
    projects = project_loader.list_loaded_projects()
    project_id = None
    
    if projects:
        print(f"\nAvailable projects ({len(projects)}):")
        print("0. No project link")
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['project_name']} ({project['load_id']})")
        
        project_choice = input(f"Link to project (0-{len(projects)}, default=0): ").strip() or '0'
        
        try:
            project_idx = int(project_choice)
            if 1 <= project_idx <= len(projects):
                project_id = projects[project_idx - 1]['load_id']
        except ValueError:
            pass
    
    try:
        incident_id = incident_manager.create_incident(
            title=title,
            description=description,
            severity=severity,
            project_id=project_id
        )
        
        print(f"\n✓ Incident created successfully!")
        print(f"Incident ID: {incident_id}")
        print(f"Title: {title}")
        print(f"Severity: {severity}")
        if project_id:
            print(f"Linked to project: {project_id}")
        
    except Exception as e:
        print(f"\n✗ Failed to create incident: {str(e)}")

def list_incidents(incident_manager):
    """List all incidents"""
    print("\n--- Incidents ---")
    
    try:
        incidents = incident_manager.get_all_incidents()
        
        if not incidents:
            print("No incidents found.")
            return
        
        for incident in incidents:
            print(f"\n#{incident['id']} - {incident['title']}")
            print(f"   Status: {incident['status']}")
            print(f"   Severity: {incident['severity']}")
            print(f"   Created: {incident['created_at']}")
            if incident.get('description'):
                print(f"   Description: {incident['description'][:100]}...")
            if incident.get('project_id'):
                print(f"   Project: {incident['project_id']}")
            
    except Exception as e:
        print(f"\n✗ Failed to list incidents: {str(e)}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)