"""
Project Loader Module for A6 Incident Reporting System

This module handles loading existing project folders into the incident reporting system.
It can process various file formats and organize project data for incident tracking.
"""

import os
import json
import yaml
import csv
import configparser
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectLoader:
    """Handles loading and managing external project folders"""
    
    def __init__(self, config_file: str = "config.ini"):
        """Initialize the project loader with configuration"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # Get configuration values
        self.supported_formats = self.config.get('project_loader', 'supported_formats', fallback='json,csv,yaml,txt').split(',')
        self.project_folder_path = self.config.get('project_loader', 'project_folder_path', fallback='projects/')
        self.backup_folder = self.config.get('project_loader', 'backup_folder', fallback='backups/')
        self.max_file_size = self.config.get('project_loader', 'max_file_size', fallback='50MB')
        
        # Create directories if they don't exist
        os.makedirs(self.project_folder_path, exist_ok=True)
        os.makedirs(self.backup_folder, exist_ok=True)
        
        self.loaded_projects = {}
    
    def load_project_folder(self, source_path: str) -> Dict[str, Any]:
        """
        Load a project folder into the incident reporting system
        
        Args:
            source_path: Path to the existing project folder
            
        Returns:
            Dict containing project information and load status
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Project path does not exist: {source_path}")
        
        if not os.path.isdir(source_path):
            raise ValueError(f"Path is not a directory: {source_path}")
        
        project_name = os.path.basename(source_path.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create project entry
        project_info = {
            'project_name': project_name,
            'source_path': source_path,
            'loaded_at': datetime.now().isoformat(),
            'load_id': f"{project_name}_{timestamp}",
            'files_processed': [],
            'errors': [],
            'status': 'loading'
        }
        
        try:
            # Copy project folder to our projects directory
            dest_path = os.path.join(self.project_folder_path, f"{project_name}_{timestamp}")
            
            # If destination exists, create a new unique name
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                dest_path = f"{original_dest_path}_{counter}"
                counter += 1
            
            shutil.copytree(source_path, dest_path)
            
            project_info['destination_path'] = dest_path
            
            # Process files in the project folder
            self._process_project_files(dest_path, project_info)
            
            # Create project metadata file
            metadata_file = os.path.join(dest_path, 'project_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(project_info, f, indent=2)
            
            project_info['status'] = 'completed'
            project_info['metadata_file'] = metadata_file
            
            # Store in loaded projects
            self.loaded_projects[project_info['load_id']] = project_info
            
            logger.info(f"Successfully loaded project: {project_name}")
            
        except Exception as e:
            project_info['status'] = 'failed'
            project_info['errors'].append(str(e))
            logger.error(f"Failed to load project {project_name}: {str(e)}")
            raise
        
        return project_info
    
    def _process_project_files(self, project_path: str, project_info: Dict[str, Any]):
        """Process files in the loaded project folder"""
        processed_files = []
        
        for root, dirs, files in os.walk(project_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)
                
                try:
                    file_info = self._analyze_file(file_path, relative_path)
                    processed_files.append(file_info)
                except Exception as e:
                    error_msg = f"Error processing {relative_path}: {str(e)}"
                    project_info['errors'].append(error_msg)
                    logger.warning(error_msg)
        
        project_info['files_processed'] = processed_files
        project_info['total_files'] = len(processed_files)
    
    def _analyze_file(self, file_path: str, relative_path: str) -> Dict[str, Any]:
        """Analyze a single file and extract relevant information"""
        file_info = {
            'path': relative_path,
            'size': os.path.getsize(file_path),
            'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'type': 'unknown',
            'processed': False
        }
        
        # Determine file type
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        if ext in ['json', 'js']:
            file_info['type'] = 'json'
            file_info = self._process_json_file(file_path, file_info)
        elif ext in ['yaml', 'yml']:
            file_info['type'] = 'yaml'
            file_info = self._process_yaml_file(file_path, file_info)
        elif ext in ['csv']:
            file_info['type'] = 'csv'
            file_info = self._process_csv_file(file_path, file_info)
        elif ext in ['txt', 'md', 'log']:
            file_info['type'] = 'text'
            file_info = self._process_text_file(file_path, file_info)
        
        return file_info
    
    def _process_json_file(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_info['structure'] = self._get_data_structure(data)
                file_info['processed'] = True
        except Exception as e:
            file_info['error'] = str(e)
        return file_info
    
    def _process_yaml_file(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process YAML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                file_info['structure'] = self._get_data_structure(data)
                file_info['processed'] = True
        except Exception as e:
            file_info['error'] = str(e)
        return file_info
    
    def _process_csv_file(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSV files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                row_count = sum(1 for _ in reader)
                file_info['headers'] = headers
                file_info['rows'] = row_count
                file_info['processed'] = True
        except Exception as e:
            file_info['error'] = str(e)
        return file_info
    
    def _process_text_file(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_info['lines'] = len(content.splitlines())
                file_info['characters'] = len(content)
                file_info['processed'] = True
        except Exception as e:
            file_info['error'] = str(e)
        return file_info
    
    def _get_data_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze the structure of data objects"""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys())[:10],  # First 10 keys
                'total_keys': len(data.keys())
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'item_types': list(set(type(item).__name__ for item in data[:10]))
            }
        else:
            return {
                'type': type(data).__name__
            }
    
    def list_loaded_projects(self) -> List[Dict[str, Any]]:
        """Return a list of all loaded projects"""
        return list(self.loaded_projects.values())
    
    def get_project_info(self, load_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific loaded project"""
        return self.loaded_projects.get(load_id)
    
    def remove_project(self, load_id: str) -> bool:
        """Remove a loaded project"""
        if load_id in self.loaded_projects:
            project_info = self.loaded_projects[load_id]
            
            # Move to backup folder
            if 'destination_path' in project_info:
                backup_path = os.path.join(self.backup_folder, os.path.basename(project_info['destination_path']))
                shutil.move(project_info['destination_path'], backup_path)
            
            del self.loaded_projects[load_id]
            return True
        
        return False

# CLI interface for the project loader
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load a project folder into A6 Incident Reporting System')
    parser.add_argument('project_path', help='Path to the project folder to load')
    
    args = parser.parse_args()
    
    loader = ProjectLoader()
    
    try:
        result = loader.load_project_folder(args.project_path)
        print("Project loaded successfully!")
        print(f"Project Name: {result['project_name']}")
        print(f"Load ID: {result['load_id']}")
        print(f"Files Processed: {result['total_files']}")
        
        if result['errors']:
            print(f"Errors: {len(result['errors'])}")
            for error in result['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
                
    except Exception as e:
        print(f"Failed to load project: {str(e)}")
        exit(1)