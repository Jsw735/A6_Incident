#!/usr/bin/env python3
"""
A6 Incident Reporting System - Main Application
This is the main entry point for the incident reporting system.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from project_loader import ProjectLoader
from incident_manager import IncidentManager

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize components
project_loader = ProjectLoader()
incident_manager = IncidentManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/load-project', methods=['GET', 'POST'])
def load_project():
    """Handle project folder loading"""
    if request.method == 'POST':
        project_path = request.form.get('project_path')
        if project_path and os.path.exists(project_path):
            try:
                result = project_loader.load_project_folder(project_path)
                return jsonify({
                    'success': True, 
                    'message': f'Project loaded successfully: {result["project_name"]}',
                    'details': result
                })
            except Exception as e:
                return jsonify({
                    'success': False, 
                    'message': f'Error loading project: {str(e)}'
                })
        else:
            return jsonify({
                'success': False, 
                'message': 'Invalid project path provided'
            })
    
    return render_template('load_project.html')

@app.route('/api/projects')
def list_projects():
    """API endpoint to list loaded projects"""
    projects = project_loader.list_loaded_projects()
    return jsonify(projects)

@app.route('/api/incidents')
def list_incidents():
    """API endpoint to list incidents"""
    incidents = incident_manager.get_all_incidents()
    return jsonify(incidents)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('projects', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    
    # Initialize database
    incident_manager.initialize_database()
    
    print("Starting A6 Incident Reporting System...")
    print("Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)