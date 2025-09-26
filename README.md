# A6 Incident Reporting System

A comprehensive incident reporting system that allows you to load existing project folders and track incidents efficiently.

## Features

- **Project Loading**: Load existing project folders with automatic file analysis
- **Incident Management**: Create, track, and manage incidents
- **File Analysis**: Automatic analysis of JSON, YAML, CSV, and text files
- **Web Interface**: User-friendly web interface for project and incident management
- **API Endpoints**: RESTful API for integration with other systems

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**:
   Open your browser to `http://localhost:5000`

4. **Load a Project**:
   - Go to the "Load Project Folder" page
   - Enter the path to an existing project folder
   - The system will copy and analyze all files

## Project Structure

```
A6_Incident/
├── app.py                 # Main Flask application
├── config.ini            # Configuration file
├── requirements.txt       # Python dependencies
├── src/                   # Source code modules
│   ├── project_loader.py  # Project loading functionality
│   ├── incident_manager.py # Incident management
│   └── __init__.py
├── templates/            # HTML templates
├── tests/                # Unit tests
├── examples/             # Sample project folders
├── data/                 # Database files (created at runtime)
├── logs/                 # Log files (created at runtime)
├── projects/            # Loaded project files (created at runtime)
└── backups/             # Project backups (created at runtime)
```

## Usage

### Loading a Project Folder

#### Via Web Interface:
1. Navigate to `/load-project`
2. Enter the full path to your project folder
3. Click "Load Project"

#### Via Command Line:
```bash
python -m src.project_loader /path/to/your/project
```

### API Endpoints

- `GET /` - Main dashboard
- `POST /load-project` - Load a project folder
- `GET /api/projects` - List all loaded projects
- `GET /api/incidents` - List all incidents

### Supported File Types

- **JSON files**: Configuration files, data files
- **YAML files**: Configuration files, metadata
- **CSV files**: Data files, spreadsheets
- **Text files**: Documentation, logs, README files

## Example

Try loading the sample project:
```bash
python -m src.project_loader examples/sample_project
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
# or
python -m unittest discover tests/
```

## Configuration

Edit `config.ini` to customize:
- Database settings
- File size limits
- Supported file formats
- Server configuration

## Development

The system is built with:
- **Flask**: Web framework
- **SQLite**: Database for incident storage
- **Python 3.8+**: Programming language
- **HTML/CSS/JavaScript**: Frontend
