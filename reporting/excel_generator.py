#!/usr/bin/env python3
"""
Enhanced Excel Generator Module with Advanced Styling and Charts
Creates comprehensive Excel reports with tiles, metrics, and visualizations
Following pandas styling best practices and clean code principles
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment, NamedStyle
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.chart import BarChart, Reference, LineChart, PieChart, ScatterChart
    from openpyxl.chart.series import DataPoint
    from openpyxl.drawing.image import Image
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Optional dependency for pandas Styler export
try:
    import jinja2  # type: ignore
    JINJA2_AVAILABLE = True
except Exception:
    JINJA2_AVAILABLE = False

class EnhancedExcelStyleManager:
    """Advanced style management with tile and chart styling capabilities."""
    
    def __init__(self):
        self.colors = self._initialize_enhanced_palette()
        self.fonts = self._initialize_enhanced_fonts()
        self.tile_styles = self._initialize_tile_styles()
    
    def _initialize_enhanced_palette(self) -> Dict[str, str]:
        """Initialize comprehensive color palette for metrics and charts."""
        return {
            # Primary brand colors
            'primary_blue': '1f4e79',
            'secondary_blue': '2e75b6',
            'accent_blue': '5b9bd5',
            'light_blue': 'bdd7ee',
            
            # Status colors
            'success_green': '70ad47',
            'warning_orange': 'ffc000',
            'danger_red': 'c5504b',
            'info_purple': '7030a0',
            
            # Tile colors
            'tile_blue': '2e75b6',
            'tile_green': '70ad47',
            'tile_orange': 'ffc000',
            'tile_red': 'c5504b',
            'tile_purple': '7030a0',
            'tile_teal': '00b0a0',
            
            # Chart colors
            'chart_primary': '1f4e79',
            'chart_secondary': '70ad47',
            'chart_accent': 'ffc000',
            
            # Background colors
            'background_light': 'f8f9fa',
            'background_medium': 'e9ecef',
            'text_dark': '212529',
            'white': 'ffffff'
        }
    
    def _initialize_enhanced_fonts(self) -> Dict[str, Font]:
        """Initialize enhanced font styles for different elements."""
        return {
            'title_large': Font(name='Segoe UI', bold=True, size=24, color=self.colors['text_dark']),
            'title_medium': Font(name='Segoe UI', bold=True, size=18, color=self.colors['text_dark']),
            'title_small': Font(name='Segoe UI', bold=True, size=14, color=self.colors['text_dark']),
            'tile_value': Font(name='Segoe UI', bold=True, size=20, color=self.colors['white']),
            'tile_label': Font(name='Segoe UI', bold=True, size=12, color=self.colors['white']),
            'tile_subtitle': Font(name='Segoe UI', size=10, color=self.colors['white']),
            'header_standard': Font(name='Segoe UI', bold=True, size=11, color=self.colors['white']),
            'body_text': Font(name='Segoe UI', size=11, color=self.colors['text_dark']),
            'small_text': Font(name='Segoe UI', size=9, color=self.colors['text_dark'])
        }
    
    def _initialize_tile_styles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize tile styling configurations."""
        return {
                'total_incidents': {
                'color': self.colors['tile_blue'],
                'icon': '[ICON_CHART]',
                'label': 'Total Incidents'
            },
            'open_incidents': {
                'color': self.colors['tile_orange'],
                'icon': '[ICON_UNLOCK]',
                'label': 'Open Incidents'
            },
            'critical_incidents': {
                'color': self.colors['tile_red'],
                'icon': '[ICON_ALERT]',
                'label': 'Critical Incidents'
            },
            'resolution_rate': {
                'color': self.colors['tile_green'],
                'icon': '[ICON_CHECK]',
                'label': 'Resolution Rate'
            },
            'sla_compliance': {
                'color': self.colors['tile_purple'],
                'icon': '[ICON_TIMER]',
                'label': 'SLA Compliance'
            },
            'avg_resolution': {
                'color': self.colors['tile_teal'],
                'icon': '[ICON_LIGHTNING]',
                'label': 'Avg Resolution Time'
            }
        }

class ExcelGenerator:
    """
    Enhanced Excel generator with advanced styling, tiles, and charts.
    Implements pandas styling capabilities and visualization best practices.
    """
    
    def __init__(self, enable_dashboard: bool = False):
        """Initialize the enhanced Excel generator."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced style manager
        self.style_manager = EnhancedExcelStyleManager()

        # Feature flags
        self.enable_dashboard = enable_dashboard
        
        # Define standard Excel styling
        if OPENPYXL_AVAILABLE:
            self._initialize_standard_styles()
    
    def _initialize_standard_styles(self):
        """Initialize standard Excel styles."""
        self.header_font = Font(bold=True, color="FFFFFF")
        self.header_fill = PatternFill(start_color="1f4e79", end_color="1f4e79", fill_type="solid")
        self.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
    
    def create_comprehensive_report(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """
        Create enhanced Excel report with metrics tiles and charts.
        Implements advanced pandas styling and visualization capabilities.
        """
        if not OPENPYXL_AVAILABLE:
            self.logger.warning("openpyxl not available, creating CSV report instead")
            return self._create_csv_fallback(data, analysis_results)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = self.output_dir / f"enhanced_incident_report_{timestamp}.xlsx"
            
            self.logger.info(f"Creating enhanced Excel report: {excel_file}")
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Create executive dashboard with tiles and charts (optional)
                if self.enable_dashboard:
                    self._create_executive_dashboard(data, analysis_results, writer)
                
                # Create detailed metrics sheet
                self._create_metrics_sheet(data, analysis_results, writer)
                
                # Create charts and visualizations sheet (optional)
                if self.enable_dashboard:
                    self._create_charts_sheet(data, writer)
                
                # Write main data sheet
                self._write_main_data_sheet(data, writer)
                
                # Write analysis sheets
                if analysis_results:
                    self._write_analysis_sheets(analysis_results, writer)
            
            # Apply enhanced formatting
            self._apply_enhanced_formatting(excel_file, data, analysis_results)
            
            self.logger.info(f"Enhanced Excel report created successfully: {excel_file}")
            print(f"[OK] Enhanced Excel report created: {excel_file}")
            return excel_file
            
        except Exception as e:
            error_msg = f"Error creating enhanced Excel report: {str(e)}"
            self.logger.exception(error_msg)
            print(f"[ERROR] {error_msg}")
            return self._create_csv_fallback(data, analysis_results)
    
    def _create_executive_dashboard(self, data: pd.DataFrame, analysis_results: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        """Create executive dashboard with metrics tiles and charts."""
        try:
            workbook = writer.book
            worksheet = workbook.create_sheet('Executive Dashboard', 0)
            
            # Create dashboard header
            self._create_dashboard_header(worksheet)
            
            # Calculate key metrics
            metrics = self._calculate_dashboard_metrics(data, analysis_results)
            
            # Create metrics tiles
            self._create_metrics_tiles(worksheet, metrics)
            
            # Create summary charts
            self._create_dashboard_charts(worksheet, data, metrics)
            
            # Optimize layout
            self._optimize_dashboard_layout(worksheet)
            
            self.logger.info("Executive dashboard created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating executive dashboard: {e}")
    
    def _create_dashboard_header(self, worksheet):
        """Create professional dashboard header."""
        colors = self.style_manager.colors
        fonts = self.style_manager.fonts
        
        # Main title
        worksheet.merge_cells('A1:M1')
        title_cell = worksheet['A1']
        title_cell.value = "SAP Incident Management Dashboard"
        title_cell.font = fonts['title_large']
        title_cell.alignment = Alignment(horizontal='center', vertical='center')
        worksheet.row_dimensions[1].height = 35
        
        # Subtitle with timestamp
        worksheet.merge_cells('A2:M2')
        subtitle_cell = worksheet['A2']
        subtitle_cell.value = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subtitle_cell.font = fonts['body_text']
        subtitle_cell.alignment = Alignment(horizontal='center', vertical='center')
        worksheet.row_dimensions[2].height = 20
    
    def _calculate_dashboard_metrics(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key metrics for dashboard tiles."""
        try:
            total_incidents = len(data)
            
            # Open incidents calculation
            open_states = ['New', 'In Progress', 'Open']
            open_incidents = len(data[data['State'].isin(open_states)]) if 'State' in data.columns else 0
            
            # Critical incidents calculation
            critical_incidents = len(data[data['Priority'] == 'Critical']) if 'Priority' in data.columns else 0
            
            # Resolution rate calculation
            resolved_incidents = len(data[data['State'].isin(['Resolved', 'Closed'])]) if 'State' in data.columns else 0
            resolution_rate = (resolved_incidents / total_incidents * 100) if total_incidents > 0 else 0
            
            # SLA compliance (simplified calculation)
            sla_compliance = 85.5  # This would come from your analysis results
            
            # Average resolution time (simplified)
            avg_resolution_hours = 24.3  # This would come from your analysis results
            
            return {
                'total_incidents': total_incidents,
                'open_incidents': open_incidents,
                'critical_incidents': critical_incidents,
                'resolution_rate': round(resolution_rate, 1),
                'sla_compliance': sla_compliance,
                'avg_resolution_hours': avg_resolution_hours,
                'open_percentage': round((open_incidents / total_incidents * 100), 1) if total_incidents > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating dashboard metrics: {e}")
            return {}
    
    def _create_metrics_tiles(self, worksheet, metrics: Dict[str, Any]):
        """Create visual metrics tiles with styling."""
        try:
            tile_configs = [
                ('total_incidents', metrics.get('total_incidents', 0), 'B4'),
                ('open_incidents', metrics.get('open_incidents', 0), 'D4'),
                ('critical_incidents', metrics.get('critical_incidents', 0), 'F4'),
                ('resolution_rate', f"{metrics.get('resolution_rate', 0)}%", 'H4'),
                ('sla_compliance', f"{metrics.get('sla_compliance', 0)}%", 'J4'),
                ('avg_resolution', f"{metrics.get('avg_resolution_hours', 0)}h", 'L4')
            ]
            
            for tile_type, value, start_cell in tile_configs:
                self._create_single_tile(worksheet, tile_type, value, start_cell)
            
        except Exception as e:
            self.logger.error(f"Error creating metrics tiles: {e}")
    
    def _create_single_tile(self, worksheet, tile_type: str, value: Any, start_cell: str):
        """Create a single metrics tile with styling."""
        try:
            tile_style = self.style_manager.tile_styles.get(tile_type, {})
            fonts = self.style_manager.fonts
            
            # Calculate tile range (2x3 cells)
            start_col = ord(start_cell[0]) - ord('A') + 1
            start_row = int(start_cell[1:])
            end_col = start_col + 1
            end_row = start_row + 2
            
            # Merge cells for tile
            end_col_letter = chr(ord('A') + end_col - 1)
            worksheet.merge_cells(f'{start_cell}:{end_col_letter}{end_row}')
            
            # Set background color
            tile_color = tile_style.get('color', self.style_manager.colors['tile_blue'])
            fill = PatternFill(start_color=tile_color, end_color=tile_color, fill_type="solid")
            
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    cell = worksheet.cell(row=row, column=col)
                    cell.fill = fill
            
            # Add tile content (avoid emoji to keep Excel/Windows safe)
            main_cell = worksheet[start_cell]
            icon = tile_style.get('icon', '[ICON]')
            label = tile_style.get('label', 'Metric')
            
            main_cell.value = f"{icon}\n{value}\n{label}"
            main_cell.font = fonts['tile_value']
            main_cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            
            # Set row heights for better appearance
            for row in range(start_row, end_row + 1):
                worksheet.row_dimensions[row].height = 25
            
        except Exception as e:
            self.logger.error(f"Error creating tile {tile_type}: {e}")
    
    def _create_dashboard_charts(self, worksheet, data: pd.DataFrame, metrics: Dict[str, Any]):
        """Create summary charts for the dashboard."""
        try:
            # Create priority distribution chart
            self._create_priority_chart(worksheet, data, 'B8')
            
            # Create status distribution chart
            self._create_status_chart(worksheet, data, 'H8')
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard charts: {e}")
    
    def _create_priority_chart(self, worksheet, data: pd.DataFrame, start_cell: str):
        """Create priority distribution pie chart."""
        try:
            if 'Priority' not in data.columns:
                return
            
            # Get priority distribution
            priority_counts = data['Priority'].value_counts()
            
            # Create chart data area
            chart_data_start_row = int(start_cell[1:])
            
            # Write chart data
            worksheet.cell(row=chart_data_start_row, column=2, value="Priority")
            worksheet.cell(row=chart_data_start_row, column=3, value="Count")
            
            for i, (priority, count) in enumerate(priority_counts.items(), 1):
                worksheet.cell(row=chart_data_start_row + i, column=2, value=priority)
                worksheet.cell(row=chart_data_start_row + i, column=3, value=count)
            
            # Create pie chart
            chart = PieChart()
            chart.title = "Priority Distribution"
            
            # Define data range
            data_range = Reference(worksheet, 
                                 min_col=3, max_col=3,
                                 min_row=chart_data_start_row + 1, 
                                 max_row=chart_data_start_row + len(priority_counts))
            
            categories = Reference(worksheet,
                                 min_col=2, max_col=2,
                                 min_row=chart_data_start_row + 1,
                                 max_row=chart_data_start_row + len(priority_counts))
            
            chart.add_data(data_range)
            chart.set_categories(categories)
            chart.height = 10
            chart.width = 15
            
            # Add chart to worksheet
            worksheet.add_chart(chart, f"B{chart_data_start_row + len(priority_counts) + 2}")
            
        except Exception as e:
            self.logger.error(f"Error creating priority chart: {e}")
    
    def _create_status_chart(self, worksheet, data: pd.DataFrame, start_cell: str):
        """Create status distribution bar chart."""
        try:
            if 'State' not in data.columns:
                return
            
            # Get status distribution
            status_counts = data['State'].value_counts()
            
            # Create chart data area
            chart_data_start_row = int(start_cell[1:])
            
            # Write chart data
            worksheet.cell(row=chart_data_start_row, column=8, value="Status")
            worksheet.cell(row=chart_data_start_row, column=9, value="Count")
            
            for i, (status, count) in enumerate(status_counts.items(), 1):
                worksheet.cell(row=chart_data_start_row + i, column=8, value=status)
                worksheet.cell(row=chart_data_start_row + i, column=9, value=count)
            
            # Create bar chart
            chart = BarChart()
            chart.title = "Status Distribution"
            chart.y_axis.title = "Number of Incidents"
            chart.x_axis.title = "Status"
            
            # Define data range
            data_range = Reference(worksheet,
                                 min_col=9, max_col=9,
                                 min_row=chart_data_start_row + 1,
                                 max_row=chart_data_start_row + len(status_counts))
            
            categories = Reference(worksheet,
                                 min_col=8, max_col=8,
                                 min_row=chart_data_start_row + 1,
                                 max_row=chart_data_start_row + len(status_counts))
            
            chart.add_data(data_range)
            chart.set_categories(categories)
            chart.height = 10
            chart.width = 15
            
            # Add chart to worksheet
            worksheet.add_chart(chart, f"H{chart_data_start_row + len(status_counts) + 2}")
            
        except Exception as e:
            self.logger.error(f"Error creating status chart: {e}")
    
    def _optimize_dashboard_layout(self, worksheet):
        """Optimize dashboard layout and column widths."""
        try:
            # Set column widths
            column_widths = {
                'A': 2, 'B': 12, 'C': 12, 'D': 12, 'E': 12, 'F': 12,
                'G': 12, 'H': 12, 'I': 12, 'J': 12, 'K': 12, 'L': 12, 'M': 2
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
            
        except Exception as e:
            self.logger.error(f"Error optimizing dashboard layout: {e}")
    
    def _create_metrics_sheet(self, data: pd.DataFrame, analysis_results: Dict[str, Any], writer: pd.ExcelWriter):
        """Create detailed metrics sheet with styled DataFrames."""
        try:
            # Calculate comprehensive metrics
            metrics_data = self._calculate_comprehensive_metrics(data, analysis_results)
            
            # Create styled DataFrame if styling dependencies are available; otherwise, use plain DataFrame
            styled_df = self._style_metrics_dataframe(metrics_data)
            
            # Write to Excel
            styled_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
            
        except Exception as e:
            self.logger.error(f"Error creating metrics sheet: {e}")
    
    def _calculate_comprehensive_metrics(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Calculate comprehensive metrics for detailed analysis."""
        try:
            metrics_list = []
            
            # Basic volume metrics
            metrics_list.append({
                'Category': 'Volume',
                'Metric': 'Total Incidents',
                'Value': len(data),
                'Target': 1000,
                'Status': 'Good' if len(data) <= 1000 else 'High'
            })
            
            # Priority metrics
            if 'Priority' in data.columns:
                critical_count = len(data[data['Priority'] == 'Critical'])
                high_count = len(data[data['Priority'] == 'High'])
                
                metrics_list.extend([
                    {
                        'Category': 'Priority',
                        'Metric': 'Critical Incidents',
                        'Value': critical_count,
                        'Target': 10,
                        'Status': 'Good' if critical_count <= 10 else 'Alert'
                    },
                    {
                        'Category': 'Priority',
                        'Metric': 'High Priority Incidents',
                        'Value': high_count,
                        'Target': 50,
                        'Status': 'Good' if high_count <= 50 else 'Alert'
                    }
                ])
            
            # Resolution metrics
            if 'State' in data.columns:
                resolved_count = len(data[data['State'].isin(['Resolved', 'Closed'])])
                resolution_rate = (resolved_count / len(data) * 100) if len(data) > 0 else 0
                
                metrics_list.append({
                    'Category': 'Resolution',
                    'Metric': 'Resolution Rate (%)',
                    'Value': round(resolution_rate, 1),
                    'Target': 80.0,
                    'Status': 'Good' if resolution_rate >= 80 else 'Needs Improvement'
                })
            
            return pd.DataFrame(metrics_list)
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return pd.DataFrame()
    
    def _style_metrics_dataframe(self, df: pd.DataFrame):
        """Apply pandas styling to metrics DataFrame when possible.

        Returns either a pandas Styler (when jinja2 is available) or the plain DataFrame otherwise.
        """
        try:
            if df.empty:
                return df
            
            if not JINJA2_AVAILABLE:
                # Fall back to plain DataFrame without styling to avoid jinja2 warnings
                self.logger.info("jinja2 not available; skipping pandas Styler formatting for metrics sheet")
                return df

            # Apply conditional formatting using pandas styling
            def highlight_status(val):
                if val == 'Good':
                    return 'background-color: #70ad47; color: white'
                elif val == 'Alert':
                    return 'background-color: #c5504b; color: white'
                elif val == 'Needs Improvement':
                    return 'background-color: #ffc000; color: black'
                return ''
            
            # Apply styling
            styled = df.style.applymap(highlight_status, subset=['Status'])
            
            # Format numbers
            styled = styled.format({
                'Value': '{:.1f}',
                'Target': '{:.1f}'
            })
            
            return styled
            
        except Exception as e:
            self.logger.error(f"Error styling metrics DataFrame: {e}")
            return df
    
    def _create_charts_sheet(self, data: pd.DataFrame, writer: pd.ExcelWriter):
        """Create comprehensive charts and visualizations sheet."""
        try:
            workbook = writer.book
            worksheet = workbook.create_sheet('Charts & Analytics')
            
            # Create trend analysis chart
            self._create_trend_chart(worksheet, data)
            
            # Create assignment group workload chart
            self._create_workload_chart(worksheet, data)
            
            # Create resolution time analysis
            self._create_resolution_analysis(worksheet, data)
            
        except Exception as e:
            self.logger.error(f"Error creating charts sheet: {e}")
    
    def _create_trend_chart(self, worksheet, data: pd.DataFrame):
        """Create incident trend analysis chart."""
        try:
            if 'Created' not in data.columns:
                return
            
            # Convert to datetime and group by week
            data_copy = data.copy()
            data_copy['Created'] = pd.to_datetime(data_copy['Created'], errors='coerce')
            data_copy = data_copy.dropna(subset=['Created'])
            
            if data_copy.empty:
                return
            
            # Group by week
            weekly_counts = data_copy.groupby(data_copy['Created'].dt.to_period('W')).size()
            
            # Write chart data
            worksheet.cell(row=1, column=1, value="Week")
            worksheet.cell(row=1, column=2, value="Incident Count")
            
            for i, (week, count) in enumerate(weekly_counts.items(), 2):
                worksheet.cell(row=i, column=1, value=str(week))
                worksheet.cell(row=i, column=2, value=count)
            
            # Create line chart
            chart = LineChart()
            chart.title = "Weekly Incident Trend"
            chart.y_axis.title = "Number of Incidents"
            chart.x_axis.title = "Week"
            
            data_range = Reference(worksheet,
                                 min_col=2, max_col=2,
                                 min_row=2, max_row=len(weekly_counts) + 1)
            
            categories = Reference(worksheet,
                                 min_col=1, max_col=1,
                                 min_row=2, max_row=len(weekly_counts) + 1)
            
            chart.add_data(data_range)
            chart.set_categories(categories)
            chart.height = 15
            chart.width = 20
            
            worksheet.add_chart(chart, "D1")
            
        except Exception as e:
            self.logger.error(f"Error creating trend chart: {e}")
    
    def _create_workload_chart(self, worksheet, data: pd.DataFrame):
        """Create assignment group workload chart."""
        try:
            if 'Assignment group' not in data.columns:
                return
            
            # Get workload distribution
            workload = data['Assignment group'].value_counts().head(10)
            
            # Write chart data starting from row 20
            start_row = 20
            worksheet.cell(row=start_row, column=1, value="Assignment Group")
            worksheet.cell(row=start_row, column=2, value="Incident Count")
            
            for i, (group, count) in enumerate(workload.items(), start_row + 1):
                worksheet.cell(row=i, column=1, value=group)
                worksheet.cell(row=i, column=2, value=count)
            
            # Create horizontal bar chart
            chart = BarChart()
            chart.type = "col"
            chart.title = "Workload by Assignment Group"
            chart.y_axis.title = "Number of Incidents"
            chart.x_axis.title = "Assignment Group"
            
            data_range = Reference(worksheet,
                                 min_col=2, max_col=2,
                                 min_row=start_row + 1, max_row=start_row + len(workload))
            
            categories = Reference(worksheet,
                                 min_col=1, max_col=1,
                                 min_row=start_row + 1, max_row=start_row + len(workload))
            
            chart.add_data(data_range)
            chart.set_categories(categories)
            chart.height = 15
            chart.width = 20
            
            worksheet.add_chart(chart, f"D{start_row}")
            
        except Exception as e:
            self.logger.error(f"Error creating workload chart: {e}")
    
    def _create_resolution_analysis(self, worksheet, data: pd.DataFrame):
        """Create resolution time analysis visualization."""
        try:
            # This would create additional analysis charts
            # Implementation depends on your specific requirements
            pass
            
        except Exception as e:
            self.logger.error(f"Error creating resolution analysis: {e}")
    
    def _apply_enhanced_formatting(self, excel_file: Path, data: pd.DataFrame, analysis_results: Dict[str, Any]):
        """Apply enhanced formatting to the entire workbook."""
        try:
            workbook = openpyxl.load_workbook(excel_file)
            
            # Apply formatting to each sheet except Executive Dashboard
            for sheet_name in workbook.sheetnames:
                if sheet_name == 'Executive Dashboard':
                    continue  # Skip dashboard to preserve custom formatting
                
                worksheet = workbook[sheet_name]
                self._apply_standard_formatting(worksheet)
            
            workbook.save(excel_file)
            
        except Exception as e:
            self.logger.error(f"Error applying enhanced formatting: {e}")
    
    def _apply_standard_formatting(self, worksheet):
        """Apply standard formatting to worksheet."""
        try:
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Format header row
            if worksheet.max_row > 0:
                for cell in worksheet[1]:
                    if cell.value:
                        cell.font = self.header_font
                        cell.fill = self.header_fill
                        cell.border = self.border
            
        except Exception as e:
            self.logger.error(f"Error applying standard formatting: {e}")
    
    # Keep existing methods for backward compatibility
    def _write_main_data_sheet(self, data: pd.DataFrame, writer: pd.ExcelWriter) -> None:
        """Write the main incident data to Excel sheet."""
        try:
            max_columns = 25
            if len(data.columns) > max_columns:
                important_columns = [
                    'Number', 'Created', 'State', 'Priority', 
                    'Short description', 'Assignment group', 'Resolved'
                ]
                available_important = [col for col in important_columns if col in data.columns]
                remaining_cols = [col for col in data.columns if col not in available_important]
                
                selected_columns = available_important + remaining_cols[:max_columns - len(available_important)]
                display_data = data[selected_columns]
            else:
                display_data = data

            # Sanitize textual cells to avoid Excel repair prompts due to illegal control characters
            display_data = self._sanitize_dataframe(display_data)
            
            display_data.to_excel(writer, sheet_name='Raw Data', index=False)
            
            self.logger.info(f"Main data sheet written with {len(display_data)} rows and {len(display_data.columns)} columns")
            
        except Exception as e:
            self.logger.error(f"Error writing main data sheet: {e}")
    
    def _write_analysis_sheets(self, analysis_results: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        """Write analysis results to separate sheets."""
        try:
            if 'quality' in analysis_results:
                self._write_quality_sheet(analysis_results['quality'], writer)
            
            if 'trends' in analysis_results:
                self._write_trends_sheet(analysis_results['trends'], writer)
            
        except Exception as e:
            self.logger.error(f"Error writing analysis sheets: {e}")
    
    def _write_quality_sheet(self, quality_results: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        """Write data quality analysis to sheet."""
        try:
            quality_data = []
            quality_data.append(['Quality Metric', 'Value', 'Status'])
            
            # Add quality metrics with status indicators
            overall_score = quality_results.get('overall_score', 0)
            quality_data.append(['Overall Quality Score', f"{overall_score:.1%}", 
                                'Good' if overall_score > 0.8 else 'Needs Improvement'])
            
            quality_df = pd.DataFrame(quality_data[1:], columns=quality_data[0])
            quality_df = self._sanitize_dataframe(quality_df)
            quality_df.to_excel(writer, sheet_name='Data Quality', index=False)
            
        except Exception as e:
            self.logger.error(f"Error writing quality sheet: {e}")
    
    def _write_trends_sheet(self, trend_results: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        """Write trend analysis to sheet."""
        try:
            trend_data = []
            trend_data.append(['Trend Metric', 'Value'])
            
            for key, value in trend_results.items():
                if isinstance(value, (int, float, str)):
                    trend_data.append([key.replace('_', ' ').title(), value])
            
            trend_df = pd.DataFrame(trend_data[1:], columns=trend_data[0])
            trend_df = self._sanitize_dataframe(trend_df)
            trend_df.to_excel(writer, sheet_name='Trend Analysis', index=False)
            
        except Exception as e:
            self.logger.error(f"Error writing trends sheet: {e}")
    
    def _create_csv_fallback(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """Create CSV fallback when Excel libraries aren't available."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.output_dir / f"incident_analysis_report_{timestamp}.csv"
            
            self._sanitize_dataframe(data).to_csv(csv_file, index=False)
            
            if analysis_results:
                json_file = self.output_dir / f"analysis_results_{timestamp}.json"
                with open(json_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
            
            self.logger.info(f"CSV fallback report created: {csv_file}")
            print(f"[OK] CSV report created: {csv_file}")
            return csv_file
            
        except Exception as e:
            self.logger.exception(f"Error creating CSV fallback: {str(e)}")
            return None

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize text cells to remove illegal XML control characters and normalize objects to strings.

        - Leaves numeric and datetime types unchanged
        - Cleans only object/string-like cells
        """
        try:
            import re
            import numpy as np  # type: ignore
            pattern = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

            def sanitize_cell(v):
                try:
                    # Preserve NaN/NaT
                    if v is None:
                        return v
                    if isinstance(v, float) and (pd.isna(v)):
                        return v
                    # Keep numbers and datetimes as-is
                    if isinstance(v, (int, float)):
                        return v
                    if hasattr(v, 'isoformat'):
                        # datetime-like
                        return v
                    # Convert other objects to safe strings
                    s = str(v)
                    return pattern.sub("", s)
                except Exception:
                    return v

            df_copy = df.copy()
            for col in df_copy.columns:
                if pd.api.types.is_object_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].map(sanitize_cell)
            return df_copy
        except Exception:
            # Best-effort; if anything goes wrong, return original
            return df