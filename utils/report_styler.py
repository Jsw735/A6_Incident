#!/usr/bin/env python3
"""
Unified styling for all incident report sheets.
Implements consistent formatting using pandas styling capabilities.
Following Python best practices for clean, maintainable code.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

class ReportStyler:
    """
    Centralized styling for Executive, Weekly, and Daily reports.
    Uses pandas styling capabilities for consistent professional appearance.
    """
    
    def __init__(self):
        """Initialize with common styling configurations and logging."""
        self.logger = logging.getLogger(__name__)
        
        # Color scheme following professional dashboard standards
        self.color_schemes = {
            'header': '#2E86AB',           # Professional blue
            'excellent': '#28a745',        # Success green
            'good': '#17a2b8',            # Info blue
            'warning': '#ffc107',         # Warning yellow
            'danger': '#dc3545',          # Danger red
            'neutral': '#6c757d',         # Neutral gray
            'light_blue': '#e3f2fd',      # Light background
            'light_green': '#e8f5e9',     # Light success
            'light_yellow': '#fff3cd',    # Light warning
            'light_red': '#f8d7da'        # Light danger
        }
        
        # Font configurations for different elements
        self.font_styles = {
            'header': {
                'font-weight': 'bold', 
                'color': 'white', 
                'font-size': '12pt',
                'text-align': 'center'
            },
            'metric': {
                'font-weight': 'bold', 
                'font-size': '11pt',
                'text-align': 'center'
            },
            'data': {
                'font-size': '10pt',
                'text-align': 'left'
            },
            'caption': {
                'font-style': 'italic', 
                'font-size': '9pt',
                'color': '#666666'
            }
        }
        
        # Performance thresholds for different metrics
        self.thresholds = {
            'resolution_rate': {'excellent': 90, 'good': 75, 'warning': 60},
            'sla_compliance': {'excellent': 95, 'good': 85, 'warning': 70},
            'response_time': {'excellent': 2, 'good': 4, 'warning': 8},  # hours
            'escalation_rate': {'excellent': 5, 'good': 10, 'warning': 20}  # percentage
        }
    
    def style_executive_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply executive-level styling with high-level KPI focus.
        Emphasizes strategic metrics and performance indicators.
        """
        try:
            styled_df = df.style
            
            # Apply executive-specific conditional formatting
            styled_df = styled_df.apply(self._highlight_executive_metrics, axis=1)
            
            # Format percentage and numeric columns
            styled_df = self._apply_executive_formatting(styled_df, df)
            
            # Apply executive table styling
            styled_df = styled_df.set_table_styles(self._get_executive_table_styles())
            
            # Add executive caption
            styled_df = styled_df.set_caption(
                "Executive Dashboard - Strategic Performance Overview"
            )
            
            return styled_df
            
        except Exception as e:
            self.logger.error(f"Executive styling error: {str(e)}")
            return df
    
    def style_weekly_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weekly analysis styling with detailed trend visualization.
        Includes trend indicators and workstream performance metrics.
        """
        try:
            styled_df = df.style
            
            # Apply weekly-specific highlighting
            styled_df = styled_df.apply(self._highlight_weekly_metrics, axis=1)
            
            # Add trend bars for volume metrics
            if 'Total_Incidents' in df.columns:
                styled_df = styled_df.bar(
                    subset=['Total_Incidents'], 
                    color=self.color_schemes['good'],
                    width=60
                )
            
            # Add gradient for performance metrics
            if 'Resolution_Rate' in df.columns:
                styled_df = styled_df.background_gradient(
                    subset=['Resolution_Rate'], 
                    cmap='RdYlGn',
                    vmin=0, vmax=100
                )
            
            # Format weekly-specific columns
            styled_df = self._apply_weekly_formatting(styled_df, df)
            
            # Apply weekly table styling
            styled_df = styled_df.set_table_styles(self._get_weekly_table_styles())
            
            # Add weekly caption
            styled_df = styled_df.set_caption(
                "Weekly Analysis - Detailed Trends and Performance Metrics"
            )
            
            return styled_df
            
        except Exception as e:
            self.logger.error(f"Weekly styling error: {str(e)}")
            return df
    
    def style_daily_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply daily operations styling with action-focused formatting.
        Emphasizes immediate actions and operational status.
        """
        try:
            styled_df = df.style
            
            # Apply daily-specific highlighting
            styled_df = styled_df.apply(self._highlight_daily_metrics, axis=1)
            
            # Highlight urgent items
            if 'Priority' in df.columns:
                styled_df = styled_df.apply(self._highlight_priority_items, axis=1)
            
            # Format daily-specific columns
            styled_df = self._apply_daily_formatting(styled_df, df)
            
            # Apply daily table styling
            styled_df = styled_df.set_table_styles(self._get_daily_table_styles())
            
            # Add daily caption
            styled_df = styled_df.set_caption(
                "Daily Operations - Current Status and Immediate Actions"
            )
            
            return styled_df
            
        except Exception as e:
            self.logger.error(f"Daily styling error: {str(e)}")
            return df
    
    def style_workstream_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply workstream analysis styling with performance comparison.
        Includes relative performance indicators and trend analysis.
        """
        try:
            styled_df = df.style
            
            # Apply workstream performance highlighting
            styled_df = styled_df.apply(self._highlight_workstream_performance, axis=1)
            
            # Add performance score gradient
            if 'Performance_Score' in df.columns:
                styled_df = styled_df.background_gradient(
                    subset=['Performance_Score'],
                    cmap='RdYlGn',
                    vmin=0, vmax=100
                )
            
            # Format workstream columns
            styled_df = self._apply_workstream_formatting(styled_df, df)
            
            # Apply workstream table styling
            styled_df = styled_df.set_table_styles(self._get_workstream_table_styles())
            
            return styled_df
            
        except Exception as e:
            self.logger.error(f"Workstream styling error: {str(e)}")
            return df
    
    def _highlight_executive_metrics(self, row):
        """Apply conditional formatting for executive-level metrics."""
        styles = [''] * len(row)
        
        # SLA Compliance highlighting
        if 'SLA_Compliance' in row.index:
            value = row['SLA_Compliance']
            color = self._get_performance_color('sla_compliance', value)
            styles[row.index.get_loc('SLA_Compliance')] = f'background-color: {color}; color: white; font-weight: bold'
        
        # Critical incidents highlighting
        if 'Critical_Incidents' in row.index:
            value = row['Critical_Incidents']
            if value > 10:
                styles[row.index.get_loc('Critical_Incidents')] = f'background-color: {self.color_schemes["danger"]}; color: white; font-weight: bold'
            elif value > 5:
                styles[row.index.get_loc('Critical_Incidents')] = f'background-color: {self.color_schemes["warning"]}; font-weight: bold'
        
        return styles
    
    def _highlight_weekly_metrics(self, row):
        """Apply conditional formatting for weekly analysis metrics."""
        styles = [''] * len(row)
        
        # Resolution rate highlighting
        if 'Resolution_Rate' in row.index:
            value = row['Resolution_Rate']
            color = self._get_performance_color('resolution_rate', value)
            styles[row.index.get_loc('Resolution_Rate')] = f'background-color: {color}; color: white; font-weight: bold'
        
        # Trend direction highlighting
        if 'Trend_Direction' in row.index:
            trend = row['Trend_Direction']
            if trend == 'Improving':
                styles[row.index.get_loc('Trend_Direction')] = f'background-color: {self.color_schemes["excellent"]}; color: white'
            elif trend == 'Declining':
                styles[row.index.get_loc('Trend_Direction')] = f'background-color: {self.color_schemes["danger"]}; color: white'
        
        return styles
    
    def _highlight_daily_metrics(self, row):
        """Apply conditional formatting for daily operations metrics."""
        styles = [''] * len(row)
        
        # Overdue incidents highlighting
        if 'Overdue_Count' in row.index:
            value = row['Overdue_Count']
            if value > 0:
                styles[row.index.get_loc('Overdue_Count')] = f'background-color: {self.color_schemes["danger"]}; color: white; font-weight: bold'
        
        # New incidents highlighting
        if 'New_Today' in row.index:
            value = row['New_Today']
            if value > 20:
                styles[row.index.get_loc('New_Today')] = f'background-color: {self.color_schemes["warning"]}; font-weight: bold'
        
        return styles
    
    def _highlight_priority_items(self, row):
        """Highlight priority items in daily reports."""
        styles = [''] * len(row)
        
        if 'Priority' in row.index:
            priority = row['Priority']
            if priority == 'Critical':
                styles = [f'background-color: {self.color_schemes["light_red"]}'] * len(row)
            elif priority == 'High':
                styles = [f'background-color: {self.color_schemes["light_yellow"]}'] * len(row)
        
        return styles
    
    def _highlight_workstream_performance(self, row):
        """Highlight workstream performance metrics."""
        styles = [''] * len(row)
        
        if 'Performance_Score' in row.index:
            score = row['Performance_Score']
            color = self._get_performance_color_light(score)
            styles[row.index.get_loc('Performance_Score')] = f'background-color: {color}; font-weight: bold'
        
        return styles
    
    def _get_performance_color(self, metric_type: str, value: float) -> str:
        """Get performance color based on metric type and value."""
        thresholds = self.thresholds.get(metric_type, {})
        
        if value >= thresholds.get('excellent', 90):
            return self.color_schemes['excellent']
        elif value >= thresholds.get('good', 75):
            return self.color_schemes['good']
        elif value >= thresholds.get('warning', 60):
            return self.color_schemes['warning']
        else:
            return self.color_schemes['danger']
    
    def _get_performance_color_light(self, value: float) -> str:
        """Get light performance color for subtle highlighting."""
        if value >= 85:
            return self.color_schemes['light_green']
        elif value >= 70:
            return self.color_schemes['light_blue']
        elif value >= 55:
            return self.color_schemes['light_yellow']
        else:
            return self.color_schemes['light_red']
    
    def _apply_executive_formatting(self, styled_df, df: pd.DataFrame):
        """Apply executive-specific number formatting."""
        format_dict = {}
        
        for col in df.columns:
            if 'rate' in col.lower() or 'compliance' in col.lower():
                format_dict[col] = '{:.1f}%'
            elif 'time' in col.lower():
                format_dict[col] = '{:.1f}h'
            elif 'count' in col.lower() or 'total' in col.lower():
                format_dict[col] = '{:,.0f}'
        
        return styled_df.format(format_dict)
    
    def _apply_weekly_formatting(self, styled_df, df: pd.DataFrame):
        """Apply weekly-specific number formatting."""
        format_dict = {}
        
        for col in df.columns:
            if 'rate' in col.lower() or 'percentage' in col.lower():
                format_dict[col] = '{:.1f}%'
            elif 'average' in col.lower():
                format_dict[col] = '{:.1f}'
            elif 'count' in col.lower() or 'total' in col.lower():
                format_dict[col] = '{:,.0f}'
        
        return styled_df.format(format_dict)
    
    def _apply_daily_formatting(self, styled_df, df: pd.DataFrame):
        """Apply daily-specific number formatting."""
        format_dict = {}
        
        for col in df.columns:
            if 'time' in col.lower():
                format_dict[col] = '{:.0f}m'
            elif 'count' in col.lower():
                format_dict[col] = '{:,.0f}'
        
        return styled_df.format(format_dict)
    
    def _apply_workstream_formatting(self, styled_df, df: pd.DataFrame):
        """Apply workstream-specific number formatting."""
        format_dict = {}
        
        for col in df.columns:
            if 'score' in col.lower() or 'rate' in col.lower():
                format_dict[col] = '{:.1f}%'
            elif 'average' in col.lower():
                format_dict[col] = '{:.1f}'
        
        return styled_df.format(format_dict)
    
    def _get_executive_table_styles(self) -> List[Dict]:
        """Get table styles for executive reports."""
        return [
            {'selector': 'thead th', 'props': [
                ('background-color', self.color_schemes['header']),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '2px solid #ddd'),
                ('padding', '12px'),
                ('font-size', '12pt')
            ]},
            {'selector': 'tbody td', 'props': [
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '10px'),
                ('font-size', '11pt')
            ]},
            {'selector': 'table', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '15px auto'),
                ('width', '100%'),
                ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
            ]},
            {'selector': 'caption', 'props': [
                ('font-size', '14pt'),
                ('font-weight', 'bold'),
                ('margin-bottom', '10px'),
                ('color', self.color_schemes['header'])
            ]}
        ]
    
    def _get_weekly_table_styles(self) -> List[Dict]:
        """Get table styles for weekly reports."""
        return [
            {'selector': 'thead th', 'props': [
                ('background-color', self.color_schemes['good']),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '10px')
            ]},
            {'selector': 'tbody td', 'props': [
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '8px')
            ]},
            {'selector': 'table', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '10px auto'),
                ('width', '100%')
            ]}
        ]
    
    def _get_daily_table_styles(self) -> List[Dict]:
        """Get table styles for daily reports."""
        return [
            {'selector': 'thead th', 'props': [
                ('background-color', self.color_schemes['warning']),
                ('color', 'black'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '8px')
            ]},
            {'selector': 'tbody td', 'props': [
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '6px')
            ]},
            {'selector': 'table', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '10px auto'),
                ('width', '100%')
            ]}
        ]
    
    def _get_workstream_table_styles(self) -> List[Dict]:
        """Get table styles for workstream reports."""
        return [
            {'selector': 'thead th', 'props': [
                ('background-color', self.color_schemes['neutral']),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '8px')
            ]},
            {'selector': 'tbody td', 'props': [
                ('text-align', 'center'),
                ('border', '1px solid #ddd'),
                ('padding', '6px')
            ]},
            {'selector': 'table', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '10px auto'),
                ('width', '100%')
            ]}
        ]