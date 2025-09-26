#!/usr/bin/env python3
"""
Daily Reporter for SAP Incident Analysis
Generates current state reporting and urgent item identification.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import os

class DailyReporter:
    """
    Generates daily status reports with current state analysis.
    Uses proper mathematical operations and Python programming fundamentals.
    """
    
    def __init__(self):
        """Initialize the Daily Reporter."""
        self.logger = logging.getLogger(__name__)
        self.daily_data = {}
    
    def generate_daily_report(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive daily incident report.
        
        Args:
            data: DataFrame containing incident data
            analysis_results: Dictionary containing analysis results from comprehensive analysis
            
        Returns:
            String path to generated report file
        """
        try:
            if data is None or data.empty:
                self.logger.warning("No data provided for daily report generation")
                return self._generate_empty_report()
            
            self.logger.info(f"Generating daily report for {len(data)} incidents")
            
            # Calculate today's metrics using mathematical operations
            todays_metrics = self._calculate_todays_metrics(data)
            
            # Identify urgent items requiring attention
            urgent_items = self._identify_urgent_items(data)
            
            # Calculate current capacity status
            capacity_status = self._calculate_current_capacity(data)
            
            # Generate aging analysis
            aging_analysis = self._analyze_incident_aging(data)
            
            # Create daily summary
            daily_summary = self._create_daily_summary(todays_metrics, urgent_items)
            
            # Store daily data
            self.daily_data = {
                'todays_metrics': todays_metrics,
                'urgent_items': urgent_items,
                'capacity_status': capacity_status,
                'aging_analysis': aging_analysis,
                'daily_summary': daily_summary,
                'generated_at': datetime.now().isoformat()
            }
            
            # Extract analysis components
            trend_results = analysis_results.get('trend_analysis', {})
            metrics_results = analysis_results.get('metrics', {})
            keyword_results = analysis_results.get('keyword_analysis', {})
            
            # Generate report content
            report_content = self._build_daily_report_content(
                data, trend_results, metrics_results, keyword_results
            )
            
            # Save report to file
            report_filename = self._save_daily_report(report_content)
            
            self.logger.info(f"Daily report generated successfully: {report_filename}")
            return report_filename
            
        except Exception as e:
            error_msg = f"Error generating daily report: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _build_daily_report_content(self, data: pd.DataFrame, trend_results: Dict, 
                                   metrics_results: Dict, keyword_results: Dict) -> str:
        """Build the content for the daily report following clean code principles."""
        try:
            report_date = datetime.now().strftime("%Y-%m-%d")
            
            # Report header
            content = f"""
SAP S/4HANA INCIDENT DAILY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Report Period: {report_date}
{'='*60}

EXECUTIVE SUMMARY
{'='*60}
Total Incidents Analyzed: {len(data)}
Analysis Period: {report_date}
Report Status: Complete

"""
            
            # Add daily metrics section
            content += self._format_daily_metrics_section()
            
            # Add urgent items section
            content += self._format_urgent_items_section()
            
            # Add capacity status section
            content += self._format_capacity_section()
            
            # Add aging analysis section
            content += self._format_aging_section()
            
            # Add comprehensive analysis results
            if metrics_results:
                content += self._format_metrics_section(metrics_results)
            
            if trend_results:
                content += self._format_trends_section(trend_results)
            
            if keyword_results:
                content += self._format_keywords_section(keyword_results)
            
            # Add incident breakdown
            content += self._format_incident_breakdown(data)
            
            # Add recommendations
            content += self._format_recommendations(data, {
                'trend_analysis': trend_results,
                'metrics': metrics_results,
                'keyword_analysis': keyword_results
            })
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error building report content: {str(e)}")
            return f"Error generating report content: {str(e)}"

    def _format_daily_metrics_section(self) -> str:
        """Format today's metrics section."""
        try:
            metrics = self.daily_data.get('todays_metrics', {})
            if 'error' in metrics:
                return f"DAILY METRICS\n{'='*60}\nError: {metrics['error']}\n\n"
            
            section = f"""
DAILY METRICS
{'='*60}
Incidents Created Today: {metrics.get('created_today', 0)}
Incidents Resolved Today: {metrics.get('resolved_today', 0)}
Current Open Incidents: {metrics.get('current_open', 0)}
Urgent Incidents: {metrics.get('urgent_count', 0)}
Hourly Creation Rate: {metrics.get('hourly_creation_rate', 0)}
Net Change Today: {metrics.get('net_change', 0)}
Report Time: {metrics.get('report_time', 'N/A')}

"""
            return section
        except Exception as e:
            return f"Error formatting daily metrics: {str(e)}\n\n"

    def _format_urgent_items_section(self) -> str:
        """Format urgent items section."""
        try:
            urgent_items = self.daily_data.get('urgent_items', [])
            
            section = f"""
URGENT ITEMS REQUIRING ATTENTION
{'='*60}
Total Urgent Items: {len(urgent_items)}

"""
            
            if urgent_items:
                section += "Top Priority Items:\n"
                for i, item in enumerate(urgent_items[:10], 1):  # Top 10
                    section += f"{i}. {item.get('number', 'N/A')} - {item.get('priority', 'N/A')} "
                    section += f"({item.get('age_hours', 0)}h) - {item.get('urgency_level', 'N/A')}\n"
                    section += f"   {item.get('short_description', 'No description')}\n"
            else:
                section += "No urgent items requiring immediate attention.\n"
            
            section += "\n"
            return section
        except Exception as e:
            return f"Error formatting urgent items: {str(e)}\n\n"

    def _format_capacity_section(self) -> str:
        """Format capacity status section."""
        try:
            capacity = self.daily_data.get('capacity_status', {})
            
            section = f"""
CAPACITY STATUS
{'='*60}
"""
            
            if 'error' in capacity:
                section += f"Error: {capacity['error']}\n\n"
            else:
                section += f"Total Open Incidents: {capacity.get('total_open_incidents', 0)}\n"
                section += f"Active Groups: {capacity.get('active_groups', 0)}\n"
                section += f"Average Group Workload: {capacity.get('average_group_workload', 0)}\n"
                section += f"Workload Imbalance Ratio: {capacity.get('workload_imbalance_ratio', 0)}\n"
                section += f"Alert Level: {capacity.get('alert_level', 'UNKNOWN')}\n"
                
                top_groups = capacity.get('top_loaded_groups', {})
                if top_groups:
                    section += "\nTop Loaded Groups:\n"
                    for group, count in list(top_groups.items())[:5]:
                        section += f"• {group}: {count} incidents\n"
            
            section += "\n"
            return section
        except Exception as e:
            return f"Error formatting capacity section: {str(e)}\n\n"

    def _format_aging_section(self) -> str:
        """Format aging analysis section."""
        try:
            aging = self.daily_data.get('aging_analysis', {})
            
            section = f"""
INCIDENT AGING ANALYSIS
{'='*60}
"""
            
            if 'error' in aging:
                section += f"Error: {aging['error']}\n\n"
            elif 'message' in aging:
                section += f"{aging['message']}\n\n"
            else:
                section += f"Total Open Incidents: {aging.get('total_open_incidents', 0)}\n"
                section += f"Average Age: {aging.get('average_age_hours', 0)} hours\n"
                section += f"Median Age: {aging.get('median_age_hours', 0)} hours\n"
                section += f"Oldest Incident: {aging.get('oldest_incident_hours', 0)} hours\n"
                
                dist = aging.get('age_distribution', {})
                if dist:
                    section += "\nAge Distribution:\n"
                    section += f"• Under 24h: {dist.get('under_24h', 0)} ({dist.get('under_24h_percent', 0)}%)\n"
                    section += f"• 24-72h: {dist.get('between_24_72h', 0)} ({dist.get('between_24_72h_percent', 0)}%)\n"
                    section += f"• Over 72h: {dist.get('over_72h', 0)} ({dist.get('over_72h_percent', 0)}%)\n"
            
            section += "\n"
            return section
        except Exception as e:
            return f"Error formatting aging section: {str(e)}\n\n"

    def _format_metrics_section(self, metrics_results: Dict) -> str:
        """Format the comprehensive metrics section."""
        try:
            section = f"""
COMPREHENSIVE METRICS ANALYSIS
{'='*60}
"""
            
            # Handle different metrics result formats following defensive programming
            if hasattr(metrics_results, 'to_dict'):
                metrics_dict = metrics_results.to_dict()
            elif hasattr(metrics_results, '__dict__'):
                metrics_dict = {attr: getattr(metrics_results, attr) 
                              for attr in dir(metrics_results) 
                              if not attr.startswith('_') and not callable(getattr(metrics_results, attr))}
            elif isinstance(metrics_results, dict):
                metrics_dict = metrics_results
            else:
                metrics_dict = {'metrics': str(metrics_results)}
            
            for key, value in metrics_dict.items():
                section += f"{key.replace('_', ' ').title()}: {value}\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            return f"Error formatting metrics section: {str(e)}\n\n"

    def _format_trends_section(self, trend_results: Dict) -> str:
        """Format the trends section."""
        try:
            section = f"""
TREND ANALYSIS
{'='*60}
"""
            
            if isinstance(trend_results, dict):
                for key, value in trend_results.items():
                    section += f"{key.replace('_', ' ').title()}: {value}\n"
            else:
                section += f"Trends: {str(trend_results)}\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            return f"Error formatting trends section: {str(e)}\n\n"

    def _format_keywords_section(self, keyword_results: Dict) -> str:
        """Format the keywords section."""
        try:
            section = f"""
KEYWORD ANALYSIS
{'='*60}
"""
            
            if isinstance(keyword_results, dict):
                summary = keyword_results.get('summary', {})
                if summary:
                    section += f"Analysis Status: {summary.get('analysis_status', 'Unknown')}\n"
                    section += f"Total Keywords Found: {summary.get('unique_keywords_found', 0)}\n"
                    section += f"Top Keyword: {summary.get('top_keyword', 'Not identified')}\n"
                    section += f"Sentiment Overview: {summary.get('sentiment_overview', 'Neutral')}\n"
                
                insights = keyword_results.get('actionable_insights', [])
                if insights:
                    section += "\nKey Insights:\n"
                    for insight in insights[:5]:
                        section += f"• {insight}\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            return f"Error formatting keywords section: {str(e)}\n\n"

    def _format_incident_breakdown(self, data: pd.DataFrame) -> str:
        """Format incident breakdown section following Pandas best practices."""
        try:
            section = f"""
INCIDENT BREAKDOWN
{'='*60}
"""
            
            section += f"Total Incidents: {len(data)}\n"
            
            # Category breakdown using proper Pandas operations
            if 'category' in data.columns:
                category_counts = data['category'].value_counts().head(5)
                section += "\nTop Categories:\n"
                for category, count in category_counts.items():
                    section += f"• {category}: {count} incidents\n"
            
            # Urgency breakdown
            if 'urgency' in data.columns:
                urgency_counts = data['urgency'].value_counts()
                section += "\nUrgency Distribution:\n"
                for urgency, count in urgency_counts.items():
                    section += f"• Urgency {urgency}: {count} incidents\n"
            
            # State breakdown
            if 'State' in data.columns:
                state_counts = data['State'].value_counts()
                section += "\nState Distribution:\n"
                for state, count in state_counts.items():
                    section += f"• {state}: {count} incidents\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            return f"Error formatting incident breakdown: {str(e)}\n\n"

    def _format_recommendations(self, data: pd.DataFrame, analysis_results: Dict) -> str:
        """Format recommendations section."""
        try:
            section = f"""
RECOMMENDATIONS & ACTION ITEMS
{'='*60}
"""
            
            recommendations = []
            
            # Add daily summary recommendations
            daily_summary = self.daily_data.get('daily_summary', {})
            for key, value in daily_summary.items():
                if 'HIGH ALERT' in value or 'CRITICAL' in value:
                    recommendations.append(f"URGENT: {value}")
                elif 'ELEVATED' in value or 'WARNING' in value:
                    recommendations.append(f"PRIORITY: {value}")
            
            # Add capacity recommendations
            capacity = self.daily_data.get('capacity_status', {})
            if capacity.get('alert_level') == 'HIGH':
                recommendations.append("IMMEDIATE: Address severe workload imbalance")
            
            # Add aging recommendations
            aging = self.daily_data.get('aging_analysis', {})
            if isinstance(aging, dict) and 'age_distribution' in aging:
                over_72h = aging['age_distribution'].get('over_72h', 0)
                if over_72h > 10:
                    recommendations.append(f"PRIORITY: Review {over_72h} incidents over 72 hours old")
            
            # Add keyword-based recommendations
            keyword_results = analysis_results.get('keyword_analysis', {})
            if isinstance(keyword_results, dict):
                insights = keyword_results.get('actionable_insights', [])
                recommendations.extend(insights[:3])
            
            if not recommendations:
                recommendations.append("Continue monitoring incident patterns for emerging trends")
            
            for i, rec in enumerate(recommendations, 1):
                section += f"{i}. {rec}\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            return f"Error formatting recommendations: {str(e)}\n\n"

    def _save_daily_report(self, content: str) -> str:
        """Save the daily report to file following clean code principles."""
        try:
            # Create reports directory if it doesn't exist
            reports_dir = "reports"
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{reports_dir}/daily_report_{timestamp}.txt"
            
            # Save content to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving daily report: {str(e)}")
            return f"Error saving report: {str(e)}"

    def _generate_empty_report(self) -> str:
        """Generate an empty report when no data is available."""
        try:
            content = f"""
SAP S/4HANA INCIDENT DAILY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*60}

No incident data available for analysis.
Please check data sources and try again.
"""
            
            return self._save_daily_report(content)
            
        except Exception as e:
            return f"Error generating empty report: {str(e)}"

    # Keep all your existing methods for backward compatibility
    def _calculate_todays_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate today's key metrics using addition and subtraction operations."""
        if 'Created' not in df.columns:
            return {'error': 'Created date column required'}
        
        try:
            # Convert to datetime for date calculations
            df['Created'] = pd.to_datetime(df['Created'])
            
            # Calculate today's date boundaries
            today = datetime.now().date()
            today_start = pd.Timestamp(today)
            today_end = today_start + timedelta(days=1)
            
            # Filter incidents created today (using comparison operations)
            todays_incidents = df[
                (df['Created'] >= today_start) & 
                (df['Created'] < today_end)
            ]
            
            # Calculate resolved today
            resolved_today = 0
            if 'Resolved' in df.columns:
                df['Resolved'] = pd.to_datetime(df['Resolved'])
                resolved_today_incidents = df[
                    (df['Resolved'] >= today_start) & 
                    (df['Resolved'] < today_end)
                ]
                resolved_today = len(resolved_today_incidents)
            
            # Calculate current open incidents (using subtraction)
            open_states = ['New', 'In Progress', 'Open']
            current_open = len(df[df['State'].isin(open_states)]) if 'State' in df.columns else 0
            
            # Calculate urgent count (using addition)
            urgent_count = 0
            if 'Priority' in df.columns:
                critical_incidents = df[df['Priority'] == 'Critical']
                high_incidents = df[df['Priority'] == 'High']
                urgent_count = len(critical_incidents) + len(high_incidents)
            
            # Calculate hourly metrics (using division)
            current_hour = datetime.now().hour
            if current_hour > 0:  # Avoid division by zero
                hourly_rate = len(todays_incidents) / current_hour
            else:
                hourly_rate = len(todays_incidents)
            
            return {
                'created_today': len(todays_incidents),
                'resolved_today': resolved_today,
                'current_open': current_open,
                'urgent_count': urgent_count,
                'hourly_creation_rate': round(hourly_rate, 2),
                'net_change': len(todays_incidents) - resolved_today,  # Using subtraction
                'report_time': datetime.now().strftime('%H:%M')
            }
            
        except Exception as e:
            return {'error': f'Error calculating daily metrics: {e}'}
    
    def _identify_urgent_items(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify urgent items requiring immediate attention."""
        urgent_items = []
        
        try:
            # Filter for urgent priorities
            if 'Priority' in df.columns:
                urgent_priorities = ['Critical', 'High']
                urgent_incidents = df[df['Priority'].isin(urgent_priorities)]
                
                # Filter for open states
                if 'State' in df.columns:
                    open_states = ['New', 'In Progress', 'Open']
                    urgent_open = urgent_incidents[urgent_incidents['State'].isin(open_states)]
                else:
                    urgent_open = urgent_incidents
                
                # Calculate age for each urgent item (using subtraction)
                if 'Created' in df.columns:
                    urgent_open['Created'] = pd.to_datetime(urgent_open['Created'])
                    current_time = pd.Timestamp.now()
                    
                    for _, incident in urgent_open.iterrows():
                        # Calculate age in hours (using mathematical operations)
                        age_delta = current_time - incident['Created']
                        age_hours = age_delta.total_seconds() / 3600  # Division operation
                        
                        # Determine urgency level (using comparison operations)
                        if incident['Priority'] == 'Critical' and age_hours > 4:
                            urgency_level = 'CRITICAL_OVERDUE'
                        elif incident['Priority'] == 'High' and age_hours > 24:
                            urgency_level = 'HIGH_OVERDUE'
                        elif incident['Priority'] == 'Critical':
                            urgency_level = 'CRITICAL_ACTIVE'
                        else:
                            urgency_level = 'HIGH_ACTIVE'
                        
                        urgent_items.append({
                            'number': incident.get('Number', 'N/A'),
                            'priority': incident.get('Priority', 'N/A'),
                            'state': incident.get('State', 'N/A'),
                            'assignment_group': incident.get('Assignment group', 'N/A'),
                            'age_hours': round(age_hours, 1),
                            'urgency_level': urgency_level,
                            'short_description': incident.get('Short description', 'N/A')[:50]
                        })
                
                # Sort by urgency level and age (using comparison operations)
                urgency_order = {
                    'CRITICAL_OVERDUE': 1,
                    'HIGH_OVERDUE': 2,
                    'CRITICAL_ACTIVE': 3,
                    'HIGH_ACTIVE': 4
                }
                
                urgent_items.sort(key=lambda x: (urgency_order.get(x['urgency_level'], 5), -x['age_hours']))
        
        except Exception as e:
            self.logger.error(f"Error identifying urgent items: {e}")
        
        return urgent_items
    
    def _calculate_current_capacity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate current capacity status using mathematical operations."""
        capacity_status = {}
        
        try:
            # Calculate workload by assignment group
            if 'Assignment group' in df.columns and 'State' in df.columns:
                open_states = ['New', 'In Progress', 'Open']
                open_incidents = df[df['State'].isin(open_states)]
                
                workload_by_group = open_incidents['Assignment group'].value_counts()
                
                # Calculate capacity metrics (using division and multiplication)
                total_open = len(open_incidents)
                if len(workload_by_group) > 0:
                    max_workload = workload_by_group.max()
                    min_workload = workload_by_group.min()
                    avg_workload = workload_by_group.mean()
                    
                    # Calculate imbalance ratio (using division)
                    imbalance_ratio = max_workload / avg_workload if avg_workload > 0 else 0
                    
                    capacity_status = {
                        'total_open_incidents': total_open,
                        'active_groups': len(workload_by_group),
                        'max_group_workload': int(max_workload),
                        'min_group_workload': int(min_workload),
                        'average_group_workload': round(avg_workload, 1),
                        'workload_imbalance_ratio': round(imbalance_ratio, 2),
                        'top_loaded_groups': workload_by_group.head(5).to_dict()
                    }
                    
                    # Capacity alert level (using comparison operations)
                    if imbalance_ratio > 3.0:
                        capacity_status['alert_level'] = 'HIGH'
                    elif imbalance_ratio > 2.0:
                        capacity_status['alert_level'] = 'MEDIUM'
                    else:
                        capacity_status['alert_level'] = 'LOW'
        
        except Exception as e:
            capacity_status['error'] = f'Error calculating capacity: {e}'
        
        return capacity_status
    
    def _analyze_incident_aging(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze incident aging using mathematical operations."""
        if 'Created' not in df.columns or 'State' not in df.columns:
            return {'error': 'Created date and State columns required'}
        
        try:
            # Filter for open incidents
            open_states = ['New', 'In Progress', 'Open']
            open_incidents = df[df['State'].isin(open_states)]
            
            if len(open_incidents) == 0:
                return {'message': 'No open incidents to analyze'}
            
            # Calculate age for each incident (using subtraction and division)
            open_incidents['Created'] = pd.to_datetime(open_incidents['Created'])
            current_time = pd.Timestamp.now()
            
            ages_hours = []
            for _, incident in open_incidents.iterrows():
                age_delta = current_time - incident['Created']
                age_hours = age_delta.total_seconds() / 3600
                ages_hours.append(age_hours)
            
            # Calculate aging statistics (using mathematical operations)
            ages_array = np.array(ages_hours)
            
            # Age buckets (using comparison operations)
            under_24h = sum(1 for age in ages_hours if age < 24)
            between_24_72h = sum(1 for age in ages_hours if 24 <= age < 72)
            over_72h = sum(1 for age in ages_hours if age >= 72)
            
            # Calculate percentages (using division and multiplication)
            total_incidents = len(ages_hours)
            under_24h_pct = (under_24h / total_incidents) * 100
            between_24_72h_pct = (between_24_72h / total_incidents) * 100
            over_72h_pct = (over_72h / total_incidents) * 100
            
            return {
                'total_open_incidents': total_incidents,
                'average_age_hours': round(ages_array.mean(), 1),
                'median_age_hours': round(np.median(ages_array), 1),
                'oldest_incident_hours': round(ages_array.max(), 1),
                'newest_incident_hours': round(ages_array.min(), 1),
                'age_distribution': {
                    'under_24h': under_24h,
                    'under_24h_percent': round(under_24h_pct, 1),
                    'between_24_72h': between_24_72h,
                    'between_24_72h_percent': round(between_24_72h_pct, 1),
                    'over_72h': over_72h,
                    'over_72h_percent': round(over_72h_pct, 1)
                }
            }
            
        except Exception as e:
            return {'error': f'Error analyzing aging: {e}'}
    
    def _create_daily_summary(self, metrics: Dict[str, Any], urgent_items: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create executive summary of daily status."""
        summary = {}
        
        if 'error' not in metrics:
            # Volume summary (using comparison operations)
            created_today = metrics.get('created_today', 0)
            if created_today > 20:
                summary['volume_status'] = f"High volume day: {created_today} incidents created"
            elif created_today > 10:
                summary['volume_status'] = f"Moderate volume: {created_today} incidents created"
            else:
                summary['volume_status'] = f"Low volume: {created_today} incidents created"
            
            # Urgent items summary
            urgent_count = len(urgent_items)
            if urgent_count > 10:
                summary['urgent_status'] = f"HIGH ALERT: {urgent_count} urgent items require attention"
            elif urgent_count > 5:
                summary['urgent_status'] = f"ELEVATED: {urgent_count} urgent items active"
            elif urgent_count > 0:
                summary['urgent_status'] = f"NORMAL: {urgent_count} urgent items being managed"
            else:
                summary['urgent_status'] = "GOOD: No urgent items requiring immediate attention"
            
            # Net change summary (using mathematical comparison)
            net_change = metrics.get('net_change', 0)
            if net_change > 5:
                summary['trend_status'] = f"Incident backlog growing by {net_change} today"
            elif net_change < -5:
                summary['trend_status'] = f"Incident backlog reducing by {abs(net_change)} today"
            else:
                summary['trend_status'] = "Incident volume stable today"
        
        return summary
    
    def get_daily_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for daily dashboard display."""
        if not self.daily_data:
            return {'error': 'No daily data available - run generate_daily_report first'}
        
        # Format for dashboard display
        dashboard_data = {
            'current_metrics': self.daily_data.get('todays_metrics', {}),
            'urgent_items': self.daily_data.get('urgent_items', [])[:10],  # Top 10 urgent
            'capacity_alerts': self._get_capacity_alerts(),
            'aging_summary': self._get_aging_summary(),
            'action_items': self._generate_action_items()
        }
        
        return dashboard_data
    
    def _get_capacity_alerts(self) -> List[str]:
        """Get capacity-related alerts."""
        alerts = []
        capacity = self.daily_data.get('capacity_status', {})
        
        if 'alert_level' in capacity:
            alert_level = capacity['alert_level']
            if alert_level == 'HIGH':
                alerts.append("CRITICAL: Severe workload imbalance detected")
            elif alert_level == 'MEDIUM':
                alerts.append("WARNING: Workload imbalance requires attention")
        
        if 'total_open_incidents' in capacity:
            total_open = capacity['total_open_incidents']
            if total_open > 100:
                alerts.append(f"HIGH VOLUME: {total_open} open incidents")
        
        return alerts if alerts else ["Capacity within normal parameters"]
    
    def _get_aging_summary(self) -> Dict[str, Any]:
        """Get simplified aging summary."""
        aging = self.daily_data.get('aging_analysis', {})
        
        if 'age_distribution' in aging:
            dist = aging['age_distribution']
            return {
                'over_72h_count': dist.get('over_72h', 0),
                'over_72h_percent': dist.get('over_72h_percent', 0),
                'average_age': aging.get('average_age_hours', 0)
            }
        
        return {}
    
    def _generate_action_items(self) -> List[str]:
        """Generate actionable items for today."""
        actions = []
        
        # Check urgent items
        urgent_items = self.daily_data.get('urgent_items', [])
        critical_overdue = [item for item in urgent_items if item.get('urgency_level') == 'CRITICAL_OVERDUE']
        
        if critical_overdue:
            actions.append(f"IMMEDIATE: Address {len(critical_overdue)} overdue critical incidents")
        
        # Check aging
        aging = self.daily_data.get('aging_analysis', {})
        if 'age_distribution' in aging:
            over_72h = aging['age_distribution'].get('over_72h', 0)
            if over_72h > 10:
                actions.append(f"PRIORITY: Review {over_72h} incidents over 72 hours old")
        
        # Check capacity
        capacity = self.daily_data.get('capacity_status', {})
        if capacity.get('alert_level') == 'HIGH':
            actions.append("URGENT: Rebalance workload distribution")
        
        return actions if actions else ["No immediate action items identified"]