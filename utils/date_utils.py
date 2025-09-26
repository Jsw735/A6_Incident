#!/usr/bin/env python3
"""
Date Utilities for SAP Incident Analysis
Provides standardized date operations and calculations.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Union, List, Tuple
import logging
from typing import Dict, List, Set, Any, Optional, Union, Tuple

class DateUtils:
    """
    Utility class for date operations following Python best practices.
    Implements proper error handling and mathematical operations.
    """
    
    def __init__(self):
        """Initialize the Date Utilities."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def parse_date(date_input: Union[str, pd.Timestamp, datetime]) -> pd.Timestamp:
        """
        Parse various date formats into standardized pandas Timestamp.
        
        Args:
            date_input: Date in various formats
            
        Returns:
            Standardized pandas Timestamp
        """
        try:
            if pd.isna(date_input) or date_input is None:
                return pd.NaT
            
            if isinstance(date_input, pd.Timestamp):
                return date_input
            
            if isinstance(date_input, datetime):
                return pd.Timestamp(date_input)
            
            # Parse string dates
            return pd.to_datetime(date_input)
            
        except Exception:
            return pd.NaT
    
    @staticmethod
    def calculate_age_hours(created_date: Union[str, pd.Timestamp], 
                           end_date: Union[str, pd.Timestamp] = None) -> float:
        """
        Calculate age in hours using subtraction and division operations.
        
        Args:
            created_date: When the incident was created
            end_date: End date for calculation (defaults to now)
            
        Returns:
            Age in hours as float
        """
        try:
            # Parse dates
            start = DateUtils.parse_date(created_date)
            end = DateUtils.parse_date(end_date) if end_date else pd.Timestamp.now()
            
            if pd.isna(start) or pd.isna(end):
                return 0.0
            
            # Calculate difference using subtraction operation
            time_diff = end - start
            
            # Convert to hours using division operation
            hours = time_diff.total_seconds() / 3600
            
            return max(0.0, hours)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    @staticmethod
    def get_week_boundaries(date_input: Union[str, pd.Timestamp] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get week start and end boundaries using addition and subtraction.
        
        Args:
            date_input: Reference date (defaults to today)
            
        Returns:
            Tuple of (week_start, week_end)
        """
        try:
            ref_date = DateUtils.parse_date(date_input) if date_input else pd.Timestamp.now()
            
            if pd.isna(ref_date):
                ref_date = pd.Timestamp.now()
            
            # Calculate week start (Monday) using subtraction
            days_since_monday = ref_date.weekday()
            week_start = ref_date - pd.Timedelta(days=days_since_monday)
            week_start = week_start.normalize()  # Set to midnight
            
            # Calculate week end using addition
            week_end = week_start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            return week_start, week_end
            
        except Exception:
            now = pd.Timestamp.now()
            return now.normalize(), now.normalize() + pd.Timedelta(days=1)
    
    @staticmethod
    def get_business_hours_between(start_date: Union[str, pd.Timestamp], 
                                  end_date: Union[str, pd.Timestamp],
                                  business_hours_per_day: int = 8) -> float:
        """
        Calculate business hours between two dates using mathematical operations.
        
        Args:
            start_date: Start date
            end_date: End date
            business_hours_per_day: Working hours per business day
            
        Returns:
            Business hours as float
        """
        try:
            start = DateUtils.parse_date(start_date)
            end = DateUtils.parse_date(end_date)
            
            if pd.isna(start) or pd.isna(end):
                return 0.0
            
            # Calculate total days using subtraction and division
            total_days = (end - start).total_seconds() / (24 * 3600)
            
            # Estimate business days (rough calculation: 5/7 of total days)
            business_days = total_days * (5 / 7)  # Using multiplication and division
            
            # Calculate business hours using multiplication
            business_hours = business_days * business_hours_per_day
            
            return max(0.0, business_hours)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def format_duration(hours: float) -> str:
        """
        Format duration in hours to human-readable string.
        
        Args:
            hours: Duration in hours
            
        Returns:
            Formatted duration string
        """
        try:
            if hours <= 0:
                return "0 hours"
            
            # Use division and modulus operations for time conversion
            days = int(hours // 24)  # Integer division
            remaining_hours = hours % 24  # Modulus operation
            
            if days > 0:
                return f"{days} days, {remaining_hours:.1f} hours"
            elif hours >= 1:
                return f"{hours:.1f} hours"
            else:
                # Convert to minutes using multiplication
                minutes = hours * 60
                return f"{minutes:.0f} minutes"
                
        except Exception:
            return "Unknown duration"
    
    @staticmethod
    def is_business_hours(date_input: Union[str, pd.Timestamp],
                         start_hour: int = 9, end_hour: int = 17) -> bool:
        """
        Check if date falls within business hours using comparison operations.
        
        Args:
            date_input: Date to check
            start_hour: Business day start hour
            end_hour: Business day end hour
            
        Returns:
            True if within business hours
        """
        try:
            date_obj = DateUtils.parse_date(date_input)
            
            if pd.isna(date_obj):
                return False
            
            # Check if weekday (Monday=0, Sunday=6) using comparison
            is_weekday = date_obj.weekday() < 5
            
            # Check if within business hours using comparison operations
            is_business_time = start_hour <= date_obj.hour < end_hour
            
            return is_weekday and is_business_time
            
        except Exception:
            return False
    
    @staticmethod
    def get_date_ranges(weeks_back: int = 4) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate list of weekly date ranges using subtraction and loops.
        
        Args:
            weeks_back: Number of weeks to generate
            
        Returns:
            List of (start_date, end_date) tuples
        """
        try:
            ranges = []
            current_date = pd.Timestamp.now()
            
            # Generate ranges using subtraction in a loop
            for week_offset in range(weeks_back):
                # Calculate week boundaries using subtraction
                week_end = current_date - pd.Timedelta(weeks=week_offset)
                week_start = week_end - pd.Timedelta(weeks=1)
                
                # Normalize to start/end of day
                week_start = week_start.normalize()
                week_end = week_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                ranges.append((week_start, week_end))
            
            return list(reversed(ranges))  # Return in chronological order
            
        except Exception:
            return []