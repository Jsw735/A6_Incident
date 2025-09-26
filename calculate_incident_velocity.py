def calculate_incident_velocity(df: pd.DataFrame) -> float:
    """
    Calculate incidents per day.
    
    Formula: total_incidents / date_range_in_days
    
    Test Cases:
    - 10 incidents over 5 days = 2.0 incidents/day
    - 1 incident over 1 day = 1.0 incidents/day
    - Empty data = 0.0 incidents/day
    """
    # Implementation with validation