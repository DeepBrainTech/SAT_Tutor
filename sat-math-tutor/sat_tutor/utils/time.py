"""
Time Utility Module
Provides timestamp generation and formatting functionality
"""

from datetime import datetime


def generate_session_id() -> str:
    """
    Generates a session ID.
    Format: YYYYMMDD_HHMMSS
    
    Returns:
        Session ID string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_timestamp() -> str:
    """
    Gets the current timestamp.
    
    Returns:
        ISO formatted timestamp
    """
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """
    Formats a duration in seconds into a readable string.
    
    Args:
        seconds: Number of seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_readable_timestamp() -> str:
    """
    Gets a human-readable timestamp.
    
    Returns:
        Formatted time string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")