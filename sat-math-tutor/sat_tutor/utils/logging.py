"""
Logging Module
Provides unified logging functionality
"""

import os
from datetime import datetime
from typing import Optional, TextIO


class Logger:
    """Simple logger class"""
    
    def __init__(self, log_file: Optional[str] = None, console: bool = True):
        """
        Initialize the logger
        
        Args:
            log_file: Path to the log file (optional)
            console: Whether to output to the console
        """
        self.log_file = log_file
        self.console = console
        self._file_handle: Optional[TextIO] = None
        
        if log_file:
            os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
            self._file_handle = open(log_file, 'a', encoding='utf-8')
    
    def __del__(self):
        """Close the file handle"""
        if self._file_handle:
            self._file_handle.close()
    
    def log(self, message: str, level: str = "info") -> None:
        """
        Record a log entry
        
        Args:
            message: The log message
            level: Log level (info, warning, error, debug)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_upper = level.upper()
        
        # Format the message
        formatted = f"[{timestamp}] [{level_upper}] {message}"
        
        # Write to file
        if self._file_handle:
            self._file_handle.write(formatted + "\n")
            self._file_handle.flush()
        
        # Output to console
        if self.console:
            try:
                from rich.console import Console
                console = Console()
                
                style_map = {
                    "info": "blue",
                    "warning": "yellow",
                    "error": "red",
                    "debug": "dim"
                }
                style = style_map.get(level, "white")
                console.print(formatted, style=style)
            except ImportError:
                print(formatted)
    
    def info(self, message: str) -> None:
        """Record info level log"""
        self.log(message, "info")
    
    def warning(self, message: str) -> None:
        """Record warning level log"""
        self.log(message, "warning")
    
    def error(self, message: str) -> None:
        """Record error level log"""
        self.log(message, "error")
    
    def debug(self, message: str) -> None:
        """Record debug level log"""
        self.log(message, "debug")
    
    def close(self) -> None:
        """Close the log file"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


def create_session_logger(session_dir: str) -> Logger:
    """
    Create a session-specific logger
    
    Args:
        session_dir: Directory for the session
    
    Returns:
        Logger instance
    """
    log_file = os.path.join(session_dir, "logs.txt")
    return Logger(log_file=log_file, console=True)