"""
Structured logging utility for consistent terminal output.

Provides formatted messages with consistent prefixes, colors, and structure.
"""

import sys
from datetime import datetime
from typing import Optional

# ANSI color codes (can be disabled for non-terminal output)
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Message type colors
    INFO = '\033[94m'      # Blue
    SUCCESS = '\033[92m'   # Green
    ERROR = '\033[91m'     # Red
    WARNING = '\033[93m'   # Yellow
    DEBUG = '\033[90m'     # Gray
    
    # Section colors
    SECTION = '\033[95m'   # Magenta
    SUBSECTION = '\033[96m'  # Cyan


class Logger:
    """Structured logger for terminal output."""
    
    def __init__(self, module: str = "", use_colors: bool = True, use_timestamp: bool = False):
        """
        Initialize logger.
        
        Args:
            module: Module name prefix (e.g., "CRAWLER", "SOLVER")
            use_colors: Enable ANSI color codes
            use_timestamp: Include timestamps in messages
        """
        self.module = module.upper() if module else ""
        self.use_colors = use_colors and sys.stdout.isatty()
        self.use_timestamp = use_timestamp
    
    def _format_prefix(self, level: str, symbol: str) -> str:
        """Format message prefix with module, level, and symbol."""
        parts = []
        
        if self.use_timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")
            parts.append(f"[{timestamp}]")
        
        if self.module:
            parts.append(f"[{self.module}]")
        
        parts.append(f"[{level}]")
        parts.append(symbol)
        
        prefix = " ".join(parts)
        
        if self.use_colors:
            color_map = {
                "INFO": Colors.INFO,
                "SUCCESS": Colors.SUCCESS,
                "ERROR": Colors.ERROR,
                "WARNING": Colors.WARNING,
                "DEBUG": Colors.DEBUG,
            }
            color = color_map.get(level, Colors.RESET)
            return f"{color}{prefix}{Colors.RESET}"
        
        return prefix
    
    def _print(self, prefix: str, message: str, indent: int = 0):
        """Print formatted message with indentation."""
        indent_str = "  " * indent
        print(f"{prefix} {indent_str}{message}", flush=True)
    
    def info(self, message: str, indent: int = 0):
        """Print info message."""
        prefix = self._format_prefix("INFO", "→")
        self._print(prefix, message, indent)
    
    def success(self, message: str, indent: int = 0):
        """Print success message."""
        prefix = self._format_prefix("SUCCESS", "✓")
        self._print(prefix, message, indent)
    
    def error(self, message: str, indent: int = 0):
        """Print error message."""
        prefix = self._format_prefix("ERROR", "✗")
        self._print(prefix, message, indent)
    
    def warning(self, message: str, indent: int = 0):
        """Print warning message."""
        prefix = self._format_prefix("WARNING", "⚠")
        self._print(prefix, message, indent)
    
    def debug(self, message: str, indent: int = 0):
        """Print debug message."""
        prefix = self._format_prefix("DEBUG", "•")
        self._print(prefix, message, indent)
    
    def section(self, title: str, char: str = "=", width: int = 60):
        """Print section header."""
        if self.use_colors:
            print(f"\n{Colors.SECTION}{Colors.BOLD}{char * width}{Colors.RESET}")
            print(f"{Colors.SECTION}{Colors.BOLD}  {title}{Colors.RESET}")
            print(f"{Colors.SECTION}{Colors.BOLD}{char * width}{Colors.RESET}\n")
        else:
            print(f"\n{char * width}")
            print(f"  {title}")
            print(f"{char * width}\n")
    
    def subsection(self, title: str, char: str = "-", width: int = 50):
        """Print subsection header."""
        if self.use_colors:
            print(f"\n{Colors.SUBSECTION}{char * width}{Colors.RESET}")
            print(f"{Colors.SUBSECTION}  {title}{Colors.RESET}")
            print(f"{Colors.SUBSECTION}{char * width}{Colors.RESET}\n")
        else:
            print(f"\n{char * width}")
            print(f"  {title}")
            print(f"{char * width}\n")
    
    def divider(self, char: str = "-", width: int = 60):
        """Print a divider line."""
        if self.use_colors:
            print(f"{Colors.DEBUG}{char * width}{Colors.RESET}")
        else:
            print(char * width)


# Global logger instances for different modules
_crawler_logger = Logger("CRAWLER", use_timestamp=False)
_clicker_logger = Logger("CLICKER", use_timestamp=False)
_solver_logger = Logger("SOLVER", use_timestamp=False)
_database_logger = Logger("DATABASE", use_timestamp=False)
_api_logger = Logger("API", use_timestamp=False)
_evaluator_logger = Logger("EVALUATOR", use_timestamp=False)


def get_logger(module: str = "") -> Logger:
    """
    Get logger instance for a module.
    
    Args:
        module: Module name (e.g., "crawler", "solver")
    
    Returns:
        Logger instance
    """
    module_upper = module.upper()
    
    if module_upper == "CRAWLER":
        return _crawler_logger
    elif module_upper == "CLICKER":
        return _clicker_logger
    elif module_upper == "SOLVER":
        return _solver_logger
    elif module_upper == "DATABASE":
        return _database_logger
    elif module_upper == "API":
        return _api_logger
    elif module_upper == "EVALUATOR":
        return _evaluator_logger
    else:
        return Logger(module, use_timestamp=False)

