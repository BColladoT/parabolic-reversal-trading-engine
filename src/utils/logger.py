"""
Structured Logging Module
Provides JSON-formatted logging for observability and debugging.
"""
import logging
import os
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from src.utils.config import CONFIG


class TradingLogger:
    """Singleton logger for the trading engine."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance
    
    def _setup_logger(self):
        """Configure structured JSON logging."""
        self.logger = logging.getLogger("parabolic_reversal")
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(getattr(logging, level_name, logging.INFO))
        self.logger.handlers = []
        
        # Ensure logs directory exists
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            CONFIG.logging.file or 'logs/trading_engine.log',
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=CONFIG.logging.backup_count or 10
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, extra=kwargs)
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra=kwargs)


# Global logger instance
logger = TradingLogger()
