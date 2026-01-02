"""
Logging configuration for the Intelligent Hydro-DTM system.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
import structlog
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json_logging: bool = False,
    correlation_id: Optional[str] = None
) -> None:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_json_logging: Enable JSON structured logging
        correlation_id: Optional correlation ID for request tracking
    """
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add correlation ID if provided
    if correlation_id:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FUNC_NAME]
            )
        )
    
    if enable_json_logging:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def __init__(self, correlation_id: str):
        super().__init__()
        self.correlation_id = correlation_id
    
    def filter(self, record):
        record.correlation_id = self.correlation_id
        return True


class ProcessingLogger:
    """Specialized logger for processing operations with progress tracking."""
    
    def __init__(self, operation_name: str, total_items: Optional[int] = None):
        self.logger = get_logger(f"processing.{operation_name}")
        self.operation_name = operation_name
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = datetime.now()
        self.errors = []
        
        self.logger.info(
            "Processing started",
            operation=operation_name,
            total_items=total_items,
            start_time=self.start_time.isoformat()
        )
    
    def log_progress(self, items_processed: int = 1, message: Optional[str] = None):
        """Log processing progress."""
        self.processed_items += items_processed
        
        progress_data = {
            "operation": self.operation_name,
            "processed": self.processed_items,
            "total": self.total_items,
        }
        
        if self.total_items:
            progress_data["percentage"] = (self.processed_items / self.total_items) * 100
        
        if message:
            progress_data["message"] = message
        
        self.logger.info("Processing progress", **progress_data)
    
    def log_error(self, error: Exception, context: Optional[dict] = None):
        """Log processing error."""
        error_data = {
            "operation": self.operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processed_items": self.processed_items,
        }
        
        if context:
            error_data.update(context)
        
        self.errors.append(error_data)
        self.logger.error("Processing error", **error_data)
    
    def log_completion(self, success: bool = True):
        """Log processing completion."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        completion_data = {
            "operation": self.operation_name,
            "success": success,
            "processed_items": self.processed_items,
            "total_items": self.total_items,
            "duration_seconds": duration,
            "errors_count": len(self.errors),
            "end_time": end_time.isoformat()
        }
        
        if success:
            self.logger.info("Processing completed successfully", **completion_data)
        else:
            self.logger.error("Processing completed with errors", **completion_data)


# Pre-configured loggers for different components
def get_point_cloud_logger() -> structlog.BoundLogger:
    """Get logger for point cloud processing."""
    return get_logger("hydro_dtm.point_cloud")


def get_dtm_logger() -> structlog.BoundLogger:
    """Get logger for DTM generation."""
    return get_logger("hydro_dtm.dtm")


def get_hydro_logger() -> structlog.BoundLogger:
    """Get logger for hydrological analysis."""
    return get_logger("hydro_dtm.hydrology")


def get_ml_logger() -> structlog.BoundLogger:
    """Get logger for ML operations."""
    return get_logger("hydro_dtm.ml")


def get_api_logger() -> structlog.BoundLogger:
    """Get logger for API operations."""
    return get_logger("hydro_dtm.api")


def get_db_logger() -> structlog.BoundLogger:
    """Get logger for database operations."""
    return get_logger("hydro_dtm.database")


# Initialize default logging
setup_logging()