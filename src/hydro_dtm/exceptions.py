"""
Custom exceptions for the Intelligent Hydro-DTM system.
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    INPUT_VALIDATION = "input_validation"
    DATA_PROCESSING = "data_processing"
    AI_MODEL = "ai_model"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    SECURITY = "security"


class HydroDTMException(Exception):
    """Base exception for all Hydro-DTM system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.DATA_PROCESSING,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.suggestions = suggestions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "suggestions": self.suggestions
        }


# Input Validation Exceptions

class InvalidFileFormatError(HydroDTMException):
    """Raised when input file format is invalid or unsupported."""
    
    def __init__(self, filename: str, expected_formats: List[str], actual_format: Optional[str] = None):
        message = f"Invalid file format for '{filename}'"
        if actual_format:
            message += f". Got '{actual_format}'"
        message += f". Expected one of: {', '.join(expected_formats)}"
        
        super().__init__(
            message=message,
            error_code="INVALID_FILE_FORMAT",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INPUT_VALIDATION,
            context={
                "filename": filename,
                "expected_formats": expected_formats,
                "actual_format": actual_format
            },
            suggestions=[
                f"Ensure file is in one of the supported formats: {', '.join(expected_formats)}",
                "Check file extension and content match",
                "Verify file is not corrupted"
            ]
        )


class InvalidCoordinateSystemError(HydroDTMException):
    """Raised when coordinate system is invalid or unsupported."""
    
    def __init__(self, crs: str, supported_crs: Optional[List[str]] = None):
        message = f"Invalid or unsupported coordinate system: {crs}"
        
        context = {"provided_crs": crs}
        suggestions = [
            "Verify the coordinate system is correctly specified",
            "Check if coordinate system is supported by the system"
        ]
        
        if supported_crs:
            context["supported_crs"] = supported_crs
            suggestions.append(f"Use one of the supported CRS: {', '.join(supported_crs)}")
        
        super().__init__(
            message=message,
            error_code="INVALID_CRS",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INPUT_VALIDATION,
            context=context,
            suggestions=suggestions
        )


class InsufficientDataError(HydroDTMException):
    """Raised when input data is insufficient for processing."""
    
    def __init__(self, data_type: str, required_amount: str, actual_amount: str):
        super().__init__(
            message=f"Insufficient {data_type} data. Required: {required_amount}, Got: {actual_amount}",
            error_code="INSUFFICIENT_DATA",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INPUT_VALIDATION,
            context={
                "data_type": data_type,
                "required_amount": required_amount,
                "actual_amount": actual_amount
            },
            suggestions=[
                f"Provide more {data_type} data to meet minimum requirements",
                "Check data quality and completeness",
                "Consider using lower resolution settings if appropriate"
            ]
        )


# Data Processing Exceptions

class PointCloudProcessingError(HydroDTMException):
    """Raised when point cloud processing fails."""
    
    def __init__(self, operation: str, details: Optional[str] = None):
        message = f"Point cloud processing failed during {operation}"
        if details:
            message += f": {details}"
        
        super().__init__(
            message=message,
            error_code="POINT_CLOUD_PROCESSING_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_PROCESSING,
            context={"operation": operation, "details": details},
            suggestions=[
                "Check input point cloud data quality",
                "Verify sufficient memory is available",
                "Try processing with smaller data chunks"
            ]
        )


class DTMGenerationError(HydroDTMException):
    """Raised when DTM generation fails."""
    
    def __init__(self, method: str, reason: Optional[str] = None):
        message = f"DTM generation failed using {method} method"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="DTM_GENERATION_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_PROCESSING,
            context={"method": method, "reason": reason},
            suggestions=[
                "Check ground point quality and distribution",
                "Try alternative interpolation method",
                "Adjust resolution parameters",
                "Verify sufficient ground points are available"
            ]
        )


class HydrologicalAnalysisError(HydroDTMException):
    """Raised when hydrological analysis fails."""
    
    def __init__(self, analysis_type: str, error_details: Optional[str] = None):
        message = f"Hydrological analysis failed: {analysis_type}"
        if error_details:
            message += f" - {error_details}"
        
        super().__init__(
            message=message,
            error_code="HYDROLOGICAL_ANALYSIS_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_PROCESSING,
            context={"analysis_type": analysis_type, "error_details": error_details},
            suggestions=[
                "Check DTM quality and resolution",
                "Verify DTM covers the required area",
                "Check for data gaps or invalid values",
                "Try different algorithm parameters"
            ]
        )


class HydrologyAnalysisError(HydroDTMException):
    """Alias for HydrologicalAnalysisError for backward compatibility."""
    
    def __init__(self, analysis_type: str, error_details: Optional[str] = None):
        message = f"Hydrological analysis failed: {analysis_type}"
        if error_details:
            message += f" - {error_details}"
        
        super().__init__(
            message=message,
            error_code="HYDROLOGY_ANALYSIS_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_PROCESSING,
            context={"analysis_type": analysis_type, "error_details": error_details},
            suggestions=[
                "Check DTM quality and resolution",
                "Verify DTM covers the required area",
                "Check for data gaps or invalid values",
                "Try different algorithm parameters"
            ]
        )


class WaterloggingPredictionError(HydroDTMException):
    """Raised when waterlogging prediction fails."""
    
    def __init__(self, operation: str, error_details: Optional[str] = None):
        message = f"Waterlogging prediction failed: {operation}"
        if error_details:
            message += f" - {error_details}"
        
        super().__init__(
            message=message,
            error_code="WATERLOGGING_PREDICTION_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AI_MODEL,
            context={"operation": operation, "error_details": error_details},
            suggestions=[
                "Check input data quality",
                "Verify model is properly trained",
                "Check feature extraction process",
                "Try different model parameters"
            ]
        )


class DrainageOptimizationError(HydroDTMException):
    """Raised when drainage optimization fails."""
    
    def __init__(self, operation: str, error_details: Optional[str] = None):
        message = f"Drainage optimization failed: {operation}"
        if error_details:
            message += f" - {error_details}"
        
        super().__init__(
            message=message,
            error_code="DRAINAGE_OPTIMIZATION_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_PROCESSING,
            context={"operation": operation, "error_details": error_details},
            suggestions=[
                "Check optimization parameters",
                "Verify input data quality",
                "Try different algorithm settings",
                "Check constraint definitions"
            ]
        )


# AI/ML Exceptions

class ModelLoadError(HydroDTMException):
    """Raised when AI model loading fails."""
    
    def __init__(self, model_name: str, model_path: str, error_details: Optional[str] = None):
        message = f"Failed to load AI model '{model_name}' from {model_path}"
        if error_details:
            message += f": {error_details}"
        
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.AI_MODEL,
            context={
                "model_name": model_name,
                "model_path": model_path,
                "error_details": error_details
            },
            suggestions=[
                "Verify model file exists and is accessible",
                "Check model file integrity",
                "Ensure compatible model version",
                "Check available system memory"
            ]
        )


class ModelPredictionError(HydroDTMException):
    """Raised when AI model prediction fails."""
    
    def __init__(self, model_name: str, input_shape: Optional[tuple] = None, error_details: Optional[str] = None):
        message = f"AI model '{model_name}' prediction failed"
        if error_details:
            message += f": {error_details}"
        
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AI_MODEL,
            context={
                "model_name": model_name,
                "input_shape": input_shape,
                "error_details": error_details
            },
            suggestions=[
                "Check input data format and shape",
                "Verify input data is within expected ranges",
                "Ensure model is properly loaded",
                "Check available GPU/CPU resources"
            ]
        )


class ModelTrainingError(HydroDTMException):
    """Raised when model training fails."""
    
    def __init__(self, model_name: str, epoch: Optional[int] = None, error_details: Optional[str] = None):
        message = f"Model training failed for '{model_name}'"
        if epoch is not None:
            message += f" at epoch {epoch}"
        if error_details:
            message += f": {error_details}"
        
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AI_MODEL,
            context={
                "model_name": model_name,
                "epoch": epoch,
                "error_details": error_details
            },
            suggestions=[
                "Check training data quality and format",
                "Verify sufficient training data is available",
                "Adjust learning rate and other hyperparameters",
                "Check available GPU memory"
            ]
        )


# System Resource Exceptions

class MemoryExhaustionError(HydroDTMException):
    """Raised when system runs out of memory."""
    
    def __init__(self, operation: str, required_memory: Optional[str] = None, available_memory: Optional[str] = None):
        message = f"Memory exhaustion during {operation}"
        if required_memory and available_memory:
            message += f". Required: {required_memory}, Available: {available_memory}"
        
        super().__init__(
            message=message,
            error_code="MEMORY_EXHAUSTION",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM_RESOURCE,
            context={
                "operation": operation,
                "required_memory": required_memory,
                "available_memory": available_memory
            },
            suggestions=[
                "Process data in smaller chunks",
                "Increase available system memory",
                "Use memory-efficient algorithms",
                "Close other applications to free memory"
            ]
        )


class DiskSpaceError(HydroDTMException):
    """Raised when insufficient disk space is available."""
    
    def __init__(self, required_space: str, available_space: str, path: str):
        super().__init__(
            message=f"Insufficient disk space at {path}. Required: {required_space}, Available: {available_space}",
            error_code="INSUFFICIENT_DISK_SPACE",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM_RESOURCE,
            context={
                "required_space": required_space,
                "available_space": available_space,
                "path": path
            },
            suggestions=[
                "Free up disk space",
                "Use a different storage location",
                "Clean up temporary files",
                "Compress or archive old data"
            ]
        )


# External Service Exceptions

class DatabaseConnectionError(HydroDTMException):
    """Raised when database connection fails."""
    
    def __init__(self, database_url: str, error_details: Optional[str] = None):
        message = f"Failed to connect to database: {database_url}"
        if error_details:
            message += f" - {error_details}"
        
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.EXTERNAL_SERVICE,
            context={"database_url": database_url, "error_details": error_details},
            suggestions=[
                "Check database server is running",
                "Verify connection credentials",
                "Check network connectivity",
                "Verify database URL format"
            ]
        )


class GovernmentAPIError(HydroDTMException):
    """Raised when government API integration fails."""
    
    def __init__(self, api_name: str, status_code: Optional[int] = None, error_details: Optional[str] = None):
        message = f"Government API error: {api_name}"
        if status_code:
            message += f" (HTTP {status_code})"
        if error_details:
            message += f" - {error_details}"
        
        super().__init__(
            message=message,
            error_code="GOVERNMENT_API_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            context={
                "api_name": api_name,
                "status_code": status_code,
                "error_details": error_details
            },
            suggestions=[
                "Check API endpoint availability",
                "Verify authentication credentials",
                "Check API rate limits",
                "Retry request after delay"
            ]
        )


# Configuration Exceptions

class ConfigurationError(HydroDTMException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, issue: str):
        super().__init__(
            message=f"Configuration error for '{config_key}': {issue}",
            error_code="CONFIGURATION_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            context={"config_key": config_key, "issue": issue},
            suggestions=[
                f"Check configuration for '{config_key}'",
                "Verify configuration file format",
                "Ensure all required configuration values are set",
                "Check configuration file permissions"
            ]
        )


# Utility functions for error handling

def handle_exception(
    exception: Exception,
    logger,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> Optional[HydroDTMException]:
    """
    Handle and log exceptions with proper context.
    
    Args:
        exception: The exception to handle
        logger: Logger instance
        operation: Description of the operation that failed
        context: Additional context information
        reraise: Whether to reraise the exception
        
    Returns:
        HydroDTMException if not reraised
    """
    if isinstance(exception, HydroDTMException):
        hydro_exception = exception
    else:
        # Convert generic exception to HydroDTMException
        hydro_exception = HydroDTMException(
            message=f"Unexpected error during {operation}: {str(exception)}",
            error_code="UNEXPECTED_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_PROCESSING,
            context=context or {}
        )
    
    # Log the exception
    logger.error(
        "Exception occurred",
        operation=operation,
        **hydro_exception.to_dict()
    )
    
    if reraise:
        raise hydro_exception
    
    return hydro_exception


def create_error_response(exception: HydroDTMException) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.
    
    Args:
        exception: The HydroDTMException to convert
        
    Returns:
        Dictionary suitable for JSON API response
    """
    return {
        "error": True,
        "error_code": exception.error_code,
        "message": exception.message,
        "severity": exception.severity.value,
        "category": exception.category.value,
        "suggestions": exception.suggestions,
        "context": exception.context
    }