"""
Configuration management for the Intelligent Hydro-DTM system.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator as field_validator
    except ImportError:
        # Fallback for older pydantic versions
        from pydantic import BaseSettings, Field
        field_validator = None


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "Intelligent Hydro-DTM System"
    app_version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = True
    
    # Database settings
    database_url: str = Field(
        default="postgresql://hydro_user:hydro_pass@localhost:5432/hydro_dtm",
        env="DATABASE_URL"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    redis_password: Optional[str] = None
    
    # File storage settings
    data_storage_path: Path = Field(
        default=Path("./data"),
        env="DATA_STORAGE_PATH"
    )
    temp_storage_path: Path = Field(
        default=Path("./temp"),
        env="TEMP_STORAGE_PATH"
    )
    max_file_size_mb: int = 1000  # Maximum file size in MB
    
    # AI/ML model settings
    models_path: Path = Field(
        default=Path("./models"),
        env="MODELS_PATH"
    )
    default_device: str = "cpu"  # "cpu", "cuda", "mps"
    model_cache_size: int = 3  # Number of models to keep in memory
    
    # Point cloud processing settings
    max_points_per_chunk: int = 1_000_000
    default_point_cloud_crs: str = "EPSG:4326"
    supported_point_cloud_formats: List[str] = ["las", "laz", "ply", "xyz"]
    
    # DTM generation settings
    default_dtm_resolution: float = 1.0  # meters
    min_dtm_resolution: float = 0.1
    max_dtm_resolution: float = 10.0
    dtm_interpolation_method: str = "ai"  # "ai", "idw", "kriging", "tin"
    
    # Hydrological analysis settings
    flow_algorithm: str = "d8"  # "d8", "dinf", "mfd"
    min_stream_threshold: int = 1000  # cells
    depression_filling_method: str = "planchon_darboux"
    
    # Waterlogging prediction settings
    waterlogging_model_type: str = "ensemble"  # "rf", "xgb", "ensemble"
    risk_classification_thresholds: Dict[str, float] = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8
    }
    
    # Drainage network optimization settings
    optimization_algorithm: str = "nsga2"
    max_optimization_generations: int = 100
    population_size: int = 50
    
    # Government integration settings
    svamitva_api_url: Optional[str] = None
    svamitva_api_key: Optional[str] = None
    panchayat_api_url: Optional[str] = None
    panchayat_api_key: Optional[str] = None
    revenue_api_url: Optional[str] = None
    revenue_api_key: Optional[str] = None
    
    # Security settings
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = 30
    
    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    enable_json_logging: bool = False
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Performance settings
    max_concurrent_processes: int = 4
    processing_timeout_minutes: int = 60
    memory_limit_gb: Optional[float] = None
    
    # Visualization settings
    enable_3d_visualization: bool = True
    max_visualization_points: int = 100_000
    tile_cache_size_mb: int = 500
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @field_validator("data_storage_path", "temp_storage_path", "models_path")
    @classmethod
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        if v == Environment.PRODUCTION:
            # Additional validation for production
            pass
        return v
    
    @field_validator("default_device")
    @classmethod
    def validate_device(cls, v):
        """Validate device setting."""
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v
    
    def get_database_url(self, async_driver: bool = False) -> str:
        """Get database URL with optional async driver."""
        if async_driver:
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        return self.database_url
    
    def get_storage_path(self, subdir: str = "") -> Path:
        """Get storage path with optional subdirectory."""
        path = self.data_storage_path
        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_temp_path(self, subdir: str = "") -> Path:
        """Get temporary storage path with optional subdirectory."""
        path = self.temp_storage_path
        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION


# Global settings instance
settings = Settings()


# Configuration for different components

class PointCloudConfig:
    """Point cloud processing configuration."""
    
    def __init__(self, settings: Settings):
        self.max_points_per_chunk = settings.max_points_per_chunk
        self.default_crs = settings.default_point_cloud_crs
        self.supported_formats = settings.supported_point_cloud_formats
        self.storage_path = settings.get_storage_path("point_clouds")
    
    def get_chunk_size(self, total_points: int) -> int:
        """Calculate optimal chunk size based on total points."""
        if total_points <= self.max_points_per_chunk:
            return total_points
        
        # Calculate number of chunks needed
        num_chunks = (total_points + self.max_points_per_chunk - 1) // self.max_points_per_chunk
        return total_points // num_chunks


class DTMConfig:
    """DTM generation configuration."""
    
    def __init__(self, settings: Settings):
        self.default_resolution = settings.default_dtm_resolution
        self.min_resolution = settings.min_dtm_resolution
        self.max_resolution = settings.max_dtm_resolution
        self.interpolation_method = settings.dtm_interpolation_method
        self.storage_path = settings.get_storage_path("dtms")
    
    def validate_resolution(self, resolution: float) -> float:
        """Validate and clamp resolution to valid range."""
        return max(self.min_resolution, min(self.max_resolution, resolution))


class HydroConfig:
    """Hydrological analysis configuration."""
    
    def __init__(self, settings: Settings):
        self.flow_algorithm = settings.flow_algorithm
        self.min_stream_threshold = settings.min_stream_threshold
        self.depression_filling_method = settings.depression_filling_method
        self.storage_path = settings.get_storage_path("hydrology")


class MLConfig:
    """Machine learning configuration."""
    
    def __init__(self, settings: Settings):
        self.models_path = settings.models_path
        self.default_device = settings.default_device
        self.cache_size = settings.model_cache_size
        self.waterlogging_model_type = settings.waterlogging_model_type
        self.risk_thresholds = settings.risk_classification_thresholds
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a specific model."""
        path = self.models_path / model_name
        path.mkdir(parents=True, exist_ok=True)
        return path


class OptimizationConfig:
    """Optimization algorithm configuration."""
    
    def __init__(self, settings: Settings):
        self.algorithm = settings.optimization_algorithm
        self.max_generations = settings.max_optimization_generations
        self.population_size = settings.population_size


class GovernmentConfig:
    """Government integration configuration."""
    
    def __init__(self, settings: Settings):
        self.svamitva_api_url = settings.svamitva_api_url
        self.svamitva_api_key = settings.svamitva_api_key
        self.panchayat_api_url = settings.panchayat_api_url
        self.panchayat_api_key = settings.panchayat_api_key
        self.revenue_api_url = settings.revenue_api_url
        self.revenue_api_key = settings.revenue_api_key
    
    def is_svamitva_enabled(self) -> bool:
        """Check if SVAMITVA integration is enabled."""
        return bool(self.svamitva_api_url and self.svamitva_api_key)
    
    def is_panchayat_enabled(self) -> bool:
        """Check if Panchayat integration is enabled."""
        return bool(self.panchayat_api_url and self.panchayat_api_key)
    
    def is_revenue_enabled(self) -> bool:
        """Check if Revenue integration is enabled."""
        return bool(self.revenue_api_url and self.revenue_api_key)


# Component configuration instances
point_cloud_config = PointCloudConfig(settings)
dtm_config = DTMConfig(settings)
hydro_config = HydroConfig(settings)
ml_config = MLConfig(settings)
optimization_config = OptimizationConfig(settings)
government_config = GovernmentConfig(settings)


def get_config() -> Settings:
    """Get the global settings instance."""
    return settings


def update_config(**kwargs) -> Settings:
    """Update configuration with new values."""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def load_config_from_file(config_file: str) -> Settings:
    """Load configuration from a specific file."""
    return Settings(_env_file=config_file)


# Environment-specific configurations
def get_development_config() -> Dict[str, Any]:
    """Get development-specific configuration overrides."""
    return {
        "debug": True,
        "log_level": LogLevel.DEBUG,
        "api_reload": True,
        "enable_metrics": False,
    }


def get_production_config() -> Dict[str, Any]:
    """Get production-specific configuration overrides."""
    return {
        "debug": False,
        "log_level": LogLevel.INFO,
        "api_reload": False,
        "enable_metrics": True,
        "enable_json_logging": True,
    }


def get_testing_config() -> Dict[str, Any]:
    """Get testing-specific configuration overrides."""
    return {
        "database_url": "sqlite:///./test.db",
        "redis_url": "redis://localhost:6379/1",
        "log_level": LogLevel.WARNING,
        "enable_metrics": False,
    }