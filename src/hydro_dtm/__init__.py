"""
Intelligent Hydro-DTM System
AI-powered geospatial platform for hydrological analysis and drainage network design.
"""

__version__ = "0.1.0"
__author__ = "Geo-AI Team"

from .models import (
    PointCloud,
    DTM,
    HydrologyResults,
    WaterloggingRisk,
    DrainageNetwork,
    StreamNetwork,
    WatershedData,
    ProcessingStatus,
    RiskLevel,
)

from .point_cloud_processor import (
    LASFileValidator,
    CoordinateSystemManager,
    LASFileReader,
    validate_las_file,
    read_las_file,
    get_las_metadata,
)

from .point_cloud_operations import (
    MemoryManager,
    CoordinateTransformer,
    NoiseFilter,
    PointCloudOperations,
    load_point_cloud,
    filter_point_cloud_noise,
    transform_point_cloud_crs,
)

# Import new modules
from .dtm_generator import DTMGenerator, generate_dtm_from_point_cloud
from .hydrology_analyzer import HydrologyAnalyzer, analyze_dtm_hydrology
from .waterlogging_predictor import WaterloggingPredictor, predict_waterlogging_risk
from .drainage_optimizer import DrainageOptimizer, optimize_drainage_network

__all__ = [
    # Core models
    "PointCloud",
    "DTM",
    "HydrologyResults",
    "WaterloggingRisk",
    "DrainageNetwork",
    "StreamNetwork",
    "WatershedData",
    "ProcessingStatus",
    "RiskLevel",
    
    # Point cloud processing
    "LASFileValidator",
    "CoordinateSystemManager", 
    "LASFileReader",
    "validate_las_file",
    "read_las_file",
    "get_las_metadata",
    
    # Point cloud operations
    "MemoryManager",
    "CoordinateTransformer",
    "NoiseFilter",
    "PointCloudOperations",
    "load_point_cloud",
    "filter_point_cloud_noise",
    "transform_point_cloud_crs",
    
    # Analysis modules
    "DTMGenerator",
    "generate_dtm_from_point_cloud",
    "HydrologyAnalyzer",
    "analyze_dtm_hydrology",
    "WaterloggingPredictor",
    "predict_waterlogging_risk",
    "DrainageOptimizer",
    "optimize_drainage_network",
]
