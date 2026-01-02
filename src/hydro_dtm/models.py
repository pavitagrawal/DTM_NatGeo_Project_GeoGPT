"""
Core data models for the Intelligent Hydro-DTM system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point


class RiskLevel(Enum):
    """Waterlogging risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingStatus(Enum):
    """Processing status for various operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PointCloudMetadata:
    """Metadata for point cloud datasets."""
    filename: str
    file_size: int
    point_count: int
    coordinate_system: str
    bounds: List[float]  # [minx, miny, minz, maxx, maxy, maxz]
    creation_date: datetime
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointCloud:
    """Point cloud data structure with AI classification support."""
    points: np.ndarray  # Nx3 array of XYZ coordinates
    colors: Optional[np.ndarray] = None  # RGB values
    intensity: Optional[np.ndarray] = None  # LiDAR intensity values
    return_number: Optional[np.ndarray] = None  # Multiple return information
    classifications: Optional[np.ndarray] = None  # AI-generated classifications
    confidence_scores: Optional[np.ndarray] = None  # Classification confidence
    metadata: Optional[PointCloudMetadata] = None
    
    def __len__(self) -> int:
        """Return number of points."""
        return len(self.points)
    
    def get_ground_points(self) -> 'PointCloud':
        """Extract ground-classified points."""
        if self.classifications is None:
            return self
        
        ground_mask = self.classifications == 2  # LAS ground classification
        return PointCloud(
            points=self.points[ground_mask],
            colors=self.colors[ground_mask] if self.colors is not None else None,
            intensity=self.intensity[ground_mask] if self.intensity is not None else None,
            return_number=self.return_number[ground_mask] if self.return_number is not None else None,
            classifications=self.classifications[ground_mask],
            confidence_scores=self.confidence_scores[ground_mask] if self.confidence_scores is not None else None,
            metadata=self.metadata
        )


@dataclass
class DTM:
    """Digital Terrain Model with metadata."""
    elevation_grid: np.ndarray  # 2D elevation array
    resolution: float  # Grid cell size in meters
    bounds: List[float]  # [minx, miny, maxx, maxy]
    coordinate_system: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> tuple:
        """Return grid shape."""
        return self.elevation_grid.shape
    
    @property
    def extent(self) -> tuple:
        """Return extent for plotting."""
        return (self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3])


@dataclass
class StreamNetwork:
    """Stream network data structure."""
    streams: List[Dict[str, Any]]  # List of stream features
    total_length: float
    stream_density: float  # km/km²
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatershedData:
    """Individual watershed information."""
    id: int
    outlet_point: Point
    area: float  # m²
    perimeter: float  # m
    stream_length: float  # m
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HydrologyResults:
    """Complete hydrological analysis results."""
    flow_direction: np.ndarray
    flow_accumulation: np.ndarray
    stream_network: StreamNetwork
    watersheds: List[WatershedData]
    slope: np.ndarray
    aspect: np.ndarray
    topographic_wetness_index: np.ndarray
    filled_dem: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaterloggingRisk:
    """Waterlogging risk assessment results."""
    risk_grid: np.ndarray  # Risk classification grid (0-3)
    probability_grid: np.ndarray  # Risk probability (0-1)
    duration_grid: np.ndarray  # Expected duration in hours
    risk_statistics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_high_risk_areas(self) -> np.ndarray:
        """Get mask of high and critical risk areas."""
        return self.risk_grid >= 2


@dataclass
class DrainageSegment:
    """Individual drainage segment."""
    start_point: tuple  # (x, y)
    end_point: tuple  # (x, y)
    diameter: float  # meters
    depth: float  # meters
    slope: float  # gradient
    length: float  # meters
    cost: float  # USD
    capacity: float  # m³/s


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization results."""
    total_cost: float
    coverage_percentage: float
    environmental_impact: float
    hydraulic_efficiency: float
    maintenance_cost: float


@dataclass
class DrainageNetwork:
    """Complete drainage network design."""
    segments: List[DrainageSegment]
    total_length: float  # meters
    total_cost: float  # USD
    coverage_area: float  # percentage
    hydraulic_capacity: float  # m³/s
    objectives: OptimizationObjectives
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_geopandas(self) -> 'gpd.GeoDataFrame':
        """Convert to GeoPandas DataFrame."""
        import geopandas as gpd
        
        geometries = []
        properties = []
        
        for i, segment in enumerate(self.segments):
            line = LineString([segment.start_point, segment.end_point])
            geometries.append(line)
            
            properties.append({
                'segment_id': i,
                'diameter': segment.diameter,
                'depth': segment.depth,
                'slope': segment.slope,
                'length': segment.length,
                'cost': segment.cost,
                'capacity': segment.capacity
            })
        
        return gpd.GeoDataFrame(properties, geometry=geometries)


@dataclass
class AIModelResults:
    """AI model prediction results."""
    predictions: np.ndarray
    confidence_scores: np.ndarray
    model_name: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundClassificationResults:
    """Ground classification results from AI models."""
    classifications: np.ndarray  # Point classifications
    confidence_scores: np.ndarray  # Classification confidence
    class_names: List[str]  # Class name mapping
    accuracy_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingReport:
    """Comprehensive processing report."""
    processing_id: str
    start_time: datetime
    end_time: datetime
    input_data: Dict[str, Any]
    results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def processing_time(self) -> float:
        """Return processing time in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success(self) -> bool:
        """Return True if processing was successful."""
        return len(self.errors) == 0


@dataclass
class ValidationResults:
    """Data validation results."""
    is_valid: bool
    validation_score: float  # 0-1
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Data quality assessment metrics."""
    completeness: float  # 0-1
    accuracy: float  # 0-1
    consistency: float  # 0-1
    timeliness: float  # 0-1
    overall_score: float  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


# Type aliases for convenience
Coordinates = tuple[float, float]
BoundingBox = tuple[float, float, float, float]  # minx, miny, maxx, maxy
ElevationGrid = np.ndarray
ClassificationGrid = np.ndarray