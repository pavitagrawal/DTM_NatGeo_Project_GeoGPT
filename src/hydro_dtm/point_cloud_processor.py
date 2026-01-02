"""
Point cloud processing and validation for LAS/LAZ files.
Optimized for drone survey data processing.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import laspy
from pyproj import CRS, Transformer
from datetime import datetime

from .models import PointCloud, PointCloudMetadata, ProcessingStatus
from .exceptions import (
    InvalidFileFormatError, InvalidCoordinateSystemError, 
    InsufficientDataError, PointCloudProcessingError
)
from .logging_config import get_point_cloud_logger, ProcessingLogger
from .config import point_cloud_config

logger = get_point_cloud_logger()


class LASFileValidator:
    """Validator for LAS/LAZ files with comprehensive format checking."""
    
    SUPPORTED_FORMATS = ['.las', '.laz']
    SUPPORTED_VERSIONS = ['1.2', '1.3', '1.4']
    
    def __init__(self):
        self.logger = logger
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate LAS/LAZ file format and extract basic information.
        
        Args:
            file_path: Path to the LAS/LAZ file
            
        Returns:
            Dictionary with validation results and file information
            
        Raises:
            InvalidFileFormatError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise InvalidFileFormatError(
                filename=str(file_path),
                expected_formats=self.SUPPORTED_FORMATS,
                actual_format=file_path.suffix
            )
        
        try:
            # Open and validate LAS file
            with laspy.open(file_path) as las_file:
                header = las_file.header
                
                # Validate LAS version
                version = f"{header.version_major}.{header.version_minor}"
                if version not in self.SUPPORTED_VERSIONS:
                    self.logger.warning(
                        "Unsupported LAS version",
                        version=version,
                        supported_versions=self.SUPPORTED_VERSIONS
                    )
                
                # Extract file information
                file_info = {
                    'valid': True,
                    'file_size': file_path.stat().st_size,
                    'point_count': header.point_count,
                    'version': version,
                    'point_data_format': header.point_data_format,
                    'bounds': {
                        'min_x': header.x_min,
                        'max_x': header.x_max,
                        'min_y': header.y_min,
                        'max_y': header.y_max,
                        'min_z': header.z_min,
                        'max_z': header.z_max
                    },
                    'scale': {
                        'x': header.x_scale,
                        'y': header.y_scale,
                        'z': header.z_scale
                    },
                    'offset': {
                        'x': header.x_offset,
                        'y': header.y_offset,
                        'z': header.z_offset
                    },
                    'creation_date': self._extract_creation_date(header),
                    'system_id': header.system_identifier.decode('utf-8').strip(),
                    'software': header.generating_software.decode('utf-8').strip()
                }
                
                # Validate point count
                if header.point_count == 0:
                    raise InsufficientDataError(
                        data_type="point cloud",
                        required_amount="at least 1 point",
                        actual_amount="0 points"
                    )
                
                # Check for reasonable bounds
                self._validate_bounds(file_info['bounds'])
                
                self.logger.info(
                    "LAS file validation successful",
                    filename=file_path.name,
                    point_count=header.point_count,
                    version=version
                )
                
                return file_info
                
        except laspy.LaspyException as e:
            raise InvalidFileFormatError(
                filename=str(file_path),
                expected_formats=self.SUPPORTED_FORMATS,
                actual_format="corrupted LAS/LAZ"
            ) from e
        except Exception as e:
            raise PointCloudProcessingError(
                operation="file validation",
                details=str(e)
            ) from e
    
    def _extract_creation_date(self, header) -> datetime:
        """Extract creation date from LAS header."""
        try:
            if header.creation_day_of_year > 0 and header.creation_year > 0:
                return datetime(header.creation_year, 1, 1) + \
                       datetime.timedelta(days=header.creation_day_of_year - 1)
            else:
                return datetime.now()
        except (ValueError, AttributeError):
            return datetime.now()
    
    def _validate_bounds(self, bounds: Dict[str, float]) -> None:
        """Validate spatial bounds are reasonable."""
        # Check for valid coordinate ranges
        x_range = bounds['max_x'] - bounds['min_x']
        y_range = bounds['max_y'] - bounds['min_y']
        z_range = bounds['max_z'] - bounds['min_z']
        
        if x_range <= 0 or y_range <= 0:
            raise InvalidFileFormatError(
                filename="current file",
                expected_formats=["valid spatial bounds"],
                actual_format=f"invalid bounds: x_range={x_range}, y_range={y_range}"
            )
        
        # Check for extremely large coordinates (likely projection issues)
        max_coord = max(abs(bounds['min_x']), abs(bounds['max_x']), 
                       abs(bounds['min_y']), abs(bounds['max_y']))
        
        if max_coord > 1e8:  # Coordinates larger than 100 million
            self.logger.warning(
                "Very large coordinates detected - check coordinate system",
                max_coordinate=max_coord
            )


class CoordinateSystemManager:
    """Manages coordinate system detection and validation."""
    
    # Common Indian coordinate systems
    INDIAN_CRS = {
        'EPSG:4326': 'WGS84 Geographic',
        'EPSG:32643': 'WGS84 UTM Zone 43N (India)',
        'EPSG:32644': 'WGS84 UTM Zone 44N (India)', 
        'EPSG:32645': 'WGS84 UTM Zone 45N (India)',
        'EPSG:7755': 'Kalianpur 1975 UTM Zone 43N',
        'EPSG:7756': 'Kalianpur 1975 UTM Zone 44N',
        'EPSG:7757': 'Kalianpur 1975 UTM Zone 45N'
    }
    
    def __init__(self):
        self.logger = logger
    
    def detect_coordinate_system(self, las_file_path: Path) -> str:
        """
        Detect coordinate system from LAS file.
        
        Args:
            las_file_path: Path to LAS file
            
        Returns:
            EPSG code as string
            
        Raises:
            InvalidCoordinateSystemError: If CRS cannot be determined
        """
        try:
            with laspy.open(las_file_path) as las_file:
                header = las_file.header
                
                # Try to get CRS from header
                if hasattr(header, 'crs') and header.crs is not None:
                    crs_code = f"EPSG:{header.crs.to_epsg()}"
                    self.logger.info("CRS detected from header", crs=crs_code)
                    return crs_code
                
                # Try to infer from coordinate ranges
                bounds = {
                    'min_x': header.x_min,
                    'max_x': header.x_max,
                    'min_y': header.y_min,
                    'max_y': header.y_max
                }
                
                inferred_crs = self._infer_crs_from_bounds(bounds)
                if inferred_crs:
                    self.logger.info("CRS inferred from bounds", crs=inferred_crs)
                    return inferred_crs
                
                # Default to WGS84 if cannot determine
                default_crs = "EPSG:4326"
                self.logger.warning(
                    "Could not determine CRS, using default",
                    default_crs=default_crs
                )
                return default_crs
                
        except Exception as e:
            raise InvalidCoordinateSystemError(
                crs="unknown",
                supported_crs=list(self.INDIAN_CRS.keys())
            ) from e
    
    def _infer_crs_from_bounds(self, bounds: Dict[str, float]) -> Optional[str]:
        """Infer CRS from coordinate bounds."""
        min_x, max_x = bounds['min_x'], bounds['max_x']
        min_y, max_y = bounds['min_y'], bounds['max_y']
        
        # Check if coordinates are in geographic range (lat/lon)
        if (-180 <= min_x <= 180 and -180 <= max_x <= 180 and
            -90 <= min_y <= 90 and -90 <= max_y <= 90):
            return "EPSG:4326"  # WGS84 Geographic
        
        # Check for Indian UTM zones based on coordinate ranges
        if 200000 <= min_x <= 800000 and 1000000 <= min_y <= 4000000:
            # Determine UTM zone based on longitude (approximate)
            center_x = (min_x + max_x) / 2
            if center_x < 500000:
                return "EPSG:32643"  # UTM Zone 43N
            elif center_x < 700000:
                return "EPSG:32644"  # UTM Zone 44N
            else:
                return "EPSG:32645"  # UTM Zone 45N
        
        return None
    
    def validate_crs(self, crs_code: str) -> bool:
        """Validate if CRS is supported."""
        try:
            crs = CRS.from_epsg(int(crs_code.split(':')[1]))
            return crs.is_valid
        except Exception:
            return False


class LASFileReader:
    """High-performance LAS/LAZ file reader with memory management."""
    
    def __init__(self):
        self.validator = LASFileValidator()
        self.crs_manager = CoordinateSystemManager()
        self.logger = logger
    
    def read_file(self, file_path: Union[str, Path], 
                  chunk_size: Optional[int] = None) -> PointCloud:
        """
        Read LAS/LAZ file and create PointCloud object.
        
        Args:
            file_path: Path to LAS/LAZ file
            chunk_size: Optional chunk size for large files
            
        Returns:
            PointCloud object with loaded data
            
        Raises:
            Various exceptions for file format, CRS, or processing errors
        """
        file_path = Path(file_path)
        
        # Validate file first
        file_info = self.validator.validate_file(file_path)
        
        # Detect coordinate system
        crs_code = self.crs_manager.detect_coordinate_system(file_path)
        
        # Create processing logger
        processing_logger = ProcessingLogger(
            operation_name=f"read_las_file_{file_path.name}",
            total_items=file_info['point_count']
        )
        
        try:
            # Determine if we need chunked reading
            use_chunking = (chunk_size is not None or 
                          file_info['point_count'] > point_cloud_config.max_points_per_chunk)
            
            if use_chunking:
                return self._read_file_chunked(file_path, file_info, crs_code, processing_logger)
            else:
                return self._read_file_complete(file_path, file_info, crs_code, processing_logger)
                
        except Exception as e:
            processing_logger.log_error(e, {"file_path": str(file_path)})
            processing_logger.log_completion(success=False)
            raise
    
    def _read_file_complete(self, file_path: Path, file_info: Dict[str, Any], 
                          crs_code: str, processing_logger: ProcessingLogger) -> PointCloud:
        """Read entire file into memory at once."""
        try:
            with laspy.open(file_path) as las_file:
                # Read all points
                points_data = las_file.read()
                
                # Extract coordinates
                points = np.column_stack([
                    points_data.x.astype(np.float64),
                    points_data.y.astype(np.float64),
                    points_data.z.astype(np.float64)
                ])
                
                # Extract optional attributes
                intensity = getattr(points_data, 'intensity', None)
                return_number = getattr(points_data, 'return_number', None)
                colors = None
                
                # Extract RGB if available
                if hasattr(points_data, 'red') and hasattr(points_data, 'green') and hasattr(points_data, 'blue'):
                    colors = np.column_stack([
                        points_data.red,
                        points_data.green,
                        points_data.blue
                    ])
                
                # Create metadata
                metadata = PointCloudMetadata(
                    filename=file_path.name,
                    file_size=file_info['file_size'],
                    point_count=file_info['point_count'],
                    coordinate_system=crs_code,
                    bounds=[
                        file_info['bounds']['min_x'], file_info['bounds']['min_y'], file_info['bounds']['min_z'],
                        file_info['bounds']['max_x'], file_info['bounds']['max_y'], file_info['bounds']['max_z']
                    ],
                    creation_date=file_info['creation_date'],
                    processing_status=ProcessingStatus.COMPLETED,
                    additional_info={
                        'las_version': file_info['version'],
                        'point_data_format': file_info['point_data_format'],
                        'system_id': file_info['system_id'],
                        'software': file_info['software']
                    }
                )
                
                processing_logger.log_progress(file_info['point_count'], "File read successfully")
                processing_logger.log_completion(success=True)
                
                return PointCloud(
                    points=points,
                    colors=colors,
                    intensity=intensity,
                    return_number=return_number,
                    metadata=metadata,
                    coordinate_system=crs_code
                )
                
        except Exception as e:
            raise PointCloudProcessingError(
                operation="complete file reading",
                details=str(e)
            ) from e
    
    def _read_file_chunked(self, file_path: Path, file_info: Dict[str, Any], 
                          crs_code: str, processing_logger: ProcessingLogger) -> PointCloud:
        """Read file in chunks for memory efficiency."""
        # For now, implement basic chunked reading
        # In production, this would use streaming readers
        self.logger.info("Chunked reading not fully implemented, falling back to complete read")
        return self._read_file_complete(file_path, file_info, crs_code, processing_logger)
    
    def extract_metadata_only(self, file_path: Union[str, Path]) -> PointCloudMetadata:
        """Extract only metadata without loading point data."""
        file_path = Path(file_path)
        file_info = self.validator.validate_file(file_path)
        crs_code = self.crs_manager.detect_coordinate_system(file_path)
        
        return PointCloudMetadata(
            filename=file_path.name,
            file_size=file_info['file_size'],
            point_count=file_info['point_count'],
            coordinate_system=crs_code,
            bounds=[
                file_info['bounds']['min_x'], file_info['bounds']['min_y'], file_info['bounds']['min_z'],
                file_info['bounds']['max_x'], file_info['bounds']['max_y'], file_info['bounds']['max_z']
            ],
            creation_date=file_info['creation_date'],
            processing_status=ProcessingStatus.PENDING,
            additional_info={
                'las_version': file_info['version'],
                'point_data_format': file_info['point_data_format'],
                'system_id': file_info['system_id'],
                'software': file_info['software']
            }
        )


# Convenience functions for easy usage
def validate_las_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate a LAS/LAZ file and return information."""
    validator = LASFileValidator()
    return validator.validate_file(file_path)


def read_las_file(file_path: Union[str, Path]) -> PointCloud:
    """Read a LAS/LAZ file and return PointCloud object."""
    reader = LASFileReader()
    return reader.read_file(file_path)


def get_las_metadata(file_path: Union[str, Path]) -> PointCloudMetadata:
    """Get metadata from LAS/LAZ file without loading points."""
    reader = LASFileReader()
    return reader.extract_metadata_only(file_path)