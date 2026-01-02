"""
Basic point cloud operations including transformations, filtering, and memory management.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
import gc
from pyproj import Transformer, CRS

from .models import PointCloud, PointCloudMetadata
from .exceptions import (
    PointCloudProcessingError, InvalidCoordinateSystemError, 
    MemoryExhaustionError
)
from .logging_config import get_point_cloud_logger, ProcessingLogger
from .config import point_cloud_config

logger = get_point_cloud_logger()


class MemoryManager:
    """Memory management utilities for large point clouds."""
    
    @staticmethod
    def estimate_memory_usage(point_count: int, attributes: List[str] = None) -> float:
        """
        Estimate memory usage in MB for a point cloud.
        
        Args:
            point_count: Number of points
            attributes: List of attribute names
            
        Returns:
            Estimated memory usage in MB
        """
        # Base memory for XYZ coordinates (3 * 8 bytes per point)
        base_memory = point_count * 3 * 8
        
        # Additional memory for attributes
        if attributes:
            for attr in attributes:
                if attr in ['intensity', 'return_number', 'classification']:
                    base_memory += point_count * 4  # 4 bytes per attribute
                elif attr == 'colors':
                    base_memory += point_count * 3 * 2  # RGB, 2 bytes each
        
        # Convert to MB and add 20% overhead
        return (base_memory / (1024 * 1024)) * 1.2
    
    @staticmethod
    def check_memory_availability(required_mb: float) -> bool:
        """Check if sufficient memory is available."""
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            return available_mb > required_mb * 1.5  # 50% safety margin
        except ImportError:
            logger.warning("psutil not available, cannot check memory")
            return True
    
    @staticmethod
    def optimize_memory_usage():
        """Force garbage collection to free memory."""
        gc.collect()


class CoordinateTransformer:
    """Handles coordinate system transformations for point clouds."""
    
    def __init__(self):
        self.logger = logger
        self._transformer_cache = {}
    
    def transform_coordinates(self, points: np.ndarray, 
                            source_crs: str, target_crs: str) -> np.ndarray:
        """
        Transform point coordinates between coordinate systems.
        
        Args:
            points: Nx3 array of XYZ coordinates
            source_crs: Source coordinate system (e.g., 'EPSG:4326')
            target_crs: Target coordinate system (e.g., 'EPSG:32643')
            
        Returns:
            Transformed coordinates as Nx3 array
            
        Raises:
            InvalidCoordinateSystemError: If CRS transformation fails
        """
        if source_crs == target_crs:
            return points.copy()
        
        try:
            # Get or create transformer
            transform_key = f"{source_crs}->{target_crs}"
            if transform_key not in self._transformer_cache:
                transformer = Transformer.from_crs(
                    source_crs, target_crs, always_xy=True
                )
                self._transformer_cache[transform_key] = transformer
            else:
                transformer = self._transformer_cache[transform_key]
            
            # Transform coordinates
            x_new, y_new = transformer.transform(points[:, 0], points[:, 1])
            
            # Create new coordinate array
            transformed_points = np.column_stack([
                x_new, y_new, points[:, 2]  # Keep Z unchanged
            ])
            
            self.logger.info(
                "Coordinate transformation completed",
                source_crs=source_crs,
                target_crs=target_crs,
                point_count=len(points)
            )
            
            return transformed_points
            
        except Exception as e:
            raise InvalidCoordinateSystemError(
                crs=f"{source_crs} -> {target_crs}",
                supported_crs=["EPSG:4326", "EPSG:32643", "EPSG:32644", "EPSG:32645"]
            ) from e
    
    def transform_point_cloud(self, point_cloud: PointCloud, 
                            target_crs: str) -> PointCloud:
        """
        Transform entire point cloud to new coordinate system.
        
        Args:
            point_cloud: Input point cloud
            target_crs: Target coordinate system
            
        Returns:
            New PointCloud with transformed coordinates
        """
        transformed_points = self.transform_coordinates(
            point_cloud.points, 
            point_cloud.coordinate_system, 
            target_crs
        )
        
        # Create new point cloud with transformed coordinates
        new_point_cloud = PointCloud(
            points=transformed_points,
            colors=point_cloud.colors,
            intensity=point_cloud.intensity,
            return_number=point_cloud.return_number,
            classifications=point_cloud.classifications,
            confidence_scores=point_cloud.confidence_scores,
            metadata=point_cloud.metadata,
            coordinate_system=target_crs
        )
        
        return new_point_cloud


class NoiseFilter:
    """Statistical and geometric noise filtering for point clouds."""
    
    def __init__(self):
        self.logger = logger
    
    def statistical_outlier_removal(self, points: np.ndarray, 
                                  k_neighbors: int = 20,
                                  std_multiplier: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers based on distance to neighbors.
        
        Args:
            points: Nx3 array of points
            k_neighbors: Number of neighbors to consider
            std_multiplier: Standard deviation multiplier for outlier threshold
            
        Returns:
            Boolean mask indicating inliers (True) and outliers (False)
        """
        if len(points) < k_neighbors:
            logger.warning("Not enough points for statistical filtering")
            return np.ones(len(points), dtype=bool)
        
        try:
            # Build KD-tree for efficient neighbor search
            tree = cKDTree(points)
            
            # Find k nearest neighbors for each point
            distances, _ = tree.query(points, k=k_neighbors + 1)  # +1 to exclude self
            
            # Calculate mean distance to neighbors (excluding self)
            mean_distances = np.mean(distances[:, 1:], axis=1)
            
            # Calculate statistics
            global_mean = np.mean(mean_distances)
            global_std = np.std(mean_distances)
            
            # Define outlier threshold
            threshold = global_mean + std_multiplier * global_std
            
            # Create inlier mask
            inlier_mask = mean_distances <= threshold
            
            outlier_count = np.sum(~inlier_mask)
            outlier_percentage = (outlier_count / len(points)) * 100
            
            self.logger.info(
                "Statistical outlier removal completed",
                total_points=len(points),
                outliers_removed=outlier_count,
                outlier_percentage=f"{outlier_percentage:.2f}%"
            )
            
            return inlier_mask
            
        except Exception as e:
            raise PointCloudProcessingError(
                operation="statistical outlier removal",
                details=str(e)
            ) from e
    
    def radius_outlier_removal(self, points: np.ndarray, 
                             radius: float, min_neighbors: int = 5) -> np.ndarray:
        """
        Remove points with too few neighbors within a given radius.
        
        Args:
            points: Nx3 array of points
            radius: Search radius
            min_neighbors: Minimum number of neighbors required
            
        Returns:
            Boolean mask indicating inliers
        """
        try:
            tree = cKDTree(points)
            
            # Count neighbors within radius for each point
            neighbor_counts = tree.query_ball_point(points, radius, return_length=True)
            
            # Create inlier mask (subtract 1 to exclude self)
            inlier_mask = (neighbor_counts - 1) >= min_neighbors
            
            outlier_count = np.sum(~inlier_mask)
            
            self.logger.info(
                "Radius outlier removal completed",
                total_points=len(points),
                outliers_removed=outlier_count,
                radius=radius,
                min_neighbors=min_neighbors
            )
            
            return inlier_mask
            
        except Exception as e:
            raise PointCloudProcessingError(
                operation="radius outlier removal",
                details=str(e)
            ) from e
    
    def local_outlier_factor_filter(self, points: np.ndarray, 
                                  n_neighbors: int = 20,
                                  contamination: float = 0.1) -> np.ndarray:
        """
        Use Local Outlier Factor (LOF) to detect outliers.
        
        Args:
            points: Nx3 array of points
            n_neighbors: Number of neighbors for LOF calculation
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean mask indicating inliers
        """
        try:
            # Use sklearn's LOF implementation
            lof = LocalOutlierFactor(
                n_neighbors=min(n_neighbors, len(points) - 1),
                contamination=contamination
            )
            
            # Fit and predict (-1 for outliers, 1 for inliers)
            outlier_labels = lof.fit_predict(points)
            inlier_mask = outlier_labels == 1
            
            outlier_count = np.sum(~inlier_mask)
            
            self.logger.info(
                "LOF outlier removal completed",
                total_points=len(points),
                outliers_removed=outlier_count,
                contamination=contamination
            )
            
            return inlier_mask
            
        except Exception as e:
            raise PointCloudProcessingError(
                operation="LOF outlier removal",
                details=str(e)
            ) from e


class PointCloudOperations:
    """Main class for point cloud operations and processing."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.transformer = CoordinateTransformer()
        self.noise_filter = NoiseFilter()
        self.logger = logger
    
    def load_with_memory_management(self, file_path: str, 
                                  max_memory_mb: Optional[float] = None) -> PointCloud:
        """
        Load point cloud with automatic memory management.
        
        Args:
            file_path: Path to point cloud file
            max_memory_mb: Maximum memory to use (MB)
            
        Returns:
            Loaded PointCloud object
        """
        from .point_cloud_processor import LASFileReader
        
        # Get file metadata first
        reader = LASFileReader()
        metadata = reader.extract_metadata_only(file_path)
        
        # Estimate memory usage
        estimated_memory = self.memory_manager.estimate_memory_usage(
            metadata.point_count, 
            ['intensity', 'colors', 'return_number']
        )
        
        # Check memory availability
        if max_memory_mb and estimated_memory > max_memory_mb:
            raise MemoryExhaustionError(
                operation="point cloud loading",
                required_memory=f"{estimated_memory:.1f} MB",
                available_memory=f"{max_memory_mb:.1f} MB"
            )
        
        if not self.memory_manager.check_memory_availability(estimated_memory):
            # Try to free memory
            self.memory_manager.optimize_memory_usage()
            
            if not self.memory_manager.check_memory_availability(estimated_memory):
                raise MemoryExhaustionError(
                    operation="point cloud loading",
                    required_memory=f"{estimated_memory:.1f} MB",
                    available_memory="insufficient"
                )
        
        # Load the point cloud
        return reader.read_file(file_path)
    
    def filter_noise(self, point_cloud: PointCloud, 
                    method: str = "statistical",
                    **kwargs) -> PointCloud:
        """
        Apply noise filtering to point cloud.
        
        Args:
            point_cloud: Input point cloud
            method: Filtering method ('statistical', 'radius', 'lof')
            **kwargs: Method-specific parameters
            
        Returns:
            Filtered point cloud
        """
        processing_logger = ProcessingLogger(
            operation_name=f"noise_filtering_{method}",
            total_items=len(point_cloud.points)
        )
        
        try:
            if method == "statistical":
                inlier_mask = self.noise_filter.statistical_outlier_removal(
                    point_cloud.points, **kwargs
                )
            elif method == "radius":
                inlier_mask = self.noise_filter.radius_outlier_removal(
                    point_cloud.points, **kwargs
                )
            elif method == "lof":
                inlier_mask = self.noise_filter.local_outlier_factor_filter(
                    point_cloud.points, **kwargs
                )
            else:
                raise ValueError(f"Unknown filtering method: {method}")
            
            # Apply filter to all arrays
            filtered_points = point_cloud.points[inlier_mask]
            filtered_colors = point_cloud.colors[inlier_mask] if point_cloud.colors is not None else None
            filtered_intensity = point_cloud.intensity[inlier_mask] if point_cloud.intensity is not None else None
            filtered_return_number = point_cloud.return_number[inlier_mask] if point_cloud.return_number is not None else None
            filtered_classifications = point_cloud.classifications[inlier_mask] if point_cloud.classifications is not None else None
            filtered_confidence = point_cloud.confidence_scores[inlier_mask] if point_cloud.confidence_scores is not None else None
            
            # Update metadata
            new_metadata = point_cloud.metadata
            if new_metadata:
                new_metadata.point_count = len(filtered_points)
                new_metadata.additional_info['noise_filtering'] = {
                    'method': method,
                    'original_count': len(point_cloud.points),
                    'filtered_count': len(filtered_points),
                    'removed_count': len(point_cloud.points) - len(filtered_points)
                }
            
            processing_logger.log_progress(
                len(filtered_points), 
                f"Noise filtering completed, {len(filtered_points)} points remaining"
            )
            processing_logger.log_completion(success=True)
            
            return PointCloud(
                points=filtered_points,
                colors=filtered_colors,
                intensity=filtered_intensity,
                return_number=filtered_return_number,
                classifications=filtered_classifications,
                confidence_scores=filtered_confidence,
                metadata=new_metadata,
                coordinate_system=point_cloud.coordinate_system
            )
            
        except Exception as e:
            processing_logger.log_error(e, {"method": method})
            processing_logger.log_completion(success=False)
            raise
    
    def subsample_points(self, point_cloud: PointCloud, 
                        method: str = "random", 
                        target_count: Optional[int] = None,
                        voxel_size: Optional[float] = None) -> PointCloud:
        """
        Subsample point cloud to reduce density.
        
        Args:
            point_cloud: Input point cloud
            method: Subsampling method ('random', 'uniform', 'voxel')
            target_count: Target number of points (for random/uniform)
            voxel_size: Voxel size for voxel-based subsampling
            
        Returns:
            Subsampled point cloud
        """
        if method == "random" and target_count:
            if target_count >= len(point_cloud.points):
                return point_cloud
            
            indices = np.random.choice(
                len(point_cloud.points), 
                size=target_count, 
                replace=False
            )
            
        elif method == "voxel" and voxel_size:
            # Simple voxel-based subsampling
            points = point_cloud.points
            
            # Create voxel grid
            min_coords = np.min(points, axis=0)
            voxel_indices = ((points - min_coords) / voxel_size).astype(int)
            
            # Find unique voxels and select one point per voxel
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            indices = unique_indices
            
        else:
            raise ValueError(f"Invalid subsampling method or missing parameters: {method}")
        
        # Apply subsampling
        subsampled_points = point_cloud.points[indices]
        subsampled_colors = point_cloud.colors[indices] if point_cloud.colors is not None else None
        subsampled_intensity = point_cloud.intensity[indices] if point_cloud.intensity is not None else None
        subsampled_return_number = point_cloud.return_number[indices] if point_cloud.return_number is not None else None
        
        # Update metadata
        new_metadata = point_cloud.metadata
        if new_metadata:
            new_metadata.point_count = len(subsampled_points)
            new_metadata.additional_info['subsampling'] = {
                'method': method,
                'original_count': len(point_cloud.points),
                'subsampled_count': len(subsampled_points)
            }
        
        self.logger.info(
            "Point cloud subsampling completed",
            method=method,
            original_count=len(point_cloud.points),
            subsampled_count=len(subsampled_points)
        )
        
        return PointCloud(
            points=subsampled_points,
            colors=subsampled_colors,
            intensity=subsampled_intensity,
            return_number=subsampled_return_number,
            metadata=new_metadata,
            coordinate_system=point_cloud.coordinate_system
        )
    
    def calculate_statistics(self, point_cloud: PointCloud) -> Dict[str, Any]:
        """Calculate comprehensive statistics for point cloud."""
        points = point_cloud.points
        
        stats_dict = {
            'point_count': len(points),
            'bounds': {
                'min_x': float(np.min(points[:, 0])),
                'max_x': float(np.max(points[:, 0])),
                'min_y': float(np.min(points[:, 1])),
                'max_y': float(np.max(points[:, 1])),
                'min_z': float(np.min(points[:, 2])),
                'max_z': float(np.max(points[:, 2]))
            },
            'centroid': {
                'x': float(np.mean(points[:, 0])),
                'y': float(np.mean(points[:, 1])),
                'z': float(np.mean(points[:, 2]))
            },
            'density': self._calculate_point_density(points),
            'coordinate_system': point_cloud.coordinate_system
        }
        
        # Add intensity statistics if available
        if point_cloud.intensity is not None:
            stats_dict['intensity'] = {
                'min': float(np.min(point_cloud.intensity)),
                'max': float(np.max(point_cloud.intensity)),
                'mean': float(np.mean(point_cloud.intensity)),
                'std': float(np.std(point_cloud.intensity))
            }
        
        return stats_dict
    
    def _calculate_point_density(self, points: np.ndarray) -> float:
        """Calculate average point density (points per square meter)."""
        if len(points) < 3:
            return 0.0
        
        # Calculate bounding box area
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        area = x_range * y_range
        
        if area > 0:
            return len(points) / area
        else:
            return 0.0


# Convenience functions
def load_point_cloud(file_path: str, max_memory_mb: Optional[float] = None) -> PointCloud:
    """Load point cloud with memory management."""
    operations = PointCloudOperations()
    return operations.load_with_memory_management(file_path, max_memory_mb)


def filter_point_cloud_noise(point_cloud: PointCloud, method: str = "statistical") -> PointCloud:
    """Apply noise filtering to point cloud."""
    operations = PointCloudOperations()
    return operations.filter_noise(point_cloud, method)


def transform_point_cloud_crs(point_cloud: PointCloud, target_crs: str) -> PointCloud:
    """Transform point cloud coordinate system."""
    transformer = CoordinateTransformer()
    return transformer.transform_point_cloud(point_cloud, target_crs)