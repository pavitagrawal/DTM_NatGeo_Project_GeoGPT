"""
AI-based ground vs non-ground classification pipeline for village-scale drone surveys.

This module implements a hybrid approach combining:
1. Cloth Simulation Filter (CSF) for baseline classification
2. Machine Learning classifier (Random Forest/XGBoost) for refinement
3. Geometric feature extraction optimized for Indian village terrain
4. Scalable architecture ready for deep learning (PointNet++) integration

Designed specifically for Indian village characteristics:
- Mixed terrain (agricultural fields, residential areas, roads)
- Complex vegetation (crops, trees, bushes)
- Traditional structures (houses, walls, sheds)
- Seasonal variations (crop growth, flooding)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import warnings

from .models import PointCloud, GroundClassificationResults
from .exceptions import PointCloudProcessingError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GeometricFeatures:
    """Container for geometric features extracted from point neighborhoods."""
    height_variance: np.ndarray
    slope: np.ndarray
    planarity: np.ndarray
    sphericity: np.ndarray
    linearity: np.ndarray
    local_density: np.ndarray
    height_above_ground: np.ndarray
    surface_roughness: np.ndarray
    curvature: np.ndarray
    return_intensity_ratio: np.ndarray


class CSFGroundFilter:
    """
    Cloth Simulation Filter implementation for baseline ground classification.
    
    WHY CSF for Indian villages:
    - Robust to complex mixed terrain (fields, roads, buildings)
    - Handles vegetation well (crops, trees, bushes)
    - No training data required (unsupervised)
    - Fast processing for large datasets
    - Proven performance in agricultural landscapes
    """
    
    def __init__(
        self,
        cloth_resolution: float = 1.0,
        max_iterations: int = 500,
        classification_threshold: float = 0.5,
        rigidness: int = 2
    ):
        """
        Initialize CSF parameters.
        
        Args:
            cloth_resolution: Grid resolution for cloth simulation (meters)
            max_iterations: Maximum simulation iterations
            classification_threshold: Distance threshold for ground classification
            rigidness: Cloth rigidness (1=steep, 2=relief, 3=flat terrain)
            
        WHY these defaults for Indian villages:
        - cloth_resolution=1.0m: Captures village-scale features (houses, fields)
        - rigidness=2: Handles mixed relief terrain typical of villages
        - threshold=0.5m: Separates ground from low vegetation/structures
        """
        self.cloth_resolution = cloth_resolution
        self.max_iterations = max_iterations
        self.classification_threshold = classification_threshold
        self.rigidness = rigidness
        
        logger.info(f"CSF initialized: resolution={cloth_resolution}m, "
                   f"threshold={classification_threshold}m, rigidness={rigidness}")
    
    def classify_ground_points(self, point_cloud: PointCloud) -> np.ndarray:
        """
        Classify ground points using Cloth Simulation Filter.
        
        WHY CSF works for villages:
        - Simulates cloth draped over terrain
        - Cloth settles to ground surface naturally
        - Points below cloth are classified as ground
        - Robust to vegetation and structures above ground
        
        Args:
            point_cloud: Input point cloud
            
        Returns:
            Boolean array: True for ground points, False for non-ground
        """
        logger.info(f"Running CSF on {len(point_cloud.points):,} points")
        
        try:
            points = point_cloud.points
            
            # Create cloth grid based on point cloud bounds
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()
            
            # Grid dimensions
            grid_width = int((x_max - x_min) / self.cloth_resolution) + 1
            grid_height = int((y_max - y_min) / self.cloth_resolution) + 1
            
            logger.info(f"Cloth grid: {grid_width} x {grid_height}")
            
            # Initialize cloth at maximum elevation
            cloth_z = np.full((grid_height, grid_width), points[:, 2].max() + 10)
            
            # Simulate cloth settling
            for iteration in range(self.max_iterations):
                cloth_z_old = cloth_z.copy()
                
                # Apply gravity and constraints
                cloth_z = self._simulate_cloth_step(cloth_z, points, x_min, y_min)
                
                # Check convergence
                if np.allclose(cloth_z, cloth_z_old, atol=0.01):
                    logger.info(f"CSF converged after {iteration + 1} iterations")
                    break
            
            # Classify points based on distance to cloth
            ground_mask = self._classify_points_to_cloth(
                points, cloth_z, x_min, y_min
            )
            
            ground_count = np.sum(ground_mask)
            ground_percentage = (ground_count / len(points)) * 100
            
            logger.info(f"CSF classification: {ground_count:,} ground points "
                       f"({ground_percentage:.1f}%)")
            
            return ground_mask
            
        except Exception as e:
            logger.error(f"CSF classification failed: {e}")
            raise PointCloudProcessingError("CSF classification", str(e))
    
    def _simulate_cloth_step(
        self,
        cloth_z: np.ndarray,
        points: np.ndarray,
        x_min: float,
        y_min: float
    ) -> np.ndarray:
        """Simulate one step of cloth physics."""
        
        # Apply gravity (downward force)
        gravity_force = 0.1
        cloth_z_new = cloth_z - gravity_force
        
        # Apply smoothing based on rigidness
        if self.rigidness > 1:
            # Smooth cloth surface
            from scipy.ndimage import gaussian_filter
            sigma = self.rigidness * 0.5
            cloth_z_new = gaussian_filter(cloth_z_new, sigma=sigma)
        
        # Constrain cloth to not go below point cloud
        for i in range(cloth_z.shape[0]):
            for j in range(cloth_z.shape[1]):
                # Find points in this grid cell
                cell_x = x_min + j * self.cloth_resolution
                cell_y = y_min + i * self.cloth_resolution
                
                # Find points within grid cell
                in_cell = (
                    (points[:, 0] >= cell_x) & 
                    (points[:, 0] < cell_x + self.cloth_resolution) &
                    (points[:, 1] >= cell_y) & 
                    (points[:, 1] < cell_y + self.cloth_resolution)
                )
                
                if np.any(in_cell):
                    max_z_in_cell = points[in_cell, 2].max()
                    cloth_z_new[i, j] = max(cloth_z_new[i, j], max_z_in_cell)
        
        return cloth_z_new
    
    def _classify_points_to_cloth(
        self,
        points: np.ndarray,
        cloth_z: np.ndarray,
        x_min: float,
        y_min: float
    ) -> np.ndarray:
        """Classify points based on distance to cloth surface."""
        
        ground_mask = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            # Find corresponding cloth grid position
            grid_x = int((point[0] - x_min) / self.cloth_resolution)
            grid_y = int((point[1] - y_min) / self.cloth_resolution)
            
            # Clamp to grid bounds
            grid_x = max(0, min(grid_x, cloth_z.shape[1] - 1))
            grid_y = max(0, min(grid_y, cloth_z.shape[0] - 1))
            
            # Get cloth elevation at this position
            cloth_elevation = cloth_z[grid_y, grid_x]
            
            # Classify based on distance to cloth
            distance_to_cloth = cloth_elevation - point[2]
            ground_mask[i] = distance_to_cloth <= self.classification_threshold
        
        return ground_mask


class GeometricFeatureExtractor:
    """
    Extract geometric features for ML-based ground classification.
    
    WHY these features for Indian villages:
    - Height variance: Distinguishes flat ground from vegetation
    - Slope: Ground is typically less steep than building walls
    - Planarity: Ground surfaces are more planar than vegetation
    - Local density: Ground has consistent point density
    - Surface roughness: Ground is smoother than vegetation
    
    Features optimized for village characteristics:
    - Agricultural fields (flat, regular patterns)
    - Residential areas (mixed heights, structures)
    - Roads and paths (linear, flat features)
    - Vegetation (irregular, varying heights)
    """
    
    def __init__(self, neighborhood_radius: float = 2.0, min_neighbors: int = 10):
        """
        Initialize feature extractor.
        
        Args:
            neighborhood_radius: Radius for neighborhood search (meters)
            min_neighbors: Minimum neighbors required for feature calculation
            
        WHY these parameters:
        - 2.0m radius: Captures local terrain characteristics
        - 10 neighbors: Sufficient for statistical reliability
        """
        self.neighborhood_radius = neighborhood_radius
        self.min_neighbors = min_neighbors
        
        logger.info(f"Feature extractor initialized: radius={neighborhood_radius}m, "
                   f"min_neighbors={min_neighbors}")
    
    def extract_features(self, point_cloud: PointCloud) -> GeometricFeatures:
        """
        Extract comprehensive geometric features for each point.
        
        WHY comprehensive features:
        - Single features can be ambiguous (e.g., flat roof vs ground)
        - Multiple features provide robust classification
        - Geometric features work across different vegetation types
        - No spectral data dependency (works with any LiDAR)
        
        Args:
            point_cloud: Input point cloud
            
        Returns:
            GeometricFeatures object with all computed features
        """
        logger.info(f"Extracting geometric features for {len(point_cloud.points):,} points")
        
        points = point_cloud.points
        n_points = len(points)
        
        # Build spatial index for efficient neighborhood queries
        # WHY KDTree: O(log n) neighborhood queries vs O(n) brute force
        tree = cKDTree(points[:, :2])  # 2D spatial index (X, Y only)
        
        # Initialize feature arrays
        height_variance = np.zeros(n_points)
        slope = np.zeros(n_points)
        planarity = np.zeros(n_points)
        sphericity = np.zeros(n_points)
        linearity = np.zeros(n_points)
        local_density = np.zeros(n_points)
        height_above_ground = np.zeros(n_points)
        surface_roughness = np.zeros(n_points)
        curvature = np.zeros(n_points)
        return_intensity_ratio = np.zeros(n_points)
        
        # Process points in batches for memory efficiency
        batch_size = 10000
        n_batches = (n_points + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_points)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx + 1}/{n_batches}")
            
            # Process batch
            batch_indices = range(start_idx, end_idx)
            self._extract_batch_features(
                points, tree, batch_indices,
                height_variance, slope, planarity, sphericity, linearity,
                local_density, height_above_ground, surface_roughness,
                curvature, return_intensity_ratio, point_cloud
            )
        
        logger.info("Feature extraction completed")
        
        return GeometricFeatures(
            height_variance=height_variance,
            slope=slope,
            planarity=planarity,
            sphericity=sphericity,
            linearity=linearity,
            local_density=local_density,
            height_above_ground=height_above_ground,
            surface_roughness=surface_roughness,
            curvature=curvature,
            return_intensity_ratio=return_intensity_ratio
        )
    
    def _extract_batch_features(
        self,
        points: np.ndarray,
        tree: cKDTree,
        batch_indices: range,
        height_variance: np.ndarray,
        slope: np.ndarray,
        planarity: np.ndarray,
        sphericity: np.ndarray,
        linearity: np.ndarray,
        local_density: np.ndarray,
        height_above_ground: np.ndarray,
        surface_roughness: np.ndarray,
        curvature: np.ndarray,
        return_intensity_ratio: np.ndarray,
        point_cloud: PointCloud
    ):
        """Extract features for a batch of points."""
        
        for i in batch_indices:
            point = points[i]
            
            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(
                point[:2], self.neighborhood_radius
            )
            
            if len(neighbor_indices) < self.min_neighbors:
                # Not enough neighbors - use default values
                continue
            
            neighbor_points = points[neighbor_indices]
            
            # Feature 1: Height Variance
            # WHY: Ground has low height variance, vegetation has high variance
            height_variance[i] = np.var(neighbor_points[:, 2])
            
            # Feature 2: Slope
            # WHY: Ground typically has lower slope than building walls
            slope[i] = self._calculate_local_slope(neighbor_points)
            
            # Feature 3-5: Eigenvalue-based features (planarity, sphericity, linearity)
            # WHY: Describe local 3D structure - ground is planar, vegetation is spherical
            eigenvalues = self._calculate_eigenvalues(neighbor_points)
            if eigenvalues is not None:
                planarity[i] = eigenvalues[0]
                sphericity[i] = eigenvalues[1] 
                linearity[i] = eigenvalues[2]
            
            # Feature 6: Local Density
            # WHY: Ground has consistent density, vegetation varies
            area = np.pi * (self.neighborhood_radius ** 2)
            local_density[i] = len(neighbor_indices) / area
            
            # Feature 7: Height Above Ground
            # WHY: Ground points are at minimum local elevation
            min_z = neighbor_points[:, 2].min()
            height_above_ground[i] = point[2] - min_z
            
            # Feature 8: Surface Roughness
            # WHY: Ground is smoother than vegetation
            surface_roughness[i] = self._calculate_surface_roughness(neighbor_points)
            
            # Feature 9: Curvature
            # WHY: Ground has low curvature, structures have high curvature
            curvature[i] = self._calculate_curvature(neighbor_points)
            
            # Feature 10: Return/Intensity Ratio
            # WHY: Ground typically has single returns, vegetation has multiple
            if point_cloud.return_number is not None and point_cloud.intensity is not None:
                return_intensity_ratio[i] = self._calculate_return_intensity_ratio(
                    i, point_cloud
                )
    
    def _calculate_local_slope(self, neighbor_points: np.ndarray) -> float:
        """Calculate local slope using plane fitting."""
        if len(neighbor_points) < 3:
            return 0.0
        
        try:
            # Fit plane to neighborhood points
            centroid = np.mean(neighbor_points, axis=0)
            centered_points = neighbor_points - centroid
            
            # SVD to find plane normal
            _, _, vh = np.linalg.svd(centered_points)
            normal = vh[-1]  # Last row is normal to best-fit plane
            
            # Calculate slope from normal vector
            slope_rad = np.arccos(abs(normal[2]))  # Angle from vertical
            slope_deg = np.degrees(slope_rad)
            
            return slope_deg
            
        except:
            return 0.0
    
    def _calculate_eigenvalues(self, neighbor_points: np.ndarray) -> Optional[np.ndarray]:
        """Calculate normalized eigenvalues for geometric features."""
        if len(neighbor_points) < 3:
            return None
        
        try:
            # Center points
            centroid = np.mean(neighbor_points, axis=0)
            centered_points = neighbor_points - centroid
            
            # Covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Eigenvalues (sorted descending)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            
            # Normalize to avoid division by zero
            eigenvals = eigenvals / (eigenvals.sum() + 1e-10)
            
            # Calculate geometric features
            # Planarity: (λ1 - λ2) / λ1
            planarity = (eigenvals[0] - eigenvals[1]) / (eigenvals[0] + 1e-10)
            
            # Sphericity: λ2 / λ1  
            sphericity = eigenvals[1] / (eigenvals[0] + 1e-10)
            
            # Linearity: (λ2 - λ3) / λ1
            linearity = (eigenvals[1] - eigenvals[2]) / (eigenvals[0] + 1e-10)
            
            return np.array([planarity, sphericity, linearity])
            
        except:
            return None
    
    def _calculate_surface_roughness(self, neighbor_points: np.ndarray) -> float:
        """Calculate surface roughness as standard deviation of distances to fitted plane."""
        if len(neighbor_points) < 3:
            return 0.0
        
        try:
            # Fit plane
            centroid = np.mean(neighbor_points, axis=0)
            centered_points = neighbor_points - centroid
            
            _, _, vh = np.linalg.svd(centered_points)
            normal = vh[-1]
            
            # Calculate distances to plane
            distances = np.abs(np.dot(centered_points, normal))
            
            return np.std(distances)
            
        except:
            return 0.0
    
    def _calculate_curvature(self, neighbor_points: np.ndarray) -> float:
        """Calculate local curvature using quadratic surface fitting."""
        if len(neighbor_points) < 6:
            return 0.0
        
        try:
            # Simple curvature approximation using height variance
            # More sophisticated methods would fit quadratic surfaces
            z_values = neighbor_points[:, 2]
            z_mean = np.mean(z_values)
            curvature = np.sum((z_values - z_mean) ** 2) / len(z_values)
            
            return curvature
            
        except:
            return 0.0
    
    def _calculate_return_intensity_ratio(
        self,
        point_idx: int,
        point_cloud: PointCloud
    ) -> float:
        """Calculate ratio of return number to intensity."""
        try:
            if (point_cloud.return_number is None or 
                point_cloud.intensity is None):
                return 0.0
            
            return_num = point_cloud.return_number[point_idx]
            intensity = point_cloud.intensity[point_idx]
            
            if intensity == 0:
                return 0.0
            
            return return_num / intensity
            
        except:
            return 0.0


class HybridGroundClassifier:
    """
    Hybrid ground classification combining CSF baseline with ML refinement.
    
    WHY hybrid approach for Indian villages:
    - CSF provides robust baseline for mixed terrain
    - ML refines classification using geometric features
    - Combines unsupervised (CSF) with supervised (ML) methods
    - Handles complex village scenarios (crops, buildings, roads)
    - Scalable to deep learning (PointNet++) later
    """
    
    def __init__(
        self,
        csf_params: Optional[Dict[str, Any]] = None,
        ml_model: str = "random_forest",
        feature_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            csf_params: CSF configuration parameters
            ml_model: ML model type ("random_forest" or "xgboost")
            feature_params: Feature extraction parameters
        """
        # Initialize CSF
        csf_params = csf_params or {}
        self.csf = CSFGroundFilter(**csf_params)
        
        # Initialize feature extractor
        feature_params = feature_params or {}
        self.feature_extractor = GeometricFeatureExtractor(**feature_params)
        
        # Initialize ML model
        self.ml_model_type = ml_model
        self.ml_model = None
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        logger.info(f"Hybrid classifier initialized: CSF + {ml_model}")
    
    def classify_ground_points(
        self,
        point_cloud: PointCloud,
        use_ml_refinement: bool = True
    ) -> GroundClassificationResults:
        """
        Classify ground points using hybrid approach.
        
        Pipeline:
        1. CSF baseline classification
        2. Extract geometric features
        3. ML refinement (if trained model available)
        4. Combine results with confidence scores
        
        Args:
            point_cloud: Input point cloud
            use_ml_refinement: Whether to apply ML refinement
            
        Returns:
            Ground classification results with confidence scores
        """
        logger.info(f"Starting hybrid ground classification for {len(point_cloud.points):,} points")
        
        try:
            # Step 1: CSF baseline classification
            logger.info("Step 1: CSF baseline classification")
            csf_ground_mask = self.csf.classify_ground_points(point_cloud)
            
            # Step 2: Extract geometric features
            logger.info("Step 2: Extracting geometric features")
            features = self.feature_extractor.extract_features(point_cloud)
            
            # Step 3: ML refinement (if model is trained)
            if use_ml_refinement and self.is_trained:
                logger.info("Step 3: ML refinement")
                ml_ground_mask, ml_confidence = self._apply_ml_refinement(
                    features, csf_ground_mask
                )
                
                # Combine CSF and ML results
                final_ground_mask, confidence_scores = self._combine_classifications(
                    csf_ground_mask, ml_ground_mask, ml_confidence
                )
                
                method = f"CSF + {self.ml_model_type}"
                
            else:
                logger.info("Step 3: Using CSF results (no ML refinement)")
                final_ground_mask = csf_ground_mask
                confidence_scores = np.ones(len(point_cloud.points)) * 0.8  # Default confidence
                method = "CSF only"
            
            # Create classification array (LAS standard: 2=ground, 1=unclassified)
            classifications = np.ones(len(point_cloud.points), dtype=np.uint8)  # Default: unclassified
            classifications[final_ground_mask] = 2  # Ground points
            
            # Calculate accuracy metrics
            ground_count = np.sum(final_ground_mask)
            ground_percentage = (ground_count / len(point_cloud.points)) * 100
            
            accuracy_metrics = {
                'ground_points': int(ground_count),
                'ground_percentage': ground_percentage,
                'total_points': len(point_cloud.points),
                'method': method,
                'mean_confidence': float(np.mean(confidence_scores))
            }
            
            logger.info(f"Classification completed: {ground_count:,} ground points "
                       f"({ground_percentage:.1f}%)")
            
            return GroundClassificationResults(
                classifications=classifications,
                confidence_scores=confidence_scores,
                class_names=['unclassified', 'ground'],
                accuracy_metrics=accuracy_metrics,
                metadata={
                    'csf_params': self.csf.__dict__,
                    'feature_params': self.feature_extractor.__dict__,
                    'ml_model': self.ml_model_type,
                    'is_trained': self.is_trained
                }
            )
            
        except Exception as e:
            logger.error(f"Ground classification failed: {e}")
            raise PointCloudProcessingError("ground classification", str(e))
    
    def train_ml_model(
        self,
        training_point_clouds: List[PointCloud],
        training_labels: List[np.ndarray],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train ML model for ground classification refinement.
        
        WHY training approach:
        - Use CSF as baseline to generate initial labels
        - Manual correction of CSF results for training data
        - Learn geometric patterns specific to village terrain
        - Cross-validation to ensure generalization
        
        Args:
            training_point_clouds: List of training point clouds
            training_labels: List of ground truth labels (boolean arrays)
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Training ML model on {len(training_point_clouds)} point clouds")
        
        try:
            # Extract features from all training data
            all_features = []
            all_labels = []
            
            for i, (pc, labels) in enumerate(zip(training_point_clouds, training_labels)):
                logger.info(f"Processing training cloud {i + 1}/{len(training_point_clouds)}")
                
                # Extract features
                features = self.feature_extractor.extract_features(pc)
                
                # Convert to feature matrix
                feature_matrix = self._features_to_matrix(features)
                
                all_features.append(feature_matrix)
                all_labels.append(labels.astype(int))
            
            # Combine all training data
            X = np.vstack(all_features)
            y = np.hstack(all_labels)
            
            logger.info(f"Training data: {X.shape[0]:,} points, {X.shape[1]} features")
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Split training/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Train model
            if self.ml_model_type == "random_forest":
                self.ml_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'  # Handle class imbalance
                )
            elif self.ml_model_type == "xgboost":
                self.ml_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                )
            
            # Train
            self.ml_model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.ml_model.score(X_train, y_train)
            val_score = self.ml_model.score(X_val, y_val)
            
            # Cross-validation
            cv_scores = cross_val_score(self.ml_model, X_scaled, y, cv=5)
            
            # Predictions for detailed metrics
            y_pred = self.ml_model.predict(X_val)
            
            self.is_trained = True
            
            results = {
                'train_accuracy': train_score,
                'validation_accuracy': val_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_val, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
                'feature_importance': self._get_feature_importance(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            logger.info(f"Training completed: val_accuracy={val_score:.3f}, "
                       f"cv_score={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            raise PointCloudProcessingError("ML training", str(e))
    
    def _apply_ml_refinement(
        self,
        features: GeometricFeatures,
        csf_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ML model to refine CSF classification."""
        
        # Convert features to matrix
        feature_matrix = self._features_to_matrix(features)
        
        # Scale features
        feature_matrix_scaled = self.feature_scaler.transform(feature_matrix)
        
        # Predict
        ml_predictions = self.ml_model.predict(feature_matrix_scaled)
        ml_probabilities = self.ml_model.predict_proba(feature_matrix_scaled)
        
        # Convert to boolean mask
        ml_ground_mask = ml_predictions.astype(bool)
        
        # Confidence scores (probability of predicted class)
        confidence_scores = np.max(ml_probabilities, axis=1)
        
        return ml_ground_mask, confidence_scores
    
    def _combine_classifications(
        self,
        csf_mask: np.ndarray,
        ml_mask: np.ndarray,
        ml_confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine CSF and ML classifications with confidence weighting."""
        
        # Agreement between CSF and ML
        agreement = csf_mask == ml_mask
        
        # Final classification
        final_mask = np.zeros_like(csf_mask, dtype=bool)
        final_confidence = np.zeros_like(ml_confidence)
        
        # Where they agree, use the agreed classification with high confidence
        final_mask[agreement] = csf_mask[agreement]
        final_confidence[agreement] = np.minimum(ml_confidence[agreement] + 0.2, 1.0)
        
        # Where they disagree, use ML with reduced confidence
        disagreement = ~agreement
        final_mask[disagreement] = ml_mask[disagreement]
        final_confidence[disagreement] = ml_confidence[disagreement] * 0.7
        
        return final_mask, final_confidence
    
    def _features_to_matrix(self, features: GeometricFeatures) -> np.ndarray:
        """Convert GeometricFeatures to feature matrix."""
        
        feature_arrays = [
            features.height_variance,
            features.slope,
            features.planarity,
            features.sphericity,
            features.linearity,
            features.local_density,
            features.height_above_ground,
            features.surface_roughness,
            features.curvature,
            features.return_intensity_ratio
        ]
        
        # Stack features as columns
        feature_matrix = np.column_stack(feature_arrays)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        
        if not self.is_trained or not hasattr(self.ml_model, 'feature_importances_'):
            return {}
        
        feature_names = [
            'height_variance', 'slope', 'planarity', 'sphericity', 'linearity',
            'local_density', 'height_above_ground', 'surface_roughness',
            'curvature', 'return_intensity_ratio'
        ]
        
        importance_dict = dict(zip(feature_names, self.ml_model.feature_importances_))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, model_path: Path) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'ml_model': self.ml_model,
            'feature_scaler': self.feature_scaler,
            'ml_model_type': self.ml_model_type,
            'csf_params': self.csf.__dict__,
            'feature_params': self.feature_extractor.__dict__
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path) -> None:
        """Load trained model from disk."""
        model_data = joblib.load(model_path)
        
        self.ml_model = model_data['ml_model']
        self.feature_scaler = model_data['feature_scaler']
        self.ml_model_type = model_data['ml_model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")


def classify_ground_points_hybrid(
    point_cloud: PointCloud,
    csf_params: Optional[Dict[str, Any]] = None,
    ml_model: str = "random_forest",
    use_ml_refinement: bool = False
) -> GroundClassificationResults:
    """
    Convenience function for hybrid ground classification.
    
    Args:
        point_cloud: Input point cloud
        csf_params: CSF parameters
        ml_model: ML model type
        use_ml_refinement: Whether to use ML refinement
        
    Returns:
        Ground classification results
    """
    classifier = HybridGroundClassifier(
        csf_params=csf_params,
        ml_model=ml_model
    )
    
    return classifier.classify_ground_points(
        point_cloud,
        use_ml_refinement=use_ml_refinement
    )


# Deep Learning Integration Path (PointNet++ Ready)
class DeepLearningGroundClassifier:
    """
    Placeholder for future PointNet++ integration.
    
    WHY PointNet++ for ground classification:
    - Learns hierarchical features from point neighborhoods
    - Handles irregular point distributions naturally
    - Can learn complex patterns (vegetation types, building styles)
    - Scalable to large point clouds through sampling
    - State-of-the-art accuracy for point cloud classification
    
    Integration path:
    1. Use hybrid classifier to generate training labels
    2. Train PointNet++ on village-specific data
    3. Fine-tune for Indian terrain characteristics
    4. Deploy for production classification
    """
    
    def __init__(self):
        """Initialize PointNet++ classifier (future implementation)."""
        logger.info("PointNet++ classifier placeholder initialized")
        logger.info("Future implementation will include:")
        logger.info("- Hierarchical feature learning")
        logger.info("- Multi-scale neighborhood processing")
        logger.info("- End-to-end training on village data")
        logger.info("- GPU acceleration for large datasets")
    
    def train_pointnet(self, training_data: List[PointCloud]) -> None:
        """Train PointNet++ model (future implementation)."""
        logger.info("PointNet++ training not yet implemented")
        logger.info("Will use hybrid classifier labels as training data")
    
    def classify_with_pointnet(self, point_cloud: PointCloud) -> GroundClassificationResults:
        """Classify using PointNet++ (future implementation)."""
        logger.info("PointNet++ classification not yet implemented")
        logger.info("Falling back to hybrid classifier")
        
        # Fallback to hybrid classifier
        classifier = HybridGroundClassifier()
        return classifier.classify_ground_points(point_cloud, use_ml_refinement=False)