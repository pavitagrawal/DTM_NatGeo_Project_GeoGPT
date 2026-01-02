"""
Machine Learning-based waterlogging prediction system.
Uses ensemble methods to predict waterlogging risk and duration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import xgboost as xgb
import joblib
from scipy.spatial.distance import cdist

from .models import DTM, HydrologyResults, WaterloggingRisk, RiskLevel
from .exceptions import WaterloggingPredictionError
from .logging_config import get_logger

logger = get_logger(__name__)


class WaterloggingPredictor:
    """ML-based waterlogging risk prediction system."""
    
    def __init__(
        self,
        model_type: str = "ensemble",
        risk_thresholds: Dict[str, float] = None
    ):
        """
        Initialize waterlogging predictor.
        
        Args:
            model_type: Type of model ('rf', 'xgb', 'ensemble')
            risk_thresholds: Risk classification thresholds
        """
        self.model_type = model_type
        self.risk_thresholds = risk_thresholds or {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }
        
        # Initialize models
        self.risk_classifier = None
        self.duration_regressor = None
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        logger.info(f"Waterlogging Predictor initialized: {model_type}")
    
    def extract_features(
        self,
        dtm: DTM,
        hydrology: HydrologyResults,
        additional_data: Optional[Dict[str, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Extract features for waterlogging prediction.
        
        Args:
            dtm: Digital Terrain Model
            hydrology: Hydrological analysis results
            additional_data: Optional additional data layers
            
        Returns:
            Feature DataFrame
        """
        logger.info("Extracting features for waterlogging prediction")
        
        try:
            rows, cols = dtm.elevation_grid.shape
            features = {}
            
            # Terrain features
            features['elevation'] = dtm.elevation_grid.flatten()
            features['slope'] = hydrology.slope.flatten()
            features['aspect'] = hydrology.aspect.flatten()
            features['twi'] = hydrology.topographic_wetness_index.flatten()
            
            # Hydrological features
            features['flow_accumulation'] = hydrology.flow_accumulation.flatten()
            features['flow_accumulation_log'] = np.log1p(hydrology.flow_accumulation.flatten())
            
            # Curvature features
            curvature = self._calculate_curvature(dtm.elevation_grid)
            features['curvature'] = curvature.flatten()
            
            # Distance features
            features['distance_to_streams'] = self._calculate_distance_to_streams(
                dtm, hydrology.stream_network
            ).flatten()
            
            # Topographic position
            features['tpi'] = self._calculate_tpi(dtm.elevation_grid).flatten()
            
            # Local relief
            features['local_relief'] = self._calculate_local_relief(dtm.elevation_grid).flatten()
            
            # Convergence index
            features['convergence_index'] = self._calculate_convergence_index(
                hydrology.flow_direction
            ).flatten()
            
            # Additional derived features
            elev_mean = np.nanmean(features['elevation'])
            elev_std = np.nanstd(features['elevation'])
            if elev_std == 0 or np.isnan(elev_std):
                elev_std = 1.0  # Avoid division by zero
            
            features['elevation_relative'] = (
                features['elevation'] - elev_mean
            ) / elev_std
            
            features['slope_position'] = self._calculate_slope_position(
                dtm.elevation_grid
            ).flatten()
            
            # Interaction features
            features['twi_slope_interaction'] = features['twi'] * features['slope']
            features['elevation_flow_interaction'] = (
                features['elevation_relative'] * features['flow_accumulation_log']
            )
            
            # Add additional data if provided
            if additional_data:
                for name, data in additional_data.items():
                    if data.shape == dtm.elevation_grid.shape:
                        features[f'additional_{name}'] = data.flatten()
            
            # Create DataFrame
            df = pd.DataFrame(features)
            
            # Remove invalid values and handle NaN more robustly
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with median for each column
            for col in df.columns:
                if df[col].isna().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):  # If median is also NaN, use 0
                        median_val = 0.0
                    df[col] = df[col].fillna(median_val)
            
            # Final check - replace any remaining NaN with 0
            df = df.fillna(0.0)
            
            # Verify no NaN values remain
            if df.isna().any().any():
                logger.warning("NaN values still present after cleaning, replacing with zeros")
                df = df.fillna(0.0)
            
            logger.info(f"Extracted {len(df.columns)} features for {len(df)} pixels")
            return df
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise WaterloggingPredictionError(f"Feature extraction failed: {e}")
    
    def train_models(
        self,
        features: pd.DataFrame,
        risk_labels: np.ndarray,
        duration_labels: Optional[np.ndarray] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train waterlogging prediction models.
        
        Args:
            features: Feature DataFrame
            risk_labels: Risk classification labels (0-3)
            duration_labels: Optional duration regression labels
            test_size: Test set proportion
            
        Returns:
            Training results and metrics
        """
        logger.info("Training waterlogging prediction models")
        
        try:
            # Split data
            X_train, X_test, y_risk_train, y_risk_test = train_test_split(
                features, risk_labels, test_size=test_size, random_state=42, stratify=risk_labels
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train risk classification model
            if self.model_type == "rf":
                self.risk_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            elif self.model_type == "xgb":
                self.risk_classifier = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # ensemble
                rf_model = RandomForestClassifier(
                    n_estimators=50, max_depth=15, random_state=42, n_jobs=-1
                )
                xgb_model = xgb.XGBClassifier(
                    n_estimators=50, max_depth=6, random_state=42, n_jobs=-1
                )
                
                # Train both models
                rf_model.fit(X_train_scaled, y_risk_train)
                xgb_model.fit(X_train_scaled, y_risk_train)
                
                # Create ensemble
                self.risk_classifier = EnsembleClassifier([rf_model, xgb_model])
            
            # Train the model
            if not isinstance(self.risk_classifier, EnsembleClassifier):
                self.risk_classifier.fit(X_train_scaled, y_risk_train)
            
            # Evaluate risk classification
            risk_pred = self.risk_classifier.predict(X_test_scaled)
            risk_score = self.risk_classifier.score(X_test_scaled, y_risk_test)
            
            results = {
                'risk_accuracy': risk_score,
                'risk_classification_report': classification_report(
                    y_risk_test, risk_pred, output_dict=True
                ),
                'feature_importance': self._get_feature_importance(features.columns)
            }
            
            # Train duration regression model if labels provided
            if duration_labels is not None:
                _, _, y_dur_train, y_dur_test = train_test_split(
                    features, duration_labels, test_size=test_size, random_state=42
                )
                
                self.duration_regressor = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                self.duration_regressor.fit(X_train_scaled, y_dur_train)
                duration_pred = self.duration_regressor.predict(X_test_scaled)
                duration_rmse = np.sqrt(mean_squared_error(y_dur_test, duration_pred))
                
                results['duration_rmse'] = duration_rmse
                results['duration_r2'] = self.duration_regressor.score(X_test_scaled, y_dur_test)
            
            self.is_trained = True
            logger.info(f"Model training completed. Risk accuracy: {risk_score:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise WaterloggingPredictionError(f"Model training failed: {e}")
    
    def predict_waterlogging(
        self,
        dtm: DTM,
        hydrology: HydrologyResults,
        additional_data: Optional[Dict[str, np.ndarray]] = None
    ) -> WaterloggingRisk:
        """
        Predict waterlogging risk for entire DTM.
        
        Args:
            dtm: Digital Terrain Model
            hydrology: Hydrological analysis results
            additional_data: Optional additional data layers
            
        Returns:
            Waterlogging risk prediction
        """
        if not self.is_trained:
            logger.warning("Model not trained, using synthetic predictions")
            return self._generate_synthetic_predictions(dtm, hydrology)
        
        logger.info("Predicting waterlogging risk")
        
        try:
            # Extract features
            features = self.extract_features(dtm, hydrology, additional_data)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features)
            
            # Predict risk probabilities
            risk_probs = self.risk_classifier.predict_proba(features_scaled)
            risk_classes = self.risk_classifier.predict(features_scaled)
            
            # Predict duration if model available
            duration_pred = None
            if self.duration_regressor is not None:
                duration_pred = self.duration_regressor.predict(features_scaled)
            
            # Reshape to grid
            rows, cols = dtm.elevation_grid.shape
            risk_grid = risk_classes.reshape(rows, cols)
            probability_grid = risk_probs.max(axis=1).reshape(rows, cols)
            
            if duration_pred is not None:
                duration_grid = duration_pred.reshape(rows, cols)
            else:
                duration_grid = np.zeros((rows, cols))
            
            # Calculate statistics
            risk_stats = self._calculate_risk_statistics(risk_grid, dtm.resolution)
            
            return WaterloggingRisk(
                risk_grid=risk_grid,
                probability_grid=probability_grid,
                duration_grid=duration_grid,
                risk_statistics=risk_stats,
                metadata={
                    'model_type': self.model_type,
                    'feature_count': len(features.columns),
                    'prediction_date': pd.Timestamp.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Waterlogging prediction failed: {e}")
            raise WaterloggingPredictionError(f"Prediction failed: {e}")
    
    def _generate_synthetic_predictions(
        self,
        dtm: DTM,
        hydrology: HydrologyResults
    ) -> WaterloggingRisk:
        """Generate synthetic predictions for demo purposes."""
        logger.info("Generating synthetic waterlogging predictions")
        
        rows, cols = dtm.elevation_grid.shape
        
        # Use TWI and flow accumulation for synthetic risk
        twi_norm = (hydrology.topographic_wetness_index - np.nanmin(hydrology.topographic_wetness_index)) / \
                   (np.nanmax(hydrology.topographic_wetness_index) - np.nanmin(hydrology.topographic_wetness_index))
        
        flow_norm = np.log1p(hydrology.flow_accumulation) / np.nanmax(np.log1p(hydrology.flow_accumulation))
        
        # Combine for risk score
        risk_score = 0.6 * twi_norm + 0.4 * flow_norm
        
        # Classify risk levels
        risk_grid = np.zeros((rows, cols), dtype=int)
        risk_grid[risk_score > 0.8] = 3  # Critical
        risk_grid[(risk_score > 0.6) & (risk_score <= 0.8)] = 2  # High
        risk_grid[(risk_score > 0.3) & (risk_score <= 0.6)] = 1  # Medium
        risk_grid[risk_score <= 0.3] = 0  # Low
        
        # Generate duration estimates
        duration_grid = risk_score * 24  # Hours
        
        # Calculate statistics
        risk_stats = self._calculate_risk_statistics(risk_grid, dtm.resolution)
        
        return WaterloggingRisk(
            risk_grid=risk_grid,
            probability_grid=risk_score,
            duration_grid=duration_grid,
            risk_statistics=risk_stats,
            metadata={
                'model_type': 'synthetic',
                'note': 'Generated for demo purposes'
            }
        )
    
    def _calculate_curvature(self, dem: np.ndarray) -> np.ndarray:
        """Calculate surface curvature."""
        try:
            # Second derivatives
            d2z_dx2 = np.gradient(np.gradient(dem, axis=1), axis=1)
            d2z_dy2 = np.gradient(np.gradient(dem, axis=0), axis=0)
            d2z_dxdy = np.gradient(np.gradient(dem, axis=1), axis=0)
            
            # Mean curvature
            curvature = d2z_dx2 + d2z_dy2
            
            # Replace any invalid values
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            
            return curvature
        except Exception as e:
            logger.warning(f"Curvature calculation failed: {e}, using zeros")
            return np.zeros_like(dem)
    
    def _calculate_distance_to_streams(
        self,
        dtm: DTM,
        stream_network
    ) -> np.ndarray:
        """Calculate distance to nearest stream."""
        rows, cols = dtm.elevation_grid.shape
        distances = np.full((rows, cols), 1000.0)  # Use large finite value instead of inf
        
        if not hasattr(stream_network, 'streams') or not stream_network.streams:
            # No streams found, return normalized distances based on position
            logger.warning("No streams found, using synthetic distance values")
            # Create synthetic distance based on position from center
            center_i, center_j = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    dist = np.sqrt((i - center_i)**2 + (j - center_j)**2) * dtm.resolution
                    distances[i, j] = min(dist, 1000.0)  # Cap at 1000m
            return distances
        
        # Get all stream points
        stream_points = []
        for stream in stream_network.streams:
            if hasattr(stream, 'geometry') and hasattr(stream['geometry'], 'coords'):
                stream_points.extend(list(stream['geometry'].coords))
        
        if not stream_points:
            logger.warning("No stream coordinates found, using synthetic distance values")
            # Create synthetic distance based on position from center
            center_i, center_j = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    dist = np.sqrt((i - center_i)**2 + (j - center_j)**2) * dtm.resolution
                    distances[i, j] = min(dist, 1000.0)  # Cap at 1000m
            return distances
        
        stream_points = np.array(stream_points)
        
        # Calculate distances for each grid cell
        try:
            minx, miny, maxx, maxy = dtm.bounds
        except:
            # If bounds not available, create synthetic bounds
            minx, miny = 0, 0
            maxx = cols * dtm.resolution
            maxy = rows * dtm.resolution
        
        for i in range(rows):
            for j in range(cols):
                x = minx + (j + 0.5) * dtm.resolution
                y = maxy - (i + 0.5) * dtm.resolution
                
                cell_point = np.array([[x, y]])
                try:
                    dist = cdist(cell_point, stream_points).min()
                    distances[i, j] = min(dist, 1000.0)  # Cap at 1000m
                except:
                    # If distance calculation fails, use position-based distance
                    dist = np.sqrt((i - rows//2)**2 + (j - cols//2)**2) * dtm.resolution
                    distances[i, j] = min(dist, 1000.0)
        
        return distances
    
    def _calculate_tpi(self, dem: np.ndarray, radius: int = 3) -> np.ndarray:
        """Calculate Topographic Position Index."""
        try:
            from scipy import ndimage
            
            # Create circular kernel
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            kernel = x**2 + y**2 <= radius**2
            kernel = kernel.astype(float)
            kernel[radius, radius] = 0  # Exclude center
            
            if kernel.sum() > 0:
                kernel = kernel / kernel.sum()
            else:
                kernel = np.ones_like(kernel) / kernel.size
            
            # Calculate mean elevation in neighborhood
            mean_elev = ndimage.convolve(dem, kernel, mode='constant', cval=0.0)
            
            # TPI = elevation - mean neighborhood elevation
            tpi = dem - mean_elev
            
            # Replace any invalid values
            tpi = np.nan_to_num(tpi, nan=0.0, posinf=0.0, neginf=0.0)
            
            return tpi
        except Exception as e:
            logger.warning(f"TPI calculation failed: {e}, using zeros")
            return np.zeros_like(dem)
    
    def _calculate_local_relief(self, dem: np.ndarray, radius: int = 5) -> np.ndarray:
        """Calculate local relief (max - min in neighborhood)."""
        try:
            from scipy import ndimage
            
            # Maximum and minimum filters
            max_elev = ndimage.maximum_filter(dem, size=2*radius+1)
            min_elev = ndimage.minimum_filter(dem, size=2*radius+1)
            
            relief = max_elev - min_elev
            
            # Replace any invalid values
            relief = np.nan_to_num(relief, nan=0.0, posinf=0.0, neginf=0.0)
            
            return relief
        except Exception as e:
            logger.warning(f"Local relief calculation failed: {e}, using zeros")
            return np.zeros_like(dem)
    
    def _calculate_convergence_index(self, flow_direction: np.ndarray) -> np.ndarray:
        """Calculate convergence index from flow directions."""
        # Simplified convergence calculation
        # Count how many neighbors flow into each cell
        rows, cols = flow_direction.shape
        convergence = np.zeros((rows, cols))
        
        # D8 directions
        directions = [1, 2, 4, 8, 16, 32, 64, 128]
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                inflow_count = 0
                
                # Check all neighbors
                for k, (di, dj) in enumerate(offsets):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Check if neighbor flows into current cell
                        neighbor_dir = flow_direction[ni, nj]
                        # Direction that would flow into current cell
                        opposite_dir = directions[(k + 4) % 8]
                        if neighbor_dir == opposite_dir:
                            inflow_count += 1
                
                convergence[i, j] = inflow_count
        
        return convergence
    
    def _calculate_slope_position(self, dem: np.ndarray) -> np.ndarray:
        """Calculate slope position (relative elevation in local area)."""
        try:
            from scipy import ndimage
            
            # Local mean elevation
            kernel = np.ones((11, 11)) / 121  # 11x11 neighborhood
            local_mean = ndimage.convolve(dem, kernel, mode='constant', cval=0.0)
            
            # Slope position = (elevation - local_mean) / local_std
            local_var = ndimage.convolve((dem - local_mean)**2, kernel, mode='constant', cval=0.0)
            local_std = np.sqrt(local_var)
            
            # Avoid division by zero
            local_std = np.where(local_std < 1e-6, 1e-6, local_std)
            
            slope_position = (dem - local_mean) / local_std
            
            # Replace any invalid values
            slope_position = np.nan_to_num(slope_position, nan=0.0, posinf=0.0, neginf=0.0)
            
            return slope_position
        except Exception as e:
            logger.warning(f"Slope position calculation failed: {e}, using zeros")
            return np.zeros_like(dem)
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if hasattr(self.risk_classifier, 'feature_importances_'):
            importance = self.risk_classifier.feature_importances_
            return dict(zip(feature_names, importance))
        else:
            return {}
    
    def _calculate_risk_statistics(
        self,
        risk_grid: np.ndarray,
        resolution: float
    ) -> Dict[str, Any]:
        """Calculate risk area statistics."""
        cell_area = resolution ** 2
        
        stats = {}
        for risk_level in range(4):
            mask = risk_grid == risk_level
            area = np.sum(mask) * cell_area
            percentage = (np.sum(mask) / risk_grid.size) * 100
            
            risk_name = ['low', 'medium', 'high', 'critical'][risk_level]
            stats[f'{risk_name}_area_m2'] = area
            stats[f'{risk_name}_percentage'] = percentage
        
        stats['total_area_m2'] = risk_grid.size * cell_area
        stats['high_risk_total_percentage'] = stats['high_percentage'] + stats['critical_percentage']
        
        return stats
    
    def save_model(self, model_path: Path) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise WaterloggingPredictionError("No trained model to save")
        
        model_data = {
            'risk_classifier': self.risk_classifier,
            'duration_regressor': self.duration_regressor,
            'feature_scaler': self.feature_scaler,
            'model_type': self.model_type,
            'risk_thresholds': self.risk_thresholds
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path) -> None:
        """Load trained model from disk."""
        model_data = joblib.load(model_path)
        
        self.risk_classifier = model_data['risk_classifier']
        self.duration_regressor = model_data.get('duration_regressor')
        self.feature_scaler = model_data['feature_scaler']
        self.model_type = model_data['model_type']
        self.risk_thresholds = model_data['risk_thresholds']
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")


class EnsembleClassifier:
    """Simple ensemble classifier for combining multiple models."""
    
    def __init__(self, models: List[Any]):
        self.models = models
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority voting."""
        predictions = np.array([model.predict(X) for model in self.models])
        # Majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using average."""
        probabilities = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probabilities, axis=0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def predict_waterlogging_risk(
    dtm: DTM,
    hydrology: HydrologyResults,
    model_type: str = "ensemble"
) -> WaterloggingRisk:
    """
    Convenience function for waterlogging prediction.
    
    Args:
        dtm: Digital Terrain Model
        hydrology: Hydrological analysis results
        model_type: Type of model to use
        
    Returns:
        Waterlogging risk prediction
    """
    predictor = WaterloggingPredictor(model_type=model_type)
    return predictor.predict_waterlogging(dtm, hydrology)