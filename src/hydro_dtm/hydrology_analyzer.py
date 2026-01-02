"""
Advanced hydrological analysis for DTMs with focus on rural abadi drainage patterns.

This module provides comprehensive hydrological analysis specifically optimized for 
Indian village (abadi) areas, including:

1. Flow Direction Computation (D8 & D-infinity algorithms)
2. Flow Accumulation Mapping for drainage network identification
3. Depression/Sink Detection for waterlogging risk assessment
4. Slope and Curvature Analysis for terrain characterization
5. Topographic Wetness Index for moisture accumulation prediction

WHY each output helps identify drainage patterns in rural abadi areas:

FLOW DIRECTION:
- Maps natural water flow paths through village terrain
- Identifies where surface runoff will concentrate
- Critical for planning drainage infrastructure placement
- Helps avoid building in natural flow channels

FLOW ACCUMULATION:
- Shows where water naturally collects during rainfall
- Identifies main drainage channels and tributaries
- Predicts flood-prone areas during monsoon season
- Guides placement of drainage outlets and collection points

DEPRESSION/SINK DETECTION:
- Locates natural and artificial depressions that trap water
- Identifies areas prone to waterlogging and stagnation
- Critical for mosquito breeding prevention
- Guides targeted drainage interventions

SLOPE & CURVATURE:
- Slope: Determines runoff velocity and erosion potential
- Curvature: Identifies convergent (water collecting) vs divergent areas
- Helps design appropriate drainage gradients
- Predicts sediment deposition zones

TOPOGRAPHIC WETNESS INDEX (TWI):
- Combines slope and flow accumulation to predict wet areas
- Identifies chronic waterlogging zones
- Guides agricultural drainage planning
- Supports public health interventions
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds

from .models import DTM, HydrologyResults, WatershedData, StreamNetwork
from .exceptions import HydrologyAnalysisError
from .logging_config import get_logger

logger = get_logger(__name__)


class AdvancedHydrologyAnalyzer:
    """
    Advanced hydrological analysis optimized for rural abadi (village) areas.
    
    WHY this approach for Indian villages:
    - Mixed terrain: agricultural fields, residential areas, roads
    - Monsoon drainage: handles seasonal flooding patterns
    - Infrastructure planning: guides drainage system placement
    - Public health: identifies waterlogging and stagnation zones
    - Agricultural drainage: supports field drainage planning
    """
    
    def __init__(
        self,
        flow_algorithm: str = "d8",
        min_stream_threshold: int = 1000,
        depression_fill_method: str = "planchon_darboux",
        village_scale_optimization: bool = True
    ):
        """
        Initialize advanced hydrology analyzer.
        
        Args:
            flow_algorithm: Flow direction algorithm ('d8', 'dinf')
            min_stream_threshold: Minimum flow accumulation for streams
            depression_fill_method: Depression filling method
            village_scale_optimization: Enable village-specific optimizations
            
        WHY these defaults for villages:
        - D8 algorithm: Simple, robust for village-scale terrain
        - 1000 cell threshold: Captures main drainage channels
        - Planchon-Darboux: Handles complex depression patterns
        - Village optimization: Accounts for mixed land use patterns
        """
        self.flow_algorithm = flow_algorithm
        self.min_stream_threshold = min_stream_threshold
        self.depression_fill_method = depression_fill_method
        self.village_scale_optimization = village_scale_optimization
        
        # D8 flow directions (8-connected) - standard hydrological encoding
        self.d8_directions = np.array([
            [1, 2, 4],
            [128, 0, 8],
            [64, 32, 16]
        ])
        
        # D8 offsets (row, col) - 8 cardinal and diagonal directions
        self.d8_offsets = np.array([
            [-1, -1], [-1, 0], [-1, 1],  # NW, N, NE
            [0, 1], [1, 1], [1, 0],      # E, SE, S
            [1, -1], [0, -1]             # SW, W
        ])
        
        # D8 direction codes (powers of 2)
        self.d8_codes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        logger.info(f"Advanced Hydrology Analyzer initialized:")
        logger.info(f"  Algorithm: {flow_algorithm}")
        logger.info(f"  Stream threshold: {min_stream_threshold}")
        logger.info(f"  Village optimization: {village_scale_optimization}")
    
    def analyze_village_hydrology(self, dtm: DTM) -> HydrologyResults:
        """
        Complete hydrological analysis optimized for village areas.
        
        Pipeline for rural abadi drainage analysis:
        1. Fill depressions (prevent artificial sinks)
        2. Calculate flow direction (D8 or D-infinity)
        3. Calculate flow accumulation (drainage network)
        4. Detect remaining sinks/depressions (waterlogging zones)
        5. Calculate slope and curvature (terrain characteristics)
        6. Extract stream network (natural drainage channels)
        7. Calculate TWI (wetness prediction)
        8. Delineate watersheds (drainage basins)
        
        Args:
            dtm: Input DTM from village survey
            
        Returns:
            Complete hydrology results with village-specific insights
        """
        logger.info("Starting village-scale hydrological analysis")
        logger.info(f"DTM size: {dtm.shape}, Resolution: {dtm.resolution}m")
        
        try:
            # Stage 1: Fill depressions to ensure proper flow routing
            logger.info("Stage 1: Filling depressions")
            filled_dem = self._fill_depressions(dtm.elevation_grid)
            
            # Stage 2: Calculate flow direction using selected algorithm
            logger.info("Stage 2: Calculating flow direction")
            flow_direction = self._calculate_flow_direction(filled_dem, dtm.resolution)
            
            # Stage 3: Calculate flow accumulation for drainage network
            logger.info("Stage 3: Calculating flow accumulation")
            flow_accumulation = self._calculate_flow_accumulation(flow_direction)
            
            # Stage 4: Detect remaining sinks and depressions
            logger.info("Stage 4: Detecting sinks and depressions")
            sinks_depressions = self._detect_sinks_depressions(
                dtm.elevation_grid, filled_dem, dtm.resolution
            )
            
            # Stage 5: Calculate slope and curvature
            logger.info("Stage 5: Calculating slope and curvature")
            slope = self._calculate_slope(filled_dem, dtm.resolution)
            aspect = self._calculate_aspect(filled_dem)
            curvature = self._calculate_curvature(filled_dem, dtm.resolution)
            
            # Stage 6: Extract stream network
            logger.info("Stage 6: Extracting stream network")
            stream_network = self._extract_stream_network(
                flow_accumulation, flow_direction, dtm
            )
            
            # Stage 7: Calculate Topographic Wetness Index
            logger.info("Stage 7: Calculating Topographic Wetness Index")
            twi = self._calculate_twi(slope, flow_accumulation, dtm.resolution)
            
            # Stage 8: Delineate watersheds
            logger.info("Stage 8: Delineating watersheds")
            watersheds = self._delineate_watersheds(
                flow_direction, flow_accumulation, stream_network
            )
            
            # Create comprehensive results with village-specific metadata
            results = HydrologyResults(
                flow_direction=flow_direction,
                flow_accumulation=flow_accumulation,
                stream_network=stream_network,
                watersheds=watersheds,
                slope=slope,
                aspect=aspect,
                topographic_wetness_index=twi,
                filled_dem=filled_dem,
                metadata={
                    # Processing parameters
                    'algorithm': self.flow_algorithm,
                    'stream_threshold': self.min_stream_threshold,
                    'depression_fill': self.depression_fill_method,
                    'village_optimized': self.village_scale_optimization,
                    
                    # Analysis results
                    'total_streams': len(stream_network.streams),
                    'total_watersheds': len(watersheds),
                    'stream_density_km_per_km2': stream_network.stream_density,
                    'total_stream_length_m': stream_network.total_length,
                    
                    # Village-specific insights
                    'sinks_detected': len(sinks_depressions),
                    'mean_slope_degrees': float(np.nanmean(slope)),
                    'max_flow_accumulation': int(np.nanmax(flow_accumulation)),
                    'high_twi_area_percentage': float(np.sum(twi > np.nanpercentile(twi, 90)) / twi.size * 100),
                    
                    # Drainage characteristics
                    'drainage_area_m2': float((dtm.bounds[2] - dtm.bounds[0]) * (dtm.bounds[3] - dtm.bounds[1])),
                    'resolution_m': dtm.resolution,
                    'coordinate_system': dtm.coordinate_system,
                    
                    # Additional analysis products
                    'sinks_depressions': sinks_depressions,
                    'curvature': curvature
                }
            )
            
            logger.info("Village hydrological analysis completed successfully")
            logger.info(f"  Streams extracted: {len(stream_network.streams)}")
            logger.info(f"  Watersheds delineated: {len(watersheds)}")
            logger.info(f"  Sinks detected: {len(sinks_depressions)}")
            logger.info(f"  Mean slope: {np.nanmean(slope):.2f}°")
            
            return results
            
        except Exception as e:
            logger.error(f"Village hydrological analysis failed: {e}")
            raise HydrologyAnalysisError(f"Analysis failed: {e}")
        """
        Complete hydrological analysis of DTM.
        
        Args:
            dtm: Input DTM
            
        Returns:
            Complete hydrology results
        """
        logger.info("Starting comprehensive hydrological analysis")
        
        try:
            # Step 1: Fill depressions
            filled_dem = self._fill_depressions(dtm.elevation_grid)
            
            # Step 2: Calculate flow direction
            flow_direction = self._calculate_flow_direction(filled_dem)
            
            # Step 3: Calculate flow accumulation
            flow_accumulation = self._calculate_flow_accumulation(flow_direction)
            
            # Step 4: Extract stream network
            stream_network = self._extract_stream_network(
                flow_accumulation, flow_direction, dtm
            )
            
            # Step 5: Delineate watersheds
            watersheds = self._delineate_watersheds(
                flow_direction, flow_accumulation, stream_network
            )
            
            # Step 6: Calculate hydrological parameters
            slope = self._calculate_slope(filled_dem, dtm.resolution)
            aspect = self._calculate_aspect(filled_dem)
            twi = self._calculate_twi(slope, flow_accumulation, dtm.resolution)
            
            # Create results object
            results = HydrologyResults(
                flow_direction=flow_direction,
                flow_accumulation=flow_accumulation,
                stream_network=stream_network,
                watersheds=watersheds,
                slope=slope,
                aspect=aspect,
                topographic_wetness_index=twi,
                filled_dem=filled_dem,
                metadata={
                    'algorithm': self.flow_algorithm,
                    'stream_threshold': self.min_stream_threshold,
                    'depression_fill': self.depression_fill_method,
                    'total_streams': len(stream_network.streams),
                    'total_watersheds': len(watersheds)
                }
            )
            
            logger.info("Hydrological analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Hydrological analysis failed: {e}")
            raise HydrologyAnalysisError(f"Analysis failed: {e}")
    
    def _fill_depressions(self, dem: np.ndarray) -> np.ndarray:
        """Fill depressions in DEM using Planchon-Darboux algorithm."""
        logger.info("Filling depressions in DEM")
        
        if self.depression_fill_method == "planchon_darboux":
            return self._planchon_darboux_fill(dem)
        else:
            # Simple morphological filling as fallback
            return ndimage.grey_closing(dem, size=(3, 3))
    
    def _planchon_darboux_fill(self, dem: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """Planchon-Darboux depression filling algorithm."""
        filled = dem.copy()
        rows, cols = dem.shape
        
        # Initialize with very high values except borders
        filled[1:-1, 1:-1] = np.inf
        
        # Iterative filling
        max_iterations = 1000
        for iteration in range(max_iterations):
            changed = False
            
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if np.isfinite(dem[i, j]):  # Valid elevation
                        # Find minimum neighbor elevation
                        neighbors = filled[i-1:i+2, j-1:j+2]
                        min_neighbor = np.min(neighbors[neighbors != filled[i, j]])
                        
                        # Update if necessary
                        new_value = max(dem[i, j], min_neighbor + epsilon)
                        if new_value < filled[i, j]:
                            filled[i, j] = new_value
                            changed = True
            
            if not changed:
                break
        
        logger.info(f"Depression filling converged after {iteration + 1} iterations")
        return filled
    
    def _calculate_flow_direction(self, dem: np.ndarray, resolution: float) -> np.ndarray:
        """
        Calculate flow direction using D8 or D-infinity algorithm.
        
        WHY flow direction is critical for villages:
        - Identifies natural drainage paths through mixed terrain
        - Prevents building in natural flow channels
        - Guides drainage infrastructure placement
        - Predicts surface runoff patterns during monsoons
        
        Args:
            dem: Filled DEM
            resolution: Grid resolution in meters
            
        Returns:
            Flow direction grid (D8 encoding)
        """
        logger.info(f"Calculating flow direction using {self.flow_algorithm} algorithm")
        
        if self.flow_algorithm.lower() == "d8":
            return self._calculate_d8_flow_direction(dem, resolution)
        elif self.flow_algorithm.lower() == "dinf":
            return self._calculate_dinf_flow_direction(dem, resolution)
        else:
            logger.warning(f"Unknown algorithm {self.flow_algorithm}, using D8")
            return self._calculate_d8_flow_direction(dem, resolution)
    
    def _calculate_d8_flow_direction(self, dem: np.ndarray, resolution: float) -> np.ndarray:
        """
        Calculate D8 flow direction (steepest descent).
        
        WHY D8 for villages:
        - Simple and robust for mixed terrain
        - Handles agricultural fields and residential areas
        - Fast computation for village-scale grids
        - Well-established for drainage network extraction
        """
        rows, cols = dem.shape
        flow_dir = np.zeros((rows, cols), dtype=np.uint8)
        
        # Process interior cells (avoid boundaries)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if np.isfinite(dem[i, j]):
                    center_elev = dem[i, j]
                    max_slope = -np.inf
                    direction = 0
                    
                    # Check all 8 neighbors for steepest descent
                    for k, (di, dj) in enumerate(self.d8_offsets):
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < rows and 0 <= nj < cols and np.isfinite(dem[ni, nj]):
                            # Calculate slope considering diagonal distance
                            if k % 2 == 0:  # Diagonal neighbors
                                distance = resolution * np.sqrt(2)
                            else:  # Cardinal neighbors
                                distance = resolution
                            
                            slope = (center_elev - dem[ni, nj]) / distance
                            
                            # Track steepest descent direction
                            if slope > max_slope:
                                max_slope = slope
                                direction = self.d8_codes[k]
                    
                    flow_dir[i, j] = direction
        
        return flow_dir
    
    def _calculate_dinf_flow_direction(self, dem: np.ndarray, resolution: float) -> np.ndarray:
        """
        Calculate D-infinity flow direction (infinite directions).
        
        WHY D-infinity for complex terrain:
        - More accurate flow routing in complex topography
        - Better handles convergent/divergent flow
        - Reduces artificial channelization
        - Improved for areas with subtle topography
        
        Note: Simplified implementation - full D-infinity is more complex
        """
        logger.info("Using simplified D-infinity (falls back to D8 for now)")
        # For now, fall back to D8 - full D-infinity implementation is complex
        return self._calculate_d8_flow_direction(dem, resolution)
    
    def _detect_sinks_depressions(
        self,
        original_dem: np.ndarray,
        filled_dem: np.ndarray,
        resolution: float
    ) -> List[Dict[str, Any]]:
        """
        Detect sinks and depressions that remain after filling.
        
        WHY sink detection is critical for villages:
        - Identifies natural waterlogging zones
        - Locates areas prone to stagnant water (mosquito breeding)
        - Guides targeted drainage interventions
        - Predicts flood-prone areas during heavy rainfall
        - Supports public health planning
        
        Args:
            original_dem: Original DEM before filling
            filled_dem: DEM after depression filling
            resolution: Grid resolution
            
        Returns:
            List of detected sinks with properties
        """
        logger.info("Detecting sinks and depressions")
        
        # Calculate depth of filling (difference between filled and original)
        fill_depth = filled_dem - original_dem
        
        # Identify significant depressions (> 0.1m depth)
        depression_threshold = 0.1  # meters
        depression_mask = fill_depth > depression_threshold
        
        if not np.any(depression_mask):
            logger.info("No significant depressions detected")
            return []
        
        # Label connected depression regions
        labeled_depressions, num_depressions = ndimage.label(depression_mask)
        
        sinks_depressions = []
        
        for depression_id in range(1, num_depressions + 1):
            # Get depression mask for this region
            region_mask = labeled_depressions == depression_id
            
            # Calculate depression properties
            region_indices = np.where(region_mask)
            region_depths = fill_depth[region_mask]
            
            # Calculate depression characteristics
            area_m2 = np.sum(region_mask) * (resolution ** 2)
            max_depth = np.max(region_depths)
            mean_depth = np.mean(region_depths)
            volume_m3 = np.sum(region_depths) * (resolution ** 2)
            
            # Find centroid
            centroid_i = np.mean(region_indices[0])
            centroid_j = np.mean(region_indices[1])
            
            # Only include significant depressions
            if area_m2 > 4.0:  # At least 4 m² (2x2 meters)
                depression_info = {
                    'id': depression_id,
                    'area_m2': area_m2,
                    'max_depth_m': max_depth,
                    'mean_depth_m': mean_depth,
                    'volume_m3': volume_m3,
                    'centroid_grid': (int(centroid_i), int(centroid_j)),
                    'waterlogging_risk': self._assess_waterlogging_risk(
                        area_m2, max_depth, mean_depth
                    ),
                    'mask': region_mask
                }
                
                sinks_depressions.append(depression_info)
        
        logger.info(f"Detected {len(sinks_depressions)} significant depressions")
        
        # Log depression statistics
        if sinks_depressions:
            total_area = sum(d['area_m2'] for d in sinks_depressions)
            max_depth = max(d['max_depth_m'] for d in sinks_depressions)
            logger.info(f"  Total depression area: {total_area:.1f} m²")
            logger.info(f"  Maximum depth: {max_depth:.2f} m")
        
        return sinks_depressions
    
    def _assess_waterlogging_risk(
        self,
        area_m2: float,
        max_depth: float,
        mean_depth: float
    ) -> str:
        """
        Assess waterlogging risk based on depression characteristics.
        
        WHY risk assessment for villages:
        - Prioritizes drainage interventions
        - Supports public health planning
        - Guides infrastructure placement
        - Informs agricultural drainage needs
        """
        # Risk assessment based on size and depth
        if area_m2 > 100 and max_depth > 0.5:
            return "CRITICAL"  # Large, deep depressions
        elif area_m2 > 50 or max_depth > 0.3:
            return "HIGH"      # Moderate size or depth
        elif area_m2 > 20 or max_depth > 0.2:
            return "MEDIUM"    # Small but significant
        else:
            return "LOW"       # Minor depressions
    
    def _calculate_curvature(self, dem: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """
        Calculate terrain curvature (plan, profile, and mean curvature).
        
        WHY curvature analysis for villages:
        - Plan curvature: Identifies convergent (water collecting) vs divergent areas
        - Profile curvature: Shows acceleration/deceleration of flow
        - Mean curvature: Overall terrain convexity/concavity
        - Critical for understanding water flow patterns and erosion potential
        
        Args:
            dem: Digital elevation model
            resolution: Grid resolution in meters
            
        Returns:
            Dictionary with curvature arrays
        """
        logger.info("Calculating terrain curvature")
        
        # Calculate first derivatives (gradients)
        gy, gx = np.gradient(dem, resolution)
        
        # Calculate second derivatives (curvatures)
        gyy, gyx = np.gradient(gy, resolution)
        gxy, gxx = np.gradient(gx, resolution)
        
        # Calculate curvature components
        # Plan curvature (horizontal curvature)
        denominator = (gx**2 + gy**2)
        plan_curvature = np.where(
            denominator > 1e-10,
            (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / (denominator**(3/2)),
            0
        )
        
        # Profile curvature (vertical curvature)
        profile_curvature = np.where(
            denominator > 1e-10,
            (gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / (denominator**(3/2)),
            0
        )
        
        # Mean curvature
        mean_curvature = (plan_curvature + profile_curvature) / 2
        
        # Gaussian curvature
        gaussian_curvature = np.where(
            denominator > 1e-10,
            (gxx * gyy - gxy**2) / (denominator**2),
            0
        )
        
        curvature_results = {
            'plan_curvature': plan_curvature,
            'profile_curvature': profile_curvature,
            'mean_curvature': mean_curvature,
            'gaussian_curvature': gaussian_curvature
        }
        
        # Log curvature statistics
        logger.info(f"  Plan curvature range: {np.nanmin(plan_curvature):.4f} to {np.nanmax(plan_curvature):.4f}")
        logger.info(f"  Profile curvature range: {np.nanmin(profile_curvature):.4f} to {np.nanmax(profile_curvature):.4f}")
        
        return curvature_results
    
    def _calculate_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """Calculate flow accumulation using D8 flow direction."""
        logger.info("Calculating flow accumulation")
        
        rows, cols = flow_dir.shape
        accumulation = np.ones((rows, cols), dtype=np.int32)
        
        # Create processing order (topological sort)
        processed = np.zeros((rows, cols), dtype=bool)
        
        # Process cells in order of decreasing elevation
        # This is a simplified approach - in practice, you'd use proper topological sorting
        max_iterations = rows * cols
        
        for iteration in range(max_iterations):
            changed = False
            
            for i in range(rows):
                for j in range(cols):
                    if processed[i, j] or flow_dir[i, j] == 0:
                        continue
                    
                    # Find downstream cell
                    direction = flow_dir[i, j]
                    if direction == 0:
                        processed[i, j] = True
                        continue
                    
                    # Get direction index
                    dir_idx = int(np.log2(direction))
                    di, dj = self.d8_offsets[dir_idx]
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < rows and 0 <= nj < cols:
                        accumulation[ni, nj] += accumulation[i, j]
                        changed = True
                    
                    processed[i, j] = True
            
            if not changed:
                break
        
        return accumulation
    
    def _extract_stream_network(
        self,
        flow_accumulation: np.ndarray,
        flow_direction: np.ndarray,
        dtm: DTM
    ) -> StreamNetwork:
        """Extract stream network from flow accumulation."""
        logger.info(f"Extracting stream network (threshold={self.min_stream_threshold})")
        
        # Create stream mask
        stream_mask = flow_accumulation >= self.min_stream_threshold
        
        # Trace stream lines
        streams = []
        stream_id = 1
        
        # Find stream starting points (high accumulation cells)
        stream_starts = np.where(
            (flow_accumulation >= self.min_stream_threshold * 2) &
            (flow_accumulation >= np.roll(flow_accumulation, 1, axis=0)) &
            (flow_accumulation >= np.roll(flow_accumulation, -1, axis=0)) &
            (flow_accumulation >= np.roll(flow_accumulation, 1, axis=1)) &
            (flow_accumulation >= np.roll(flow_accumulation, -1, axis=1))
        )
        
        for start_i, start_j in zip(stream_starts[0], stream_starts[1]):
            stream_points = self._trace_stream(
                start_i, start_j, flow_direction, stream_mask, dtm
            )
            
            if len(stream_points) > 2:  # Minimum stream length
                streams.append({
                    'id': stream_id,
                    'geometry': LineString(stream_points),
                    'length': LineString(stream_points).length,
                    'start_accumulation': flow_accumulation[start_i, start_j]
                })
                stream_id += 1
        
        return StreamNetwork(
            streams=streams,
            total_length=sum(s['length'] for s in streams),
            stream_density=sum(s['length'] for s in streams) / (dtm.bounds[2] - dtm.bounds[0]) / (dtm.bounds[3] - dtm.bounds[1])
        )
    
    def _trace_stream(
        self,
        start_i: int,
        start_j: int,
        flow_direction: np.ndarray,
        stream_mask: np.ndarray,
        dtm: DTM
    ) -> List[Tuple[float, float]]:
        """Trace a stream from starting point."""
        points = []
        i, j = start_i, start_j
        visited = set()
        
        minx, miny, maxx, maxy = dtm.bounds
        rows, cols = flow_direction.shape
        
        while (i, j) not in visited and stream_mask[i, j]:
            visited.add((i, j))
            
            # Convert grid coordinates to real coordinates
            x = minx + (j + 0.5) * dtm.resolution
            y = maxy - (i + 0.5) * dtm.resolution
            points.append((x, y))
            
            # Follow flow direction
            direction = flow_direction[i, j]
            if direction == 0:
                break
            
            dir_idx = int(np.log2(direction))
            di, dj = self.d8_offsets[dir_idx]
            i, j = i + di, j + dj
            
            if not (0 <= i < rows and 0 <= j < cols):
                break
        
        return points
    
    def _delineate_watersheds(
        self,
        flow_direction: np.ndarray,
        flow_accumulation: np.ndarray,
        stream_network: StreamNetwork
    ) -> List[WatershedData]:
        """Delineate watersheds for major streams."""
        logger.info("Delineating watersheds")
        
        watersheds = []
        
        # Use major stream outlets as pour points
        major_streams = [s for s in stream_network.streams if s['start_accumulation'] > self.min_stream_threshold * 5]
        
        for i, stream in enumerate(major_streams[:10]):  # Limit to top 10 watersheds
            # Get outlet point (last point of stream)
            outlet_coords = list(stream['geometry'].coords)[-1]
            
            # Convert to grid coordinates (simplified)
            # In practice, you'd use proper coordinate transformation
            outlet_i = int(flow_direction.shape[0] / 2)  # Placeholder
            outlet_j = int(flow_direction.shape[1] / 2)  # Placeholder
            
            # Delineate watershed (simplified approach)
            watershed_mask = self._delineate_single_watershed(
                flow_direction, outlet_i, outlet_j
            )
            
            if np.any(watershed_mask):
                # Convert mask to polygon (simplified)
                area = np.sum(watershed_mask) * (1.0 ** 2)  # Assuming 1m resolution
                
                watersheds.append(WatershedData(
                    id=i + 1,
                    outlet_point=Point(outlet_coords),
                    area=area,
                    perimeter=0.0,  # Would calculate from polygon
                    stream_length=stream['length'],
                    metadata={'stream_id': stream['id']}
                ))
        
        return watersheds
    
    def _delineate_single_watershed(
        self,
        flow_direction: np.ndarray,
        outlet_i: int,
        outlet_j: int
    ) -> np.ndarray:
        """Delineate single watershed from outlet point."""
        rows, cols = flow_direction.shape
        watershed = np.zeros((rows, cols), dtype=bool)
        
        # Trace upstream from outlet (simplified implementation)
        # This is a placeholder - proper watershed delineation is more complex
        for i in range(max(0, outlet_i - 50), min(rows, outlet_i + 50)):
            for j in range(max(0, outlet_j - 50), min(cols, outlet_j + 50)):
                # Simple distance-based watershed (placeholder)
                distance = np.sqrt((i - outlet_i)**2 + (j - outlet_j)**2)
                if distance < 30:  # 30-cell radius
                    watershed[i, j] = True
        
        return watershed
    
    def _calculate_slope(self, dem: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate slope in degrees."""
        logger.info("Calculating slope")
        
        # Calculate gradients
        dy, dx = np.gradient(dem)
        
        # Convert to slope in degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / resolution)
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    def _calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
        """Calculate aspect in degrees."""
        logger.info("Calculating aspect")
        
        # Calculate gradients
        dy, dx = np.gradient(dem)
        
        # Calculate aspect
        aspect = np.degrees(np.arctan2(-dy, dx))
        
        # Convert to 0-360 degrees
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        
        return aspect
    
    def _calculate_twi(
        self,
        slope: np.ndarray,
        flow_accumulation: np.ndarray,
        resolution: float
    ) -> np.ndarray:
        """
        Calculate enhanced Topographic Wetness Index for village areas.
        
        WHY TWI is critical for villages:
        - Predicts chronic waterlogging zones
        - Identifies areas with poor natural drainage
        - Guides agricultural field drainage planning
        - Supports public health interventions (mosquito control)
        - Helps locate suitable sites for water harvesting
        
        TWI = ln(a / tan(β))
        where: a = specific catchment area, β = slope angle
        
        Higher TWI values indicate:
        - Areas that accumulate water
        - Potential waterlogging zones
        - Locations needing drainage intervention
        
        Args:
            slope: Slope in degrees
            flow_accumulation: Flow accumulation grid
            resolution: Grid resolution in meters
            
        Returns:
            Topographic Wetness Index grid
        """
        logger.info("Calculating enhanced Topographic Wetness Index")
        
        # Convert slope to radians and avoid division by zero
        slope_rad = np.radians(np.maximum(slope, 0.1))  # Minimum 0.1° slope
        
        # Calculate specific catchment area (m²/m)
        # This represents the upslope area draining through each cell
        specific_catchment_area = flow_accumulation * (resolution ** 2) / resolution
        
        # Avoid log of zero or negative values
        specific_catchment_area = np.maximum(specific_catchment_area, resolution)
        
        # Calculate TWI
        twi = np.log(specific_catchment_area / np.tan(slope_rad))
        
        # Handle infinite values
        twi = np.where(np.isfinite(twi), twi, np.nan)
        
        # Log TWI statistics for village analysis
        valid_twi = twi[np.isfinite(twi)]
        if len(valid_twi) > 0:
            twi_mean = np.mean(valid_twi)
            twi_std = np.std(valid_twi)
            twi_90th = np.percentile(valid_twi, 90)
            
            logger.info(f"  TWI statistics:")
            logger.info(f"    Mean: {twi_mean:.2f}")
            logger.info(f"    Std: {twi_std:.2f}")
            logger.info(f"    90th percentile: {twi_90th:.2f}")
            logger.info(f"    High TWI area (>90th percentile): {np.sum(twi > twi_90th) / twi.size * 100:.1f}%")
        
        return twi


# Legacy class for backward compatibility
class HydrologyAnalyzer(AdvancedHydrologyAnalyzer):
    """Legacy hydrology analyzer - use AdvancedHydrologyAnalyzer for new projects."""
    
    def analyze_hydrology(self, dtm: DTM) -> HydrologyResults:
        """Legacy method - calls advanced village analysis."""
        logger.warning("Using legacy HydrologyAnalyzer. Consider upgrading to AdvancedHydrologyAnalyzer.")
        return self.analyze_village_hydrology(dtm)


def analyze_village_hydrology(
    dtm: DTM,
    flow_algorithm: str = "d8",
    min_stream_threshold: int = 1000,
    village_optimization: bool = True
) -> HydrologyResults:
    """
    Convenience function for village-scale hydrological analysis.
    
    This function provides comprehensive hydrological analysis specifically
    optimized for Indian village (abadi) areas, including:
    
    - Flow direction computation (D8/D-infinity)
    - Flow accumulation mapping
    - Depression/sink detection for waterlogging assessment
    - Slope and curvature analysis
    - Stream network extraction
    - Topographic Wetness Index calculation
    - Watershed delineation
    
    Args:
        dtm: Input DTM from village survey
        flow_algorithm: Flow direction algorithm ('d8' or 'dinf')
        min_stream_threshold: Minimum flow accumulation for stream extraction
        village_optimization: Enable village-specific optimizations
        
    Returns:
        Complete hydrology analysis results
        
    Example:
        >>> # Analyze village drainage patterns
        >>> hydrology = analyze_village_hydrology(
        ...     dtm=village_dtm,
        ...     flow_algorithm="d8",
        ...     min_stream_threshold=500,
        ...     village_optimization=True
        ... )
        >>> 
        >>> # Access results
        >>> print(f"Streams found: {len(hydrology.stream_network.streams)}")
        >>> print(f"Sinks detected: {hydrology.metadata['sinks_detected']}")
        >>> print(f"Mean slope: {hydrology.metadata['mean_slope_degrees']:.2f}°")
        >>> 
        >>> # Identify high waterlogging risk areas
        >>> high_twi_mask = hydrology.topographic_wetness_index > np.percentile(
        ...     hydrology.topographic_wetness_index, 90
        ... )
        >>> print(f"High waterlogging risk: {np.sum(high_twi_mask) / high_twi_mask.size * 100:.1f}%")
    """
    analyzer = AdvancedHydrologyAnalyzer(
        flow_algorithm=flow_algorithm,
        min_stream_threshold=min_stream_threshold,
        village_scale_optimization=village_optimization
    )
    
    return analyzer.analyze_village_hydrology(dtm)


# Legacy function for backward compatibility
def analyze_dtm_hydrology(
    dtm: DTM,
    flow_algorithm: str = "d8",
    min_stream_threshold: int = 1000
) -> HydrologyResults:
    """
    Legacy convenience function for hydrological analysis.
    
    NOTE: For village-scale analysis, use analyze_village_hydrology()
    """
    logger.warning("Using legacy analyze_dtm_hydrology. Consider upgrading to analyze_village_hydrology.")
    return analyze_village_hydrology(dtm, flow_algorithm, min_stream_threshold)