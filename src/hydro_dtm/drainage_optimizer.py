"""
Automated Drainage Network Design Module.

Features:
- Extract natural channels from flow accumulation
- Optimize drainage paths using graph-based routing
- Minimize excavation length while maximizing coverage
- Avoid buildings, roads, and sensitive zones
- Estimate drainage capacity based on rainfall & catchment area
- Output GIS-ready polylines with attributes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, field
import random
import heapq
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.ndimage import label, binary_dilation
from shapely.geometry import LineString, Point, Polygon, MultiLineString
from shapely.ops import unary_union, linemerge
import geopandas as gpd
import networkx as nx
from skimage.morphology import skeletonize
from skimage.measure import regionprops

from .models import DTM, HydrologyResults, WaterloggingRisk, DrainageNetwork
from .exceptions import DrainageOptimizationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DrainageSegment:
    """Individual drainage segment with comprehensive attributes."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    geometry: LineString
    diameter: float
    depth: float
    slope: float
    length: float
    cost: float
    capacity: float
    segment_type: str  # 'natural', 'constructed', 'enhanced'
    priority: int  # 1=critical, 2=high, 3=medium, 4=low
    catchment_area: float  # m²
    design_flow: float  # m³/s
    velocity: float  # m/s
    material: str  # 'concrete', 'hdpe', 'pvc', 'earthen'
    construction_method: str  # 'excavation', 'trenchless', 'enhancement'
    maintenance_access: bool
    environmental_impact: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NaturalChannel:
    """Natural drainage channel extracted from flow accumulation."""
    geometry: LineString
    flow_accumulation: float
    length: float
    average_slope: float
    catchment_area: float
    channel_order: int
    enhancement_potential: float  # 0-1 scale
    capacity: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintZone:
    """Constraint zone for drainage optimization."""
    geometry: Polygon
    zone_type: str  # 'building', 'road', 'sensitive', 'protected'
    restriction_level: str  # 'avoid', 'minimize', 'prohibit'
    cost_multiplier: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrainageNode:
    """Node in drainage network graph."""
    node_id: str
    location: Tuple[float, float]
    elevation: float
    node_type: str  # 'source', 'junction', 'outlet', 'treatment'
    upstream_area: float
    design_flow: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization results."""
    total_cost: float
    coverage_percentage: float
    environmental_impact: float
    hydraulic_efficiency: float
    maintenance_cost: float


class AutomatedDrainageDesigner:
    """
    Automated drainage network design with natural channel extraction,
    graph-based routing, and constraint handling.
    """
    
    def __init__(
        self,
        min_flow_accumulation: float = 1000,
        rainfall_intensity: float = 50.0,  # mm/hr for design storm
        design_return_period: int = 10,  # years
        manning_n: float = 0.013,
        safety_factor: float = 1.5
    ):
        """
        Initialize automated drainage designer.
        
        Args:
            min_flow_accumulation: Minimum flow accumulation for channel extraction
            rainfall_intensity: Design rainfall intensity (mm/hr)
            design_return_period: Design return period (years)
            manning_n: Manning's roughness coefficient
            safety_factor: Safety factor for capacity calculations
        """
        self.min_flow_accumulation = min_flow_accumulation
        self.rainfall_intensity = rainfall_intensity
        self.design_return_period = design_return_period
        self.manning_n = manning_n
        self.safety_factor = safety_factor
        
        # Cost parameters
        self.excavation_cost_per_m3 = 25.0  # USD
        self.pipe_costs = {  # USD per meter
            0.3: 30, 0.4: 45, 0.5: 65, 0.6: 90, 0.8: 130,
            1.0: 180, 1.2: 250, 1.5: 350, 2.0: 500, 2.5: 700
        }
        self.enhancement_cost_per_m = 50.0  # USD per meter for natural channel enhancement
        
        logger.info(f"Automated Drainage Designer initialized")
        logger.info(f"  Design storm: {rainfall_intensity} mm/hr, {design_return_period}-year return period")
    
    def design_drainage_network(
        self,
        dtm: DTM,
        hydrology: HydrologyResults,
        waterlogging_risk: WaterloggingRisk,
        constraint_zones: Optional[List[ConstraintZone]] = None,
        buildings: Optional[gpd.GeoDataFrame] = None,
        roads: Optional[gpd.GeoDataFrame] = None
    ) -> DrainageNetwork:
        """
        Design comprehensive drainage network.
        
        Args:
            dtm: Digital Terrain Model
            hydrology: Hydrological analysis results
            waterlogging_risk: Waterlogging risk assessment
            constraint_zones: Optional constraint zones
            buildings: Optional building polygons
            roads: Optional road polylines
            
        Returns:
            Optimized drainage network with GIS-ready attributes
        """
        logger.info("Starting automated drainage network design")
        
        try:
            # Step 1: Extract natural channels
            natural_channels = self._extract_natural_channels(dtm, hydrology)
            logger.info(f"Extracted {len(natural_channels)} natural channels")
            
            # Step 2: Identify critical drainage points
            critical_points = self._identify_critical_points(dtm, waterlogging_risk)
            logger.info(f"Identified {len(critical_points)} critical points")
            
            # Step 3: Create drainage network graph
            network_graph = self._create_network_graph(
                dtm, natural_channels, critical_points, constraint_zones, buildings, roads
            )
            logger.info(f"Created network graph with {network_graph.number_of_nodes()} nodes")
            
            # Step 4: Optimize drainage paths
            optimal_segments = self._optimize_drainage_paths(
                network_graph, dtm, hydrology, waterlogging_risk
            )
            logger.info(f"Optimized {len(optimal_segments)} drainage segments")
            
            # Step 5: Calculate hydraulic design
            designed_segments = self._calculate_hydraulic_design(optimal_segments, dtm)
            logger.info("Completed hydraulic design calculations")
            
            # Step 6: Create final drainage network
            drainage_network = self._create_drainage_network(designed_segments, dtm)
            
            logger.info(f"Drainage network design completed:")
            logger.info(f"  Total length: {drainage_network.total_length:.0f} m")
            logger.info(f"  Total cost: ${drainage_network.total_cost:,.0f}")
            logger.info(f"  Coverage: {drainage_network.coverage_area:.1f}%")
            
            return drainage_network
            
        except Exception as e:
            logger.error(f"Drainage network design failed: {e}")
            raise DrainageOptimizationError(f"Design failed: {e}")
    
    def _extract_natural_channels(
        self,
        dtm: DTM,
        hydrology: HydrologyResults
    ) -> List[NaturalChannel]:
        """Extract natural drainage channels from flow accumulation."""
        logger.info("Extracting natural drainage channels")
        
        # Create binary mask of high flow accumulation areas
        flow_acc = hydrology.flow_accumulation
        channel_mask = flow_acc >= self.min_flow_accumulation
        
        # Skeletonize to get channel centerlines
        skeleton = skeletonize(channel_mask)
        
        # Label connected components
        labeled_skeleton, num_features = label(skeleton)
        
        natural_channels = []
        
        for i in range(1, num_features + 1):
            # Extract individual channel
            channel_mask = labeled_skeleton == i
            channel_coords = np.where(channel_mask)
            
            if len(channel_coords[0]) < 3:  # Skip very short channels
                continue
            
            # Convert to real coordinates
            rows, cols = channel_coords
            minx, miny, maxx, maxy = dtm.bounds
            
            x_coords = minx + (cols + 0.5) * dtm.resolution
            y_coords = maxy - (rows + 0.5) * dtm.resolution
            
            # Order coordinates along channel (simplified)
            points = list(zip(x_coords, y_coords))
            
            # Create LineString
            if len(points) >= 2:
                try:
                    geometry = LineString(points)
                    
                    # Calculate channel properties
                    length = geometry.length
                    
                    # Average flow accumulation along channel
                    avg_flow_acc = np.mean([flow_acc[r, c] for r, c in zip(rows, cols)])
                    
                    # Calculate average slope
                    elevations = [dtm.elevation_grid[r, c] for r, c in zip(rows, cols)]
                    if len(elevations) > 1:
                        avg_slope = abs(elevations[-1] - elevations[0]) / length
                    else:
                        avg_slope = 0.001  # Default minimum slope
                    
                    # Estimate catchment area (simplified)
                    catchment_area = avg_flow_acc * dtm.resolution ** 2
                    
                    # Calculate channel order (simplified Strahler ordering)
                    channel_order = min(int(np.log10(avg_flow_acc / 100) + 1), 6)
                    
                    # Enhancement potential based on existing capacity vs needed capacity
                    current_capacity = self._estimate_natural_capacity(
                        avg_slope, length, channel_order
                    )
                    needed_capacity = self._calculate_design_flow(catchment_area)
                    enhancement_potential = min(1.0, needed_capacity / max(current_capacity, 0.1))
                    
                    channel = NaturalChannel(
                        geometry=geometry,
                        flow_accumulation=avg_flow_acc,
                        length=length,
                        average_slope=max(avg_slope, 0.001),
                        catchment_area=catchment_area,
                        channel_order=channel_order,
                        enhancement_potential=enhancement_potential,
                        capacity=current_capacity,
                        attributes={
                            'extraction_method': 'flow_accumulation_skeleton',
                            'min_elevation': min(elevations),
                            'max_elevation': max(elevations)
                        }
                    )
                    
                    natural_channels.append(channel)
                    
                except Exception as e:
                    logger.warning(f"Failed to create channel {i}: {e}")
                    continue
        
        return natural_channels
    
    def _create_network_graph(
        self,
        dtm: DTM,
        natural_channels: List[NaturalChannel],
        critical_points: List[Tuple[float, float]],
        constraint_zones: Optional[List[ConstraintZone]] = None,
        buildings: Optional[gpd.GeoDataFrame] = None,
        roads: Optional[gpd.GeoDataFrame] = None
    ) -> nx.DiGraph:
        """Create directed graph for drainage network optimization."""
        logger.info("Creating drainage network graph")
        
        G = nx.DiGraph()
        
        # Add nodes for critical points
        for i, (x, y) in enumerate(critical_points):
            elevation = self._get_elevation_at_point(dtm, x, y)
            upstream_area = self._estimate_upstream_area(dtm, x, y)
            design_flow = self._calculate_design_flow(upstream_area)
            
            G.add_node(
                f"critical_{i}",
                location=(x, y),
                elevation=elevation,
                node_type='source',
                upstream_area=upstream_area,
                design_flow=design_flow
            )
        
        # Add nodes for natural channel endpoints
        for i, channel in enumerate(natural_channels):
            start_coord = channel.geometry.coords[0]
            end_coord = channel.geometry.coords[-1]
            
            start_elev = self._get_elevation_at_point(dtm, start_coord[0], start_coord[1])
            end_elev = self._get_elevation_at_point(dtm, end_coord[0], end_coord[1])
            
            G.add_node(
                f"channel_start_{i}",
                location=start_coord,
                elevation=start_elev,
                node_type='junction',
                upstream_area=channel.catchment_area,
                design_flow=self._calculate_design_flow(channel.catchment_area)
            )
            
            G.add_node(
                f"channel_end_{i}",
                location=end_coord,
                elevation=end_elev,
                node_type='junction',
                upstream_area=channel.catchment_area,
                design_flow=self._calculate_design_flow(channel.catchment_area)
            )
        
        # Add outlet nodes (boundary low points)
        outlet_points = self._identify_outlet_points(dtm)
        for i, (x, y) in enumerate(outlet_points):
            elevation = self._get_elevation_at_point(dtm, x, y)
            G.add_node(
                f"outlet_{i}",
                location=(x, y),
                elevation=elevation,
                node_type='outlet',
                upstream_area=0,
                design_flow=0
            )
        
        # Add edges for natural channels
        for i, channel in enumerate(natural_channels):
            start_node = f"channel_start_{i}"
            end_node = f"channel_end_{i}"
            
            # Calculate cost for enhancing natural channel
            enhancement_cost = channel.length * self.enhancement_cost_per_m * channel.enhancement_potential
            
            G.add_edge(
                start_node,
                end_node,
                length=channel.length,
                cost=enhancement_cost,
                capacity=channel.capacity,
                segment_type='natural',
                geometry=channel.geometry,
                slope=channel.average_slope,
                enhancement_potential=channel.enhancement_potential
            )
        
        # Add edges between nearby nodes (potential constructed segments)
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if node1 == node2:
                    continue
                
                loc1 = G.nodes[node1]['location']
                loc2 = G.nodes[node2]['location']
                elev1 = G.nodes[node1]['elevation']
                elev2 = G.nodes[node2]['elevation']
                
                distance = np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
                
                # Only connect nearby nodes (within reasonable distance)
                if distance > 2000:  # 2km maximum
                    continue
                
                # Calculate slope
                slope = abs(elev2 - elev1) / distance if distance > 0 else 0.001
                
                # Check constraints
                cost_multiplier = self._calculate_constraint_cost_multiplier(
                    loc1, loc2, constraint_zones, buildings, roads
                )
                
                if cost_multiplier >= 10:  # Prohibitive cost
                    continue
                
                # Calculate construction cost
                base_cost = self._estimate_construction_cost(distance, slope)
                total_cost = base_cost * cost_multiplier
                
                # Add edge in both directions (will be filtered by elevation later)
                if elev1 > elev2:  # Downhill flow
                    G.add_edge(
                        node1, node2,
                        length=distance,
                        cost=total_cost,
                        capacity=0,  # Will be calculated later
                        segment_type='constructed',
                        geometry=LineString([loc1, loc2]),
                        slope=slope,
                        cost_multiplier=cost_multiplier
                    )
                elif elev2 > elev1:  # Uphill flow (reverse direction)
                    G.add_edge(
                        node2, node1,
                        length=distance,
                        cost=total_cost,
                        capacity=0,
                        segment_type='constructed',
                        geometry=LineString([loc2, loc1]),
                        slope=slope,
                        cost_multiplier=cost_multiplier
                    )
        
        return G
    
    def _optimize_drainage_paths(
        self,
        graph: nx.DiGraph,
        dtm: DTM,
        hydrology: HydrologyResults,
        waterlogging_risk: WaterloggingRisk
    ) -> List[DrainageSegment]:
        """Optimize drainage paths using graph-based routing."""
        logger.info("Optimizing drainage paths")
        
        optimal_segments = []
        
        # Find all source nodes (critical points)
        source_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'source']
        outlet_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'outlet']
        
        # For each source, find optimal path to nearest outlet
        for source in source_nodes:
            best_path = None
            best_cost = float('inf')
            
            for outlet in outlet_nodes:
                try:
                    # Find shortest path considering both distance and cost
                    path = nx.shortest_path(
                        graph, source, outlet, weight='cost'
                    )
                    
                    # Calculate total cost
                    total_cost = sum(
                        graph[path[i]][path[i+1]]['cost']
                        for i in range(len(path)-1)
                    )
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path
                        
                except nx.NetworkXNoPath:
                    continue
            
            # Convert path to drainage segments
            if best_path and len(best_path) > 1:
                for i in range(len(best_path) - 1):
                    node1, node2 = best_path[i], best_path[i+1]
                    edge_data = graph[node1][node2]
                    
                    # Get node data
                    node1_data = graph.nodes[node1]
                    node2_data = graph.nodes[node2]
                    
                    # Calculate priority based on risk
                    priority = self._calculate_segment_priority(
                        node1_data['location'], waterlogging_risk
                    )
                    
                    # Create drainage segment
                    segment = DrainageSegment(
                        start_point=node1_data['location'],
                        end_point=node2_data['location'],
                        geometry=edge_data['geometry'],
                        diameter=0.5,  # Will be calculated in hydraulic design
                        depth=1.0,     # Will be calculated in hydraulic design
                        slope=edge_data['slope'],
                        length=edge_data['length'],
                        cost=edge_data['cost'],
                        capacity=edge_data['capacity'],
                        segment_type=edge_data['segment_type'],
                        priority=priority,
                        catchment_area=node1_data['upstream_area'],
                        design_flow=node1_data['design_flow'],
                        velocity=0.0,  # Will be calculated
                        material='hdpe',  # Default material
                        construction_method='excavation',
                        maintenance_access=True,
                        environmental_impact=edge_data.get('cost_multiplier', 1.0),
                        attributes={
                            'source_node': node1,
                            'target_node': node2,
                            'path_index': i
                        }
                    )
                    
                    optimal_segments.append(segment)
        
        return optimal_segments
    
    def _calculate_hydraulic_design(
        self,
        segments: List[DrainageSegment],
        dtm: DTM
    ) -> List[DrainageSegment]:
        """Calculate hydraulic design parameters for drainage segments."""
        logger.info("Calculating hydraulic design parameters")
        
        designed_segments = []
        
        for segment in segments:
            # Calculate required diameter based on design flow
            required_diameter = self._calculate_required_diameter(
                segment.design_flow, segment.slope
            )
            
            # Select standard pipe diameter
            standard_diameter = self._select_standard_diameter(required_diameter)
            
            # Calculate depth based on terrain and minimum cover
            min_cover = 1.0  # meters
            start_elev = self._get_elevation_at_point(dtm, *segment.start_point)
            end_elev = self._get_elevation_at_point(dtm, *segment.end_point)
            
            # Calculate invert elevations
            start_invert = start_elev - min_cover - standard_diameter
            end_invert = end_elev - min_cover - standard_diameter
            
            # Ensure positive slope
            if start_invert <= end_invert:
                end_invert = start_invert - segment.length * max(segment.slope, 0.001)
            
            avg_depth = min_cover + standard_diameter + (start_elev + end_elev) / 2 - (start_invert + end_invert) / 2
            
            # Calculate actual capacity
            actual_capacity = self._calculate_pipe_capacity(
                standard_diameter, segment.slope, self.manning_n
            )
            
            # Calculate velocity
            area = np.pi * (standard_diameter / 2) ** 2
            velocity = segment.design_flow / area if area > 0 else 0
            
            # Update segment
            segment.diameter = standard_diameter
            segment.depth = max(avg_depth, min_cover + standard_diameter)
            segment.capacity = actual_capacity
            segment.velocity = velocity
            
            # Select material based on conditions
            segment.material = self._select_pipe_material(
                standard_diameter, segment.depth, segment.velocity
            )
            
            # Update cost with actual design
            segment.cost = self._calculate_detailed_cost(segment)
            
            designed_segments.append(segment)
        
        return designed_segments
    
    def _create_drainage_network(
        self,
        segments: List[DrainageSegment],
        dtm: DTM
    ) -> DrainageNetwork:
        """Create final drainage network with GIS-ready attributes."""
        logger.info("Creating final drainage network")
        
        # Calculate network statistics
        total_length = sum(seg.length for seg in segments)
        total_cost = sum(seg.cost for seg in segments)
        total_capacity = sum(seg.capacity for seg in segments)
        
        # Calculate coverage area (simplified)
        total_catchment = sum(seg.catchment_area for seg in segments)
        dtm_area = (dtm.bounds[2] - dtm.bounds[0]) * (dtm.bounds[3] - dtm.bounds[1])
        coverage_percentage = min(100, (total_catchment / dtm_area) * 100)
        
        # Create GIS-ready attributes for each segment
        gis_segments = []
        for i, seg in enumerate(segments):
            gis_attributes = {
                'segment_id': f"drain_{i:03d}",
                'length_m': round(seg.length, 2),
                'diameter_m': round(seg.diameter, 3),
                'depth_m': round(seg.depth, 2),
                'slope_percent': round(seg.slope * 100, 3),
                'capacity_m3s': round(seg.capacity, 3),
                'velocity_ms': round(seg.velocity, 2),
                'cost_usd': round(seg.cost, 0),
                'material': seg.material,
                'segment_type': seg.segment_type,
                'priority': seg.priority,
                'catchment_area_m2': round(seg.catchment_area, 0),
                'design_flow_m3s': round(seg.design_flow, 4),
                'construction_method': seg.construction_method,
                'maintenance_access': seg.maintenance_access,
                'environmental_impact': round(seg.environmental_impact, 2)
            }
            
            # Update segment attributes
            seg.attributes.update(gis_attributes)
            gis_segments.append(seg)
        
        # Create drainage network
        drainage_network = DrainageNetwork(
            segments=gis_segments,
            total_length=total_length,
            total_cost=total_cost,
            coverage_area=coverage_percentage,
            hydraulic_capacity=total_capacity,
            objectives=OptimizationObjectives(
                total_cost=total_cost,
                coverage_percentage=coverage_percentage,
                environmental_impact=sum(seg.environmental_impact for seg in segments),
                hydraulic_efficiency=min(100, total_capacity * 10),
                maintenance_cost=total_cost * 0.05
            ),
            metadata={
                'design_method': 'automated_graph_based',
                'design_storm_intensity': self.rainfall_intensity,
                'design_return_period': self.design_return_period,
                'total_segments': len(segments),
                'natural_channels': len([s for s in segments if s.segment_type == 'natural']),
                'constructed_segments': len([s for s in segments if s.segment_type == 'constructed']),
                'safety_factor': self.safety_factor
            }
        )
        
        return drainage_network
    
    def export_to_gis(
        self,
        drainage_network: DrainageNetwork,
        output_path: Path,
        crs: str = 'EPSG:4326'
    ) -> Dict[str, Path]:
        """Export drainage network to GIS formats."""
        logger.info(f"Exporting drainage network to GIS: {output_path}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        # Create GeoDataFrame for segments
        geometries = []
        attributes = []
        
        for segment in drainage_network.segments:
            geometries.append(segment.geometry)
            attributes.append(segment.attributes)
        
        # Create segments shapefile
        segments_gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
        segments_file = output_path / 'drainage_segments.shp'
        segments_gdf.to_file(segments_file)
        exported_files['segments'] = segments_file
        
        # Create nodes shapefile
        nodes_data = []
        node_geometries = []
        
        for i, segment in enumerate(drainage_network.segments):
            # Start node
            nodes_data.append({
                'node_id': f"node_{i}_start",
                'node_type': 'junction',
                'elevation_m': self._get_elevation_at_point(None, *segment.start_point),
                'segment_id': segment.attributes['segment_id']
            })
            node_geometries.append(Point(segment.start_point))
            
            # End node
            nodes_data.append({
                'node_id': f"node_{i}_end",
                'node_type': 'junction',
                'elevation_m': self._get_elevation_at_point(None, *segment.end_point),
                'segment_id': segment.attributes['segment_id']
            })
            node_geometries.append(Point(segment.end_point))
        
        nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry=node_geometries, crs=crs)
        nodes_file = output_path / 'drainage_nodes.shp'
        nodes_gdf.to_file(nodes_file)
        exported_files['nodes'] = nodes_file
        
        # Create summary report
        report_file = output_path / 'drainage_network_summary.txt'
        with open(report_file, 'w') as f:
            f.write("AUTOMATED DRAINAGE NETWORK DESIGN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Length: {drainage_network.total_length:.0f} m\n")
            f.write(f"Total Cost: ${drainage_network.total_cost:,.0f}\n")
            f.write(f"Coverage Area: {drainage_network.coverage_area:.1f}%\n")
            f.write(f"Total Capacity: {drainage_network.hydraulic_capacity:.2f} m³/s\n")
            f.write(f"Number of Segments: {len(drainage_network.segments)}\n\n")
            
            f.write("SEGMENT BREAKDOWN:\n")
            f.write("-" * 20 + "\n")
            for segment_type in ['natural', 'constructed', 'enhanced']:
                count = len([s for s in drainage_network.segments if s.segment_type == segment_type])
                if count > 0:
                    f.write(f"{segment_type.title()} segments: {count}\n")
            
            f.write(f"\nDesign Parameters:\n")
            f.write(f"- Design storm: {self.rainfall_intensity} mm/hr\n")
            f.write(f"- Return period: {self.design_return_period} years\n")
            f.write(f"- Safety factor: {self.safety_factor}\n")
        
        exported_files['summary'] = report_file
        
        logger.info(f"Exported {len(exported_files)} files to {output_path}")
        return exported_files
    
    # Helper methods
    def _identify_critical_points(
        self,
        dtm: DTM,
        waterlogging_risk: WaterloggingRisk
    ) -> List[Tuple[float, float]]:
        """Identify critical points requiring drainage."""
        critical_points = []
        rows, cols = dtm.elevation_grid.shape
        minx, miny, maxx, maxy = dtm.bounds
        
        # Find high-risk areas
        high_risk_mask = waterlogging_risk.risk_grid >= 2
        
        # Find local minima in high-risk areas
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if high_risk_mask[i, j]:
                    center_elev = dtm.elevation_grid[i, j]
                    neighbors = dtm.elevation_grid[i-1:i+2, j-1:j+2]
                    
                    if center_elev <= np.min(neighbors):
                        x = minx + (j + 0.5) * dtm.resolution
                        y = maxy - (i + 0.5) * dtm.resolution
                        critical_points.append((x, y))
        
        # Limit to top 30 points by risk
        if len(critical_points) > 30:
            risk_values = []
            for x, y in critical_points:
                i = int((maxy - y) / dtm.resolution)
                j = int((x - minx) / dtm.resolution)
                risk_values.append(waterlogging_risk.probability_grid[i, j])
            
            sorted_indices = np.argsort(risk_values)[::-1][:30]
            critical_points = [critical_points[i] for i in sorted_indices]
        
        return critical_points
    
    def _identify_outlet_points(self, dtm: DTM) -> List[Tuple[float, float]]:
        """Identify outlet points on domain boundaries."""
        outlet_points = []
        rows, cols = dtm.elevation_grid.shape
        minx, miny, maxx, maxy = dtm.bounds
        
        # Sample boundary points
        boundary_points = []
        
        # Sample boundaries
        for j in range(0, cols, 20):
            # Top boundary
            if np.isfinite(dtm.elevation_grid[0, j]):
                x = minx + (j + 0.5) * dtm.resolution
                y = maxy - 0.5 * dtm.resolution
                boundary_points.append((x, y, dtm.elevation_grid[0, j]))
            
            # Bottom boundary
            if np.isfinite(dtm.elevation_grid[rows-1, j]):
                x = minx + (j + 0.5) * dtm.resolution
                y = maxy - (rows - 0.5) * dtm.resolution
                boundary_points.append((x, y, dtm.elevation_grid[rows-1, j]))
        
        for i in range(0, rows, 20):
            # Left boundary
            if np.isfinite(dtm.elevation_grid[i, 0]):
                x = minx + 0.5 * dtm.resolution
                y = maxy - (i + 0.5) * dtm.resolution
                boundary_points.append((x, y, dtm.elevation_grid[i, 0]))
            
            # Right boundary
            if np.isfinite(dtm.elevation_grid[i, cols-1]):
                x = minx + (cols - 0.5) * dtm.resolution
                y = maxy - (i + 0.5) * dtm.resolution
                boundary_points.append((x, y, dtm.elevation_grid[i, cols-1]))
        
        # Select lowest points
        if boundary_points:
            boundary_points.sort(key=lambda p: p[2])
            outlet_points = [(p[0], p[1]) for p in boundary_points[:3]]
        
        return outlet_points
    
    def _get_elevation_at_point(self, dtm: DTM, x: float, y: float) -> float:
        """Get elevation at specific point."""
        if dtm is None:
            return 0.0
        
        minx, miny, maxx, maxy = dtm.bounds
        i = int((maxy - y) / dtm.resolution)
        j = int((x - minx) / dtm.resolution)
        
        if 0 <= i < dtm.elevation_grid.shape[0] and 0 <= j < dtm.elevation_grid.shape[1]:
            return float(dtm.elevation_grid[i, j])
        return 0.0
    
    def _estimate_upstream_area(self, dtm: DTM, x: float, y: float) -> float:
        """Estimate upstream catchment area."""
        # Simplified estimation based on local topography
        return 10000.0  # Default 1 hectare
    
    def _calculate_design_flow(self, catchment_area: float) -> float:
        """Calculate design flow using rational method."""
        # Q = C * I * A / 360 (where Q in m³/s, I in mm/hr, A in hectares)
        runoff_coefficient = 0.6  # Typical for mixed development
        area_hectares = catchment_area / 10000
        design_flow = (runoff_coefficient * self.rainfall_intensity * area_hectares) / 360
        return design_flow * self.safety_factor
    
    def _estimate_natural_capacity(self, slope: float, length: float, order: int) -> float:
        """Estimate capacity of natural channel."""
        # Simplified capacity estimation
        base_capacity = 0.1 * order  # m³/s per order
        slope_factor = min(2.0, slope * 1000)  # Slope enhancement
        return base_capacity * slope_factor
    
    def _calculate_constraint_cost_multiplier(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        constraint_zones: Optional[List[ConstraintZone]],
        buildings: Optional[gpd.GeoDataFrame],
        roads: Optional[gpd.GeoDataFrame]
    ) -> float:
        """Calculate cost multiplier based on constraints."""
        multiplier = 1.0
        
        # Create line geometry
        line = LineString([start_point, end_point])
        
        # Check constraint zones
        if constraint_zones:
            for zone in constraint_zones:
                if line.intersects(zone.geometry):
                    multiplier *= zone.cost_multiplier
        
        # Check buildings
        if buildings is not None:
            for _, building in buildings.iterrows():
                if line.intersects(building.geometry):
                    multiplier *= 5.0  # High cost for building conflicts
        
        # Check roads
        if roads is not None:
            for _, road in roads.iterrows():
                if line.intersects(road.geometry):
                    multiplier *= 2.0  # Moderate cost for road crossings
        
        return multiplier
    
    def _estimate_construction_cost(self, distance: float, slope: float) -> float:
        """Estimate construction cost for segment."""
        base_cost_per_m = 200.0  # USD per meter
        slope_multiplier = 1.0 + min(2.0, slope * 100)  # Increase cost for steep slopes
        return distance * base_cost_per_m * slope_multiplier
    
    def _calculate_segment_priority(
        self,
        location: Tuple[float, float],
        waterlogging_risk: WaterloggingRisk
    ) -> int:
        """Calculate segment priority based on risk."""
        # Get risk at location (simplified)
        return 2  # Default medium priority
    
    def _calculate_required_diameter(self, design_flow: float, slope: float) -> float:
        """Calculate required pipe diameter using Manning's equation."""
        # Solve Manning's equation for diameter
        # Q = (1/n) * A * R^(2/3) * S^(1/2)
        # For circular pipe: A = π*D²/4, R = D/4
        
        if design_flow <= 0 or slope <= 0:
            return 0.3  # Minimum diameter
        
        # Iterative solution
        for diameter in np.arange(0.3, 3.0, 0.1):
            capacity = self._calculate_pipe_capacity(diameter, slope, self.manning_n)
            if capacity >= design_flow:
                return diameter
        
        return 2.5  # Maximum diameter
    
    def _select_standard_diameter(self, required_diameter: float) -> float:
        """Select standard pipe diameter."""
        standard_sizes = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        return min(standard_sizes, key=lambda x: abs(x - required_diameter) if x >= required_diameter else float('inf'))
    
    def _calculate_pipe_capacity(self, diameter: float, slope: float, manning_n: float) -> float:
        """Calculate pipe capacity using Manning's equation."""
        area = np.pi * (diameter / 2) ** 2
        perimeter = np.pi * diameter
        hydraulic_radius = area / perimeter
        capacity = (1 / manning_n) * area * (hydraulic_radius ** (2/3)) * (slope ** 0.5)
        return capacity
    
    def _select_pipe_material(self, diameter: float, depth: float, velocity: float) -> str:
        """Select appropriate pipe material."""
        if diameter <= 0.6:
            return 'hdpe'
        elif diameter <= 1.2:
            return 'concrete'
        else:
            return 'reinforced_concrete'
    
    def _calculate_detailed_cost(self, segment: DrainageSegment) -> float:
        """Calculate detailed construction cost."""
        # Excavation cost
        excavation_volume = segment.length * segment.depth * 1.5  # 1.5m width
        excavation_cost = excavation_volume * self.excavation_cost_per_m3
        
        # Pipe cost
        pipe_cost_per_m = self.pipe_costs.get(segment.diameter, 300)
        pipe_cost = segment.length * pipe_cost_per_m
        
        # Installation and backfill
        installation_cost = pipe_cost * 0.3
        
        return excavation_cost + pipe_cost + installation_cost


# Legacy DrainageOptimizer class for backward compatibility
class DrainageOptimizer:
    """Legacy genetic algorithm-based drainage network optimization."""
    
    def __init__(self, **kwargs):
        """Initialize with automated designer."""
        self.designer = AutomatedDrainageDesigner()
        logger.info("Legacy DrainageOptimizer initialized (using AutomatedDrainageDesigner)")
    
    def optimize_drainage_network(
        self,
        dtm: DTM,
        hydrology: HydrologyResults,
        waterlogging_risk: WaterloggingRisk,
        constraints: Optional[Dict[str, Any]] = None
    ) -> DrainageNetwork:
        """Optimize using automated designer."""
        return self.designer.design_drainage_network(dtm, hydrology, waterlogging_risk)


def design_automated_drainage_network(
    dtm: DTM,
    hydrology: HydrologyResults,
    waterlogging_risk: WaterloggingRisk,
    constraint_zones: Optional[List[ConstraintZone]] = None,
    buildings: Optional[gpd.GeoDataFrame] = None,
    roads: Optional[gpd.GeoDataFrame] = None,
    rainfall_intensity: float = 50.0,
    design_return_period: int = 10
) -> DrainageNetwork:
    """
    Design automated drainage network with natural channel extraction
    and graph-based optimization.
    
    Args:
        dtm: Digital Terrain Model
        hydrology: Hydrological analysis results
        waterlogging_risk: Waterlogging risk assessment
        constraint_zones: Optional constraint zones
        buildings: Optional building polygons
        roads: Optional road polylines
        rainfall_intensity: Design rainfall intensity (mm/hr)
        design_return_period: Design return period (years)
        
    Returns:
        Optimized drainage network with GIS-ready attributes
    """
    designer = AutomatedDrainageDesigner(
        rainfall_intensity=rainfall_intensity,
        design_return_period=design_return_period
    )
    
    return designer.design_drainage_network(
        dtm, hydrology, waterlogging_risk, constraint_zones, buildings, roads
    )


def optimize_drainage_network(
    dtm: DTM,
    hydrology: HydrologyResults,
    waterlogging_risk: WaterloggingRisk,
    population_size: int = 50,
    max_generations: int = 100
) -> DrainageNetwork:
    """
    Legacy convenience function for drainage network optimization.
    Now uses AutomatedDrainageDesigner for better results.
    
    Args:
        dtm: Digital Terrain Model
        hydrology: Hydrological analysis results
        waterlogging_risk: Waterlogging risk assessment
        population_size: Ignored (legacy parameter)
        max_generations: Ignored (legacy parameter)
        
    Returns:
        Optimized drainage network
    """
    return design_automated_drainage_network(dtm, hydrology, waterlogging_risk)