#!/usr/bin/env python3
"""
Government-Friendly Visualization Demo for Intelligent Hydro-DTM System.

This demo creates professional visualizations suitable for government officials,
judges, and decision-makers using real ML waterlogging analysis results.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hydro_dtm.dtm_generator import DTMGenerator
from hydro_dtm.hydrology_analyzer import HydrologyAnalyzer
from hydro_dtm.waterlogging_predictor import WaterloggingPredictor
from hydro_dtm.drainage_optimizer import AutomatedDrainageDesigner
from hydro_dtm.government_visualizer import GovernmentVisualizer

def create_synthetic_terrain(size=80, resolution=5.0):
    """Create realistic synthetic terrain for demonstration."""
    print("🏔️  Generating realistic terrain data...")
    
    # Create coordinates
    x = np.linspace(0, size * resolution, size)
    y = np.linspace(0, size * resolution, size)
    X, Y = np.meshgrid(x, y)
    
    # Create realistic terrain with multiple features
    elevation = (
        100 +  # Base elevation
        30 * np.sin(X/50) * np.cos(Y/40) +  # Large hills
        15 * np.sin(X/25) * np.sin(Y/20) +  # Medium features
        8 * np.sin(X/12) * np.cos(Y/15) +   # Small features
        5 * np.random.random((size, size)) - 2.5  # Random noise
    )
    
    # Add some valleys (lower areas prone to flooding)
    valley_mask = ((X - size*resolution/2)**2 + (Y - size*resolution/3)**2) < (size*resolution/4)**2
    elevation[valley_mask] -= 15
    
    # Add a river valley
    river_mask = np.abs(Y - size*resolution/2) < 20
    elevation[river_mask] -= 8
    
    # Create point cloud (simplified)
    points = []
    for i in range(size):
        for j in range(size):
            # Add some noise to simulate real LiDAR data
            z = elevation[i, j] + np.random.normal(0, 0.1)
            points.append([x[j], y[i], z])
    
    points = np.array(points)
    
    return points, elevation, (size, size), resolution

def main():
    """Run the government visualization demo."""
    print("🏛️  Government-Friendly Visualization Demo")
    print("=" * 60)
    print("Creating professional visualizations for government officials")
    print("and decision-makers using ML-based waterlogging analysis.")
    print()
    
    # Parameters
    village_name = "Rampur Village"
    project_title = "Smart Flood Management System"
    
    # Step 1: Generate terrain
    points, elevation_grid, grid_shape, resolution = create_synthetic_terrain()
    
    # Step 2: Create DTM
    print("📐 Creating Digital Terrain Model...")
    dtm_generator = DTMGenerator(resolution=resolution, method='idw')
    
    # Create PointCloud object
    from hydro_dtm.models import PointCloud
    point_cloud = PointCloud(points=points)
    
    dtm = dtm_generator.generate_dtm(point_cloud)
    
    # Step 3: Analyze hydrology
    print("💧 Analyzing hydrology...")
    hydrology_analyzer = HydrologyAnalyzer()
    hydrology = hydrology_analyzer.analyze_village_hydrology(dtm)
    
    # Step 4: Predict waterlogging risk
    print("🤖 Predicting waterlogging risk using ML...")
    waterlogging_predictor = WaterloggingPredictor(model_type='ensemble')
    
    # Use the same approach as the working ML demo
    from ml_waterlogging_demo import MLWaterloggingSystem
    ml_system = MLWaterloggingSystem()
    
    # Create training data
    print("   📚 Creating training data...")
    training_features, training_labels = ml_system.create_synthetic_training_data(dtm, hydrology)
    
    # Train the model
    print("   🎯 Training ML model...")
    waterlogging_predictor.train_models(training_features, training_labels)
    
    # Make predictions
    print("   🔮 Making predictions...")
    waterlogging_risk = waterlogging_predictor.predict_waterlogging(dtm, hydrology)
    
    # Step 5: Design drainage network
    print("🚰 Designing automated drainage network...")
    drainage_designer = AutomatedDrainageDesigner()
    
    # Design drainage network
    drainage_network = drainage_designer.design_drainage_network(
        dtm=dtm,
        hydrology=hydrology,
        waterlogging_risk=waterlogging_risk,
        constraint_zones=None,  # No constraint zones in this demo
        buildings=None,         # No buildings in this demo
        roads=None             # No roads in this demo
    )
    
    # Step 6: Create government visualizations
    print("🎨 Creating government-friendly visualizations...")
    visualizer = GovernmentVisualizer(language='english')
    
    # Create output directory
    output_dir = Path("demo_outputs/government_presentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive report
    print("   📊 Generating comprehensive visual report...")
    generated_files = visualizer.create_comprehensive_report(
        dtm=dtm,
        hydrology=hydrology,
        waterlogging_risk=waterlogging_risk,
        drainage_network=drainage_network,
        output_dir=output_dir,
        village_name=village_name,
        project_title=project_title
    )
    
    # Step 7: Display results
    print("\n" + "=" * 60)
    print("✅ GOVERNMENT PRESENTATION READY!")
    print("=" * 60)
    
    print(f"📍 Village: {village_name}")
    print(f"📊 Project: {project_title}")
    print(f"📁 Output Directory: {output_dir}")
    print()
    
    # Analysis summary
    high_risk_pixels = np.sum(waterlogging_risk.risk_grid >= 2)
    total_pixels = waterlogging_risk.risk_grid.size
    high_risk_percent = (high_risk_pixels / total_pixels) * 100
    
    print("📈 ANALYSIS SUMMARY:")
    print(f"   • Total Area: {(grid_shape[0] * grid_shape[1] * resolution**2)/10000:.1f} hectares")
    print(f"   • High Risk Area: {high_risk_percent:.1f}%")
    print(f"   • Proposed Drainage: {drainage_network.total_length/1000:.1f} km")
    print(f"   • Estimated Cost: ₹{drainage_network.total_cost/10000000:.1f} crores")
    print(f"   • Expected Coverage: {drainage_network.coverage_area:.1f}%")
    print()
    
    print("📋 GENERATED VISUALIZATIONS:")
    for report_type, file_path in generated_files.items():
        print(f"   • {report_type.replace('_', ' ').title()}: {file_path.name}")
    
    print()
    print("🎯 KEY FEATURES FOR GOVERNMENT OFFICIALS:")
    print("   ✅ Clear, non-technical language")
    print("   ✅ Professional color schemes and layouts")
    print("   ✅ Cost-benefit analysis with timelines")
    print("   ✅ Before/after comparisons")
    print("   ✅ Risk assessment with probability maps")
    print("   ✅ Technical specifications for engineers")
    print("   ✅ Compliance with Indian standards")
    print("   ✅ High-resolution images for presentations")
    
    print()
    print("📊 READY FOR:")
    print("   • Government presentations")
    print("   • Court proceedings")
    print("   • Public consultations")
    print("   • Technical reviews")
    print("   • Budget approvals")
    print("   • Environmental clearances")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 All files saved to: {output_dir}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Government visualization demo completed successfully!")
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)