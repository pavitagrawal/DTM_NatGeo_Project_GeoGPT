#!/usr/bin/env python3
"""
Run All Demos Script for Intelligent Hydro-DTM System

This script runs all the main demonstrations in sequence to showcase
the complete capabilities of the system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_demo(script_name, description):
    """Run a demo script and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {description}")
    print(f"📄 Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            print(f"⏱️  Time: {end_time - start_time:.1f} seconds")
            if result.stdout:
                print("📋 Output:")
                print(result.stdout[-500:])  # Last 500 characters
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-500:])
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    """Run all demonstrations."""
    print("🌊 Intelligent Hydro-DTM System - Complete Demo Suite")
    print("=" * 60)
    print("This script will run all major demonstrations to showcase")
    print("the complete AI pipeline capabilities.")
    print()
    
    # Check if we're in the right directory
    if not Path("src/hydro_dtm").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected structure: src/hydro_dtm/")
        return False
    
    # Create output directory
    Path("demo_outputs").mkdir(exist_ok=True)
    
    # List of demos to run
    demos = [
        ("working_complete_demo.py", "Complete AI Pipeline Demo"),
        ("ml_waterlogging_demo.py", "ML Waterlogging Prediction Demo"),
        ("government_demo.py", "Government Visualization Demo"),
    ]
    
    results = []
    total_start = time.time()
    
    for script, description in demos:
        if Path(script).exists():
            success = run_demo(script, description)
            results.append((script, description, success))
        else:
            print(f"⚠️  SKIPPED: {script} (file not found)")
            results.append((script, description, False))
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DEMO SUITE SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for script, description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    print(f"\n🎯 Overall Results: {successful}/{total} demos successful")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    
    if successful == total:
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("📁 Check 'demo_outputs/' folder for generated results")
        print("🌐 Run 'streamlit run demo_ui.py' for interactive interface")
    else:
        print(f"\n⚠️  {total - successful} demos failed. Check error messages above.")
    
    # Check outputs
    output_dir = Path("demo_outputs")
    if output_dir.exists():
        output_files = list(output_dir.rglob("*"))
        print(f"\n📁 Generated {len(output_files)} output files:")
        for file_path in sorted(output_files)[:10]:  # Show first 10
            if file_path.is_file():
                size = file_path.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"   📄 {file_path.relative_to(output_dir)} ({size_str})")
        
        if len(output_files) > 10:
            print(f"   ... and {len(output_files) - 10} more files")
    
    return successful == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)