"""
Master Experiment Runner
Runs all research experiments: ablation studies, category analysis, 
field validation, and full dataset investigation
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*100)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("="*100 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd="/home/saumil/Desktop/360paper",
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully in {elapsed:.2f}s")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} encountered error after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False, elapsed

def main():
    """Run all experiments"""
    print("\n" + "="*100)
    print("COMPREHENSIVE MULTI-VIEW YOLO RESEARCH EXPERIMENT SUITE")
    print("="*100)
    
    experiments = [
        # Quick analyses first
        ("product_category_analysis.py", "Product Category Performance Analysis"),
        ("full_dataset_analysis.py", "Full Dataset Underperformance Investigation"),
        ("field_validation.py", "Field Validation in Retail Environment"),
        
        # Long-running ablation study last (can be interrupted if needed)
        ("ablation_hyperparameter_study.py", "Hyperparameter Ablation Study")
    ]
    
    results = {}
    total_start = time.time()
    
    for script, description in experiments:
        script_path = Path("/home/saumil/Desktop/360paper") / script
        
        if not script_path.exists():
            print(f"\n✗ Script not found: {script}")
            results[description] = {"success": False, "time": 0, "error": "Script not found"}
            continue
        
        success, elapsed = run_script(str(script_path), description)
        results[description] = {
            "success": success,
            "time": elapsed,
            "script": script
        }
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*100)
    print("EXPERIMENT SUITE SUMMARY")
    print("="*100)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    print(f"\nTotal experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Total time: {total_time/60:.2f} minutes")
    
    print("\nDetailed Results:")
    print("-" * 100)
    
    for description, result in results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        time_str = f"{result['time']:.2f}s" if result['time'] < 60 else f"{result['time']/60:.2f}m"
        print(f"{status:12s} | {time_str:10s} | {description}")
    
    print("\n" + "="*100)
    print("All results are saved in: /home/saumil/Desktop/360paper/data/results/")
    print("="*100)
    
    # List output directories
    print("\nOutput Directories:")
    output_dirs = [
        "data/results/category_analysis",
        "data/results/full_dataset_analysis",
        "data/results/field_validation",
        "data/results/ablation_studies"
    ]
    
    for dir_path in output_dirs:
        full_path = Path("/home/saumil/Desktop/360paper") / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (not created)")
    
    print("\n" + "="*100)
    
    if successful == total:
        print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    else:
        print(f"⚠️  {total - successful} experiment(s) failed. Check logs above for details.")
    
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
