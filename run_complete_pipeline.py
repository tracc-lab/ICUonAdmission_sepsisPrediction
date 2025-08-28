#!/usr/bin/env python3
"""
Complete Sepsis Prediction Pipeline

This script puts together the complete machine learning pipeline for sepsis prediction:
1. Feature selection and cross-validation (pipeline_featSel_wHeatmap.py)
2. Hold-out evaluation with or without onset data (wrapHoldOutEval_2.py or wrapHoldOutEval_noOnset.py)

How to use:
    python run_complete_pipeline.py --with-onset     # Run with onset analysis
    python run_complete_pipeline.py --no-onset      # Run without onset analysis
    python run_complete_pipeline.py                 # Default: no onset
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
import glob

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration from a file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if not os.path.exists(filepath):
        print(f"ERROR: {description} not found at: {filepath}")
        return False
    print(f"✓ Found {description}: {filepath}")
    return True

def validate_environment():
    """Validate that all required files and dependencies are available."""
    print("=" * 80)
    print("VALIDATING PIPELINE ENVIRONMENT")
    print("=" * 80)
    
    required_files = [
        ("pipeline_featSel_wHeatmap.py", "Feature selection pipeline"),
        ("evaluate_holdout.py", "Hold-out evaluation with onset"),
        ("evaluate_holdout_noOnset.py", "Hold-out evaluation without onset"),
        ("config.yaml", "Configuration file"),
        ("Methods_utils/methods.py", "Core methods module"),
        ("Methods_utils/methods_heatmap.py", "Heatmap methods module")
    ]
    
    all_good = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    if not all_good:
        print("\nERROR: Missing required files. Please ensure all files are in the correct location.")
        return False
    
    print("\n All required files found! We can proceed.")
    return True

def run_feature_selection_pipeline(cv_nr: int = 10, extras: bool = True, 
                                 wish_toPlot_AUROC: bool = False, 
                                 wish_toPlot_AUPRC: bool = False):
    """Run the feature selection and cross-validation pipeline."""
    print("=" * 80)
    print("STEP 1: RUNNING FEATURE SELECTION PIPELINE")
    print("=" * 80)
    
    try:
        # Import and run the feature selection pipeline
        sys.path.append('.')
        from pipeline_featSel_wHeatmap import feature_selection_and_predictions
        
        print(f"Starting feature selection with {cv_nr}-fold cross-validation...")
        feature_selection_and_predictions(cv_nr, extras, wish_toPlot_AUROC, wish_toPlot_AUPRC)
        
        print("✓ Feature selection pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR in feature selection pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_holdout_evaluation(with_onset: bool = False, cv_nr: int = 10, 
                          run_number: int = 11, n_iterations: int = 3):
    """Run the hold-out evaluation pipeline."""
    print("=" * 80)
    if with_onset:
        print("STEP 2: RUNNING HOLD-OUT EVALUATION WITH ONSET ANALYSIS")
    else:
        print("STEP 2: RUNNING HOLD-OUT EVALUATION WITHOUT ONSET ANALYSIS")
    print("=" * 80)
    
    try:
        if with_onset:
            # Import and run with onset analysis
            from evaluate_holdout import wrapAdvancedAnalysis
            print(f"Starting hold-out evaluation WITH onset analysis...")
            print(f"Parameters: CV={cv_nr}, Run={run_number}, Iterations={n_iterations}")
            wrapAdvancedAnalysis(cv_nr, run_number, n_iterations)
        else:
            # Import and run without onset analysis
            from evaluate_holdout_noOnset import wrapAdvancedAnalysis
            print(f"Starting hold-out evaluation WITHOUT onset analysis...")
            print(f"Parameters: CV={cv_nr}, Run={run_number}, Iterations={n_iterations}")
            wrapAdvancedAnalysis(cv_nr, run_number, n_iterations)
        
        print("✓ Hold-out evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR in hold-out evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_results(config: dict, with_onset: bool):
    """Verify that expected result files were generated."""
    print("=" * 80)
    print("VERIFYING PIPELINE RESULTS")
    print("=" * 80)
    
    # Check feature selection results
    feat_sel_dir = config["results_feat_sel"]["dir_path"]
    pt10_file = os.path.join(feat_sel_dir, os.path.basename(config["results_feat_sel"]["PT10_res"]))
    cv_results_pattern = os.path.join(feat_sel_dir, "resultsAllCVs_pipeline_*_split_*.csv") # the stars are there because the numbers can vary
    
    print("Feature Selection Results:")
    if os.path.exists(pt10_file):
        print(f"✓ PT10 results: {pt10_file}")
    else:
        print(f"✗ PT10 results missing: {pt10_file}")
    
    # Check for CV results (at least one should exist)
    cv_files = glob.glob(cv_results_pattern)
    if cv_files:
        print(f"✓ CV results found: {len(cv_files)} files")
        for f in cv_files[:3]:  # Show first 3
            print(f"  - {f}")
    else:
        print(f"✗ CV results missing: {cv_results_pattern}")
    
    # Check hold-out results
    if with_onset:
        holdout_dir = config["results_holdOut"]["results_folder"]
    else:
        holdout_dir = config["results_holdOut"]["results_folder_noOnset"]
    
    print(f"\nHold-out Evaluation Results:")
    if os.path.exists(holdout_dir):
        print(f"✓ Results directory: {holdout_dir}")
        
        # Look for log files and confusion matrices
        log_files = glob.glob(os.path.join(holdout_dir, "*.txt"))
        cm_files = glob.glob(os.path.join(holdout_dir, "*CM*.png"))
        roc_files = glob.glob(os.path.join(holdout_dir, "*holdout.png"))
        
        print(f"  - Log files: {len(log_files)}")
        print(f"  - Confusion matrices: {len(cm_files)}")
        print(f"  - ROC curves: {len(roc_files)}")
        
    else:
        print(f"✗ Results directory missing: {holdout_dir}")

def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Complete Sepsis Prediction Pipeline")
    parser.add_argument("--with-onset", action="store_true", 
                       help="Run hold-out evaluation with onset analysis")
    parser.add_argument("--no-onset", action="store_true", 
                       help="Run hold-out evaluation without onset analysis (default)")
    parser.add_argument("--cv-folds", type=int, default=10,
                       help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--run-number", type=int, default=11,
                       help="Run number identifier (default: 11)")
    parser.add_argument("--iterations", type=int, default=20,
                       help="Number of hold-out iterations (default: 20)")
    parser.add_argument("--plot-auroc", action="store_true",
                       help="Generate AUROC plots during feature selection")
    parser.add_argument("--plot-auprc", action="store_true",
                       help="Generate AUPRC plots during feature selection")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip environment validation")
    parser.add_argument("--skip-feature-selection", action="store_true",
                       help="Skip feature selection step (use existing results)")
    
    args = parser.parse_args()
    
    # Determine onset flag
    if args.with_onset and args.no_onset:
        print("ERROR: Cannot specify both --with-onset and --no-onset")
        sys.exit(1)
    
    with_onset = args.with_onset  # Default is False (no onset)
    
    print("=" * 80)
    print("SEPSIS PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Onset analysis: {'YES' if with_onset else 'NO'}")
    print(f"  - CV folds: {args.cv_folds}")
    print(f"  - Run number: {args.run_number}")
    print(f"  - Hold-out iterations: {args.iterations}")
    print(f"  - Plot AUROC: {'YES' if args.plot_auroc else 'NO'}")
    print(f"  - Plot AUPRC: {'YES' if args.plot_auprc else 'NO'}")
    print()
    
    # Load configuration
    try:
        config = load_config()
        print("^_^ Configuration loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate environment
    if not args.skip_validation:
        if not validate_environment():
            sys.exit(1)
    
    # Step 1: Feature Selection Pipeline
    if not args.skip_feature_selection:
        success = run_feature_selection_pipeline(
            cv_nr=args.cv_folds,
            extras=True,
            wish_toPlot_AUROC=args.plot_auroc,
            wish_toPlot_AUPRC=args.plot_auprc
        )
        if not success:
            print("PIPELINE FAILED at feature selection step!")
            sys.exit(1)
    else:
        print("Skipping feature selection step (using existing results)")
    
    # Step 2: Hold-out Evaluation
    success = run_holdout_evaluation(
        with_onset=with_onset,
        cv_nr=args.cv_folds,
        run_number=args.run_number,
        n_iterations=args.iterations
    )
    if not success:
        print("PIPELINE FAILED at hold-out evaluation step!")
        sys.exit(1)
    
    # Verify results
    verify_results(config, with_onset)
    
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"^_^ Feature selection with {args.cv_folds}-fold CV")
    print(f"^_^ Hold-out evaluation {'with' if with_onset else 'without'} onset analysis")
    print(f"^_^ {args.iterations} hold-out iterations completed")
    print("\nCheck the results directories for generated files:")
    print(f"  - Feature selection: {config['results_feat_sel']['dir_path']}")
    if with_onset:
        print(f"  - Hold-out evaluation: {config['results_holdOut']['results_folder']}")
    else:
        print(f"  - Hold-out evaluation: {config['results_holdOut']['results_folder_noOnset']}")

if __name__ == "__main__":
    main()
