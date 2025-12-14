# ablation.py
"""
Feature Ablation Experiments (V2.4)
====================================

Systematically removes features to validate importance claims.
Results are cross-validated against SHAP in cross_validator.py.

V2.4 Updates:
- Dual ablation strategy: RF baseline + winning model
- Model-agnostic ablation measures data quality
- Model-specific ablation measures what the model relies on
- Both are scientifically valuable and complementary

Author: XCE Framework
Version: 2.4
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.base import clone

RANDOM_STATE = 42


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def create_rf_baseline():
    """Standard RandomForest model for model-agnostic ablation."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def evaluate_model(model, X, y, groups, n_splits=10):
    """
    Evaluate model with GroupShuffleSplit CV.
    
    Args:
        model: sklearn model or Pipeline (will be cloned for each fold)
        X: Features
        y: Labels
        groups: Group labels for CV
        n_splits: Number of CV splits
    
    Returns:
        Dict with composite score and std
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=RANDOM_STATE)
    
    composites = []
    for train_idx, test_idx in gss.split(X, y, groups):
        try:
            fold_model = clone(model)
            fold_model.fit(X[train_idx], y[train_idx])
            
            y_pred = fold_model.predict(X[test_idx])
            
            if hasattr(fold_model, 'predict_proba'):
                y_prob = fold_model.predict_proba(X[test_idx])[:, 1]
            elif hasattr(fold_model, 'decision_function'):
                y_prob = fold_model.decision_function(X[test_idx])
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
            else:
                y_prob = y_pred.astype(float)
            
            f1 = f1_score(y[test_idx], y_pred)
            mcc = matthews_corrcoef(y[test_idx], y_pred)
            auc = roc_auc_score(y[test_idx], y_prob)
            composites.append(f1 + mcc + auc)
        except Exception as e:
            continue
    
    if len(composites) < 2:
        return {'composite': 0.0, 'composite_std': 0.0, 'error': 'Too few successful folds'}
    
    return {
        'composite': float(np.mean(composites)),
        'composite_std': float(np.std(composites))
    }


def run_ablation_experiments(X, y, groups, feature_names, feature_metadata,
                            shap_file=None, output_dir='results',
                            winning_model=None):
    """
    Run ablation experiments with dual strategy.
    
    Experiments:
    1. Model-agnostic ablation (RF baseline) - measures feature informativeness
    2. Model-specific ablation (winning model) - measures what model uses
    3. Category ablation
    4. Pressure vs Static ablation
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Group labels for CV
        feature_names: List of feature names
        feature_metadata: Dict with feature info
        shap_file: Path to SHAP importance CSV (optional)
        output_dir: Output directory
        winning_model: Optional trained model for model-specific ablation
    
    Returns:
        Dict with all ablation results
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'dual' if winning_model else 'model_agnostic_only'
    }
    
    # =========================================================================
    # MODEL-AGNOSTIC ABLATION (RF Baseline)
    # =========================================================================
    print("\n  ── Model-Agnostic Ablation (RF Baseline) ──")
    print("  Purpose: Measure general feature informativeness")
    
    rf_baseline = create_rf_baseline()
    
    # Baseline performance
    print("\n  Computing RF baseline...")
    baseline_rf = evaluate_model(rf_baseline, X, y, groups)
    results['rf_baseline'] = {
        'composite': baseline_rf['composite'],
        'n_features': X.shape[1],
        'model': 'RandomForestClassifier'
    }
    print(f"  RF Baseline: {baseline_rf['composite']:.3f} ± {baseline_rf['composite_std']:.3f}")
    
    # Get feature ranking for ablation order
    if shap_file and os.path.exists(shap_file):
        shap_df = pd.read_csv(shap_file)
        top_features = shap_df.head(10)['feature'].tolist()
        print(f"\n  Using SHAP ranking from {shap_file}")
    else:
        # Fallback: variance ranking
        var = np.var(X, axis=0)
        top_idx = np.argsort(var)[::-1][:10]
        top_features = [feature_names[i] for i in top_idx]
        print("\n  Using variance-based ranking (no SHAP file)")
    
    # Individual feature ablation (RF)
    print("\n  Individual feature ablation (RF):")
    results['rf_individual'] = {}
    
    for feat in top_features:
        if feat not in feature_names:
            continue
        
        idx = feature_names.index(feat)
        X_ablated = np.delete(X, idx, axis=1)
        
        perf = evaluate_model(rf_baseline, X_ablated, y, groups)
        drop = 100 * (baseline_rf['composite'] - perf['composite']) / baseline_rf['composite']
        
        results['rf_individual'][feat] = {
            'composite': perf['composite'],
            'drop_percent': drop
        }
        
        arrow = "↓" if drop > 0 else "↑"
        print(f"    Remove {feat[:35]:<35}: Δ = {drop:+6.2f}% {arrow}")
    
    # =========================================================================
    # MODEL-SPECIFIC ABLATION (Winning Model)
    # =========================================================================
    if winning_model is not None:
        print("\n  ── Model-Specific Ablation (Winning Model) ──")
        print(f"  Purpose: Measure what {type(winning_model).__name__} relies on")
        
        # Baseline with winning model
        print("\n  Computing winning model baseline...")
        baseline_winning = evaluate_model(winning_model, X, y, groups)
        results['winning_baseline'] = {
            'composite': baseline_winning['composite'],
            'model': str(type(winning_model).__name__)
        }
        print(f"  Winning Model Baseline: {baseline_winning['composite']:.3f} ± {baseline_winning.get('composite_std', 0):.3f}")
        
        # Individual feature ablation (winning model)
        print("\n  Individual feature ablation (Winning Model):")
        results['winning_individual'] = {}
        
        for feat in top_features:
            if feat not in feature_names:
                continue
            
            idx = feature_names.index(feat)
            X_ablated = np.delete(X, idx, axis=1)
            
            try:
                perf = evaluate_model(winning_model, X_ablated, y, groups)
                drop = 100 * (baseline_winning['composite'] - perf['composite']) / baseline_winning['composite']
                
                results['winning_individual'][feat] = {
                    'composite': perf['composite'],
                    'drop_percent': drop
                }
                
                arrow = "↓" if drop > 0 else "↑"
                print(f"    Remove {feat[:35]:<35}: Δ = {drop:+6.2f}% {arrow}")
            except Exception as e:
                print(f"    Remove {feat[:35]:<35}: ERROR - {str(e)[:50]}")
                results['winning_individual'][feat] = {'error': str(e)[:100]}
    
    # =========================================================================
    # CATEGORY ABLATION (RF only - for consistency)
    # =========================================================================
    print("\n  ── Category Ablation (RF) ──")
    results['category'] = {}
    
    feature_info = feature_metadata.get('feature_info', {})
    categories = ['Electronegativity', 'Relative_EN', 'Atomic_Radius', 'Spin_State']
    
    for cat in categories:
        cat_features = [fn for fn in feature_names 
                       if feature_info.get(fn, {}).get('category') == cat]
        
        if not cat_features:
            # Fallback pattern matching
            prefix = {'Electronegativity': 'EN_', 'Relative_EN': 'RE_',
                     'Atomic_Radius': 'RA_', 'Spin_State': 'SP_'}.get(cat, '')
            cat_features = [fn for fn in feature_names if fn.startswith(prefix)]
        
        if not cat_features:
            continue
        
        keep_idx = [i for i, fn in enumerate(feature_names) if fn not in cat_features]
        X_ablated = X[:, keep_idx]
        
        perf = evaluate_model(rf_baseline, X_ablated, y, groups)
        drop = 100 * (baseline_rf['composite'] - perf['composite']) / baseline_rf['composite']
        
        results['category'][cat] = {
            'n_removed': len(cat_features),
            'composite': perf['composite'],
            'drop_percent': drop
        }
        
        print(f"    Remove {cat:<20} ({len(cat_features):2d} feat): Δ = {drop:+6.2f}%")
    
    # =========================================================================
    # PRESSURE VS STATIC ABLATION
    # =========================================================================
    print("\n  ── Pressure vs Static Ablation ──")
    results['pressure_vs_static'] = {}
    
    pressure_features = [fn for fn in feature_names 
                        if feature_info.get(fn, {}).get('pressure_dependent', False)]
    static_features = [fn for fn in feature_names if fn not in pressure_features]
    
    # Remove pressure-dependent
    if pressure_features:
        keep_idx = [i for i, fn in enumerate(feature_names) if fn not in pressure_features]
        X_static_only = X[:, keep_idx]
        
        perf = evaluate_model(rf_baseline, X_static_only, y, groups)
        drop = 100 * (baseline_rf['composite'] - perf['composite']) / baseline_rf['composite']
        
        results['pressure_vs_static']['remove_pressure'] = {
            'n_removed': len(pressure_features),
            'composite': perf['composite'],
            'drop_percent': drop
        }
        print(f"    Remove pressure-dependent ({len(pressure_features):2d} feat): Δ = {drop:+6.2f}%")
    
    # Remove static
    if static_features:
        keep_idx = [i for i, fn in enumerate(feature_names) if fn not in static_features]
        X_pressure_only = X[:, keep_idx]
        
        perf = evaluate_model(rf_baseline, X_pressure_only, y, groups)
        drop = 100 * (baseline_rf['composite'] - perf['composite']) / baseline_rf['composite']
        
        results['pressure_vs_static']['remove_static'] = {
            'n_removed': len(static_features),
            'composite': perf['composite'],
            'drop_percent': drop
        }
        print(f"    Remove static ({len(static_features):2d} feat):            Δ = {drop:+6.2f}%")
    
    # =========================================================================
    # COMPARISON ANALYSIS (if dual ablation)
    # =========================================================================
    if winning_model is not None and 'winning_individual' in results:
        print("\n  ── RF vs Winning Model Comparison ──")
        
        comparison = []
        for feat in top_features:
            if feat in results['rf_individual'] and feat in results['winning_individual']:
                rf_drop = results['rf_individual'][feat].get('drop_percent', 0)
                win_drop = results['winning_individual'][feat].get('drop_percent', 0)
                
                if isinstance(rf_drop, (int, float)) and isinstance(win_drop, (int, float)):
                    diff = abs(rf_drop - win_drop)
                    agreement = "✓" if diff < 2.0 else "?"
                    comparison.append({
                        'feature': feat,
                        'rf_drop': rf_drop,
                        'winning_drop': win_drop,
                        'difference': diff,
                        'agreement': agreement == "✓"
                    })
                    print(f"    {feat[:30]:<30}: RF={rf_drop:+5.1f}%, Win={win_drop:+5.1f}% {agreement}")
        
        results['comparison'] = comparison
        
        n_agree = sum(1 for c in comparison if c['agreement'])
        print(f"\n  Agreement: {n_agree}/{len(comparison)} features")
    
    # =========================================================================
    # BACKWARD COMPATIBILITY: Create 'individual' key from RF results
    # =========================================================================
    results['individual'] = results['rf_individual']
    results['baseline'] = results['rf_baseline']
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'ablation_results.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n  ✓ Saved: {filepath}")
    
    return results


if __name__ == '__main__':
    # Test loading and running
    X = np.load('X_full.npy')
    y = np.load('y_labels.npy')
    groups = np.load('groups.npy')
    
    with open('feature_names_full.json') as f:
        feature_names = json.load(f)
    with open('feature_metadata_full.json') as f:
        feature_metadata = json.load(f)
    
    run_ablation_experiments(X, y, groups, feature_names, feature_metadata)
