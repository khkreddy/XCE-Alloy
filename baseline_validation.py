# baseline_validation.py
"""
Baseline Validation - Hypothesis Testing Hard Gate (V2.3)
==========================================================

This module implements Phase 2: the critical hypothesis test that determines
whether to proceed with the full pipeline.

HYPOTHESIS: Pressure-dependent features improve prediction beyond ambient-only.

DECISION GATE:
  IF Δ(pressure - ambient) < 0.10 OR p > 0.05 → STOP
  IF Δ(pressure - ambient) ≥ 0.10 AND p ≤ 0.05 → PROCEED

Author: XCE Framework
Version: 2.3
"""

import numpy as np
import json
import os
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

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


def create_baseline_model():
    """Create standardized RandomForest for fair comparison."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def evaluate_model(X, y, groups, n_splits=10):
    """Evaluate model with GroupShuffleSplit and bootstrap CIs."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=RANDOM_STATE)
    
    f1_scores, mcc_scores, auc_scores = [], [], []
    
    for train_idx, test_idx in gss.split(X, y, groups):
        model = create_baseline_model()
        model.fit(X[train_idx], y[train_idx])
        
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        
        f1_scores.append(f1_score(y[test_idx], y_pred))
        mcc_scores.append(matthews_corrcoef(y[test_idx], y_pred))
        auc_scores.append(roc_auc_score(y[test_idx], y_prob))
    
    # Bootstrap CI for composite
    n_bootstrap = 1000
    composites = np.array(f1_scores) + np.array(mcc_scores) + np.array(auc_scores)
    bootstrap_composites = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(composites), size=len(composites), replace=True)
        bootstrap_composites.append(np.mean(composites[idx]))
    
    return {
        'f1': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'mcc': float(np.mean(mcc_scores)),
        'mcc_std': float(np.std(mcc_scores)),
        'auc': float(np.mean(auc_scores)),
        'auc_std': float(np.std(auc_scores)),
        'composite': float(np.mean(composites)),
        'composite_std': float(np.std(composites)),
        'composite_ci_95': [float(np.percentile(bootstrap_composites, 2.5)),
                           float(np.percentile(bootstrap_composites, 97.5))],
        'fold_composites': [float(c) for c in composites]
    }


def run_baseline_validation(X_ambient, X_pressure, X_full, y, groups,
                           delta_threshold=0.10, p_threshold=0.05):
    """
    Run baseline validation experiment.
    
    This is the HARD GATE for the pipeline.
    
    Args:
        X_ambient: Ambient-only features
        X_pressure: Pressure-only features  
        X_full: Full feature set
        y: Labels
        groups: CV groups
        delta_threshold: Minimum improvement required
        p_threshold: Significance threshold
    
    Returns:
        dict with results and proceed/stop decision
    """
    print("\n" + "="*70)
    print("PHASE 2: BASELINE VALIDATION (Hypothesis Test)")
    print("="*70)
    print(f"\nHypothesis: Pressure features improve prediction over ambient-only")
    print(f"Decision thresholds: Δ ≥ {delta_threshold}, p ≤ {p_threshold}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'delta_threshold': delta_threshold,
        'p_threshold': p_threshold
    }
    
    # Evaluate each model
    print("\n" + "-"*50)
    print("Evaluating models (10-fold GroupShuffleSplit)...")
    print("-"*50)
    
    print("\n  [A] Ambient-Only model...")
    results['ambient'] = evaluate_model(X_ambient, y, groups)
    results['ambient']['n_features'] = X_ambient.shape[1]
    print(f"      Composite: {results['ambient']['composite']:.3f} ± {results['ambient']['composite_std']:.3f}")
    print(f"      95% CI: [{results['ambient']['composite_ci_95'][0]:.3f}, {results['ambient']['composite_ci_95'][1]:.3f}]")
    
    print("\n  [B] Pressure-Only model...")
    results['pressure'] = evaluate_model(X_pressure, y, groups)
    results['pressure']['n_features'] = X_pressure.shape[1]
    print(f"      Composite: {results['pressure']['composite']:.3f} ± {results['pressure']['composite_std']:.3f}")
    print(f"      95% CI: [{results['pressure']['composite_ci_95'][0]:.3f}, {results['pressure']['composite_ci_95'][1]:.3f}]")
    
    print("\n  [C] Full model...")
    results['full'] = evaluate_model(X_full, y, groups)
    results['full']['n_features'] = X_full.shape[1]
    print(f"      Composite: {results['full']['composite']:.3f} ± {results['full']['composite_std']:.3f}")
    print(f"      95% CI: [{results['full']['composite_ci_95'][0]:.3f}, {results['full']['composite_ci_95'][1]:.3f}]")
    
    # Statistical comparisons
    print("\n" + "-"*50)
    print("Statistical Comparisons")
    print("-"*50)
    
    # Pressure vs Ambient (primary test)
    t_stat, p_value = stats.ttest_rel(
        results['pressure']['fold_composites'],
        results['ambient']['fold_composites']
    )
    delta = results['pressure']['composite'] - results['ambient']['composite']
    delta_percent = 100 * delta / results['ambient']['composite']
    
    # Cohen's d
    pooled_std = np.sqrt((np.var(results['pressure']['fold_composites']) + 
                         np.var(results['ambient']['fold_composites'])) / 2)
    cohens_d = delta / pooled_std if pooled_std > 0 else 0
    
    results['comparison_pressure_vs_ambient'] = {
        'delta': float(delta),
        'delta_percent': float(delta_percent),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': p_value <= p_threshold,
        'meets_delta_threshold': delta >= delta_threshold
    }
    
    print(f"\n  PRIMARY TEST: Pressure vs Ambient")
    print(f"    Δ Composite: {delta:+.3f} ({delta_percent:+.1f}%)")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Cohen's d: {cohens_d:.2f}")
    
    # Full vs Ambient
    t_stat_full, p_value_full = stats.ttest_rel(
        results['full']['fold_composites'],
        results['ambient']['fold_composites']
    )
    delta_full = results['full']['composite'] - results['ambient']['composite']
    
    results['comparison_full_vs_ambient'] = {
        'delta': float(delta_full),
        'delta_percent': float(100 * delta_full / results['ambient']['composite']),
        'p_value': float(p_value_full),
        'significant': p_value_full <= p_threshold
    }
    
    print(f"\n  SECONDARY TEST: Full vs Ambient")
    print(f"    Δ Composite: {delta_full:+.3f} ({100*delta_full/results['ambient']['composite']:+.1f}%)")
    print(f"    p-value: {p_value_full:.4f}")
    
    # Full vs Pressure (check if ambient adds value)
    t_stat_fp, p_value_fp = stats.ttest_rel(
        results['full']['fold_composites'],
        results['pressure']['fold_composites']
    )
    delta_fp = results['full']['composite'] - results['pressure']['composite']
    
    results['comparison_full_vs_pressure'] = {
        'delta': float(delta_fp),
        'delta_percent': float(100 * delta_fp / results['pressure']['composite']) if results['pressure']['composite'] > 0 else 0,
        'p_value': float(p_value_fp),
        'significant': p_value_fp <= p_threshold
    }
    
    print(f"\n  TERTIARY TEST: Full vs Pressure-only")
    print(f"    Δ Composite: {delta_fp:+.3f}")
    print(f"    p-value: {p_value_fp:.4f}")
    print(f"    (Tests if ambient features add value when pressure present)")
    
    # Decision
    print("\n" + "="*70)
    print("DECISION")
    print("="*70)
    
    hypothesis_supported = (delta >= delta_threshold) and (p_value <= p_threshold)
    
    results['decision'] = {
        'hypothesis_supported': hypothesis_supported,
        'proceed_with_pipeline': hypothesis_supported,
        'reasoning': []
    }
    
    if delta >= delta_threshold:
        results['decision']['reasoning'].append(f"✓ Δ = {delta:.3f} ≥ {delta_threshold} (threshold met)")
    else:
        results['decision']['reasoning'].append(f"✗ Δ = {delta:.3f} < {delta_threshold} (threshold NOT met)")
    
    if p_value <= p_threshold:
        results['decision']['reasoning'].append(f"✓ p = {p_value:.4f} ≤ {p_threshold} (statistically significant)")
    else:
        results['decision']['reasoning'].append(f"✗ p = {p_value:.4f} > {p_threshold} (NOT significant)")
    
    for reason in results['decision']['reasoning']:
        print(f"  {reason}")
    
    if hypothesis_supported:
        print(f"\n  ╔{'═'*60}╗")
        print(f"  ║{'HYPOTHESIS SUPPORTED - PROCEED WITH PIPELINE':^60}║")
        print(f"  ╚{'═'*60}╝")
        results['decision']['recommendation'] = "Proceed with multi-agent model exploration"
    else:
        print(f"\n  ╔{'═'*60}╗")
        print(f"  ║{'HYPOTHESIS NOT SUPPORTED - STOP PIPELINE':^60}║")
        print(f"  ╚{'═'*60}╝")
        results['decision']['recommendation'] = "Investigate feature engineering or pivot to negative results"
    
    # Additional insights
    print("\n" + "-"*50)
    print("Additional Insights")
    print("-"*50)
    
    if abs(delta_fp) < 0.02 and p_value_fp > 0.5:
        insight = "Pressure-only ≈ Full model: ambient features are redundant"
        print(f"  • {insight}")
        results['decision']['insights'] = [insight]
    
    if cohens_d > 0.8:
        print(f"  • Large effect size (d={cohens_d:.2f}): Pressure features have strong impact")
    elif cohens_d > 0.5:
        print(f"  • Medium effect size (d={cohens_d:.2f}): Moderate but meaningful improvement")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_results.json', 'w') as f:
        # Remove fold_composites for cleaner JSON
        save_results = results.copy()
        for key in ['ambient', 'pressure', 'full']:
            if 'fold_composites' in save_results[key]:
                del save_results[key]['fold_composites']
        json.dump(save_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Saved: results/baseline_results.json")
    
    return results


def load_and_run_baseline():
    """Load feature files and run baseline validation."""
    from feature_engineering import run_feature_engineering
    
    # Generate all three feature sets
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION")
    print("="*70)
    
    X_ambient, y, groups, _, _ = run_feature_engineering(mode='ambient')
    X_pressure, _, _, _, _ = run_feature_engineering(mode='pressure', save_outputs=True)
    X_full, _, _, _, _ = run_feature_engineering(mode='full', save_outputs=True)
    
    # Run baseline validation
    results = run_baseline_validation(X_ambient, X_pressure, X_full, y, groups)
    
    return results


if __name__ == '__main__':
    results = load_and_run_baseline()
    
    if not results['decision']['proceed_with_pipeline']:
        print("\n⚠️  Pipeline stopped at baseline validation.")
        print("    Review results/baseline_results.json for details.")
