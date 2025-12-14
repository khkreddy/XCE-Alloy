# cross_validator.py
"""
SHAP-Ablation Cross-Validation (V2.4)
======================================

Validates importance claims by comparing SHAP and ablation results.
Flags discrepancies where rankings don't match actual impact.

V2.4 Updates:
- Graceful degradation when SHAP unavailable
- Ablation-only validation with clear confidence reporting
- Improved flag categorization

Purpose: Hallucination resistance - ensure claims are validated.

Author: XCE Framework
Version: 2.4
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime


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


def cross_validate_importance(shap_file='results/shap_importance.csv',
                             ablation_file='results/ablation_results.json',
                             output_dir='results',
                             mismatch_threshold=0.20):
    """
    Cross-validate SHAP importance against ablation experiments.
    
    If SHAP unavailable, provides ablation-only validation with clear reporting.
    
    Args:
        shap_file: Path to SHAP importance CSV
        ablation_file: Path to ablation results JSON
        output_dir: Output directory
        mismatch_threshold: Rank difference threshold for flagging (0.20 = 20%)
    
    Returns:
        Dict with validation results and confidence labels
    """
    print("\n" + "="*60)
    print("SHAP-ABLATION CROSS-VALIDATION")
    print("="*60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'shap_available': False,
        'ablation_available': False,
        'validation_mode': None,
        'validated': False,
        'flags': [],
        'warnings': [],
        'confidence': 'LOW'
    }
    
    # Check for SHAP file
    shap_df = None
    if shap_file and os.path.exists(shap_file):
        try:
            shap_df = pd.read_csv(shap_file)
            if len(shap_df) > 0 and 'feature' in shap_df.columns and 'importance' in shap_df.columns:
                results['shap_available'] = True
                print(f"\n  ✓ SHAP file loaded: {len(shap_df)} features")
            else:
                print(f"\n  ⚠ SHAP file invalid: missing required columns")
        except Exception as e:
            print(f"\n  ⚠ SHAP file error: {str(e)[:50]}")
    else:
        print(f"\n  ⚠ SHAP file not found: {shap_file}")
    
    # Check for ablation file
    ablation_results = None
    if ablation_file and os.path.exists(ablation_file):
        try:
            with open(ablation_file) as f:
                ablation_results = json.load(f)
            if 'rf_individual' in ablation_results or 'individual' in ablation_results:
                results['ablation_available'] = True
                print(f"  ✓ Ablation file loaded")
            else:
                print(f"  ⚠ Ablation file invalid: missing individual results")
        except Exception as e:
            print(f"  ⚠ Ablation file error: {str(e)[:50]}")
    else:
        print(f"  ⚠ Ablation file not found: {ablation_file}")
    
    # Determine validation mode
    if results['shap_available'] and results['ablation_available']:
        results['validation_mode'] = 'full_cross_validation'
        print(f"\n  Mode: FULL CROSS-VALIDATION (SHAP + Ablation)")
    elif results['ablation_available']:
        results['validation_mode'] = 'ablation_only'
        print(f"\n  Mode: ABLATION-ONLY VALIDATION")
        print(f"  ⚠ Reduced confidence (no SHAP cross-reference)")
    else:
        results['validation_mode'] = 'none'
        results['flags'].append("No validation data available")
        print(f"\n  ✗ Cannot perform validation: no data available")
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'validation_flags.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        return results
    
    # =========================================================================
    # ABLATION-ONLY VALIDATION
    # =========================================================================
    if results['validation_mode'] == 'ablation_only':
        print("\n  ── Ablation-Only Analysis ──")
        
        # Get individual ablation results
        individual = ablation_results.get('rf_individual', ablation_results.get('individual', {}))
        
        if individual:
            # Sort by drop percent (higher drop = more important)
            sorted_features = sorted(
                [(f, d.get('drop_percent', 0)) for f, d in individual.items() if isinstance(d, dict)],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            print("\n  Top features by ablation impact:")
            for i, (feat, drop) in enumerate(sorted_features[:10], 1):
                direction = "↓ hurts" if drop > 0 else "↑ helps"
                print(f"    {i}. {feat[:35]:<35}: {abs(drop):5.2f}% {direction}")
            
            results['ablation_ranking'] = [
                {'feature': f, 'drop_percent': d, 'rank': i+1}
                for i, (f, d) in enumerate(sorted_features)
            ]
            
            # Check for concerning patterns
            n_positive = sum(1 for _, d in sorted_features if d > 0)
            n_negative = sum(1 for _, d in sorted_features if d < 0)
            
            if n_negative > n_positive:
                results['warnings'].append(
                    f"More features hurt performance when removed ({n_negative}) than help ({n_positive}). "
                    "This may indicate overfitting or feature redundancy."
                )
            
            # Check pressure vs static from ablation
            pressure_static = ablation_results.get('pressure_vs_static', {})
            remove_pressure = pressure_static.get('remove_pressure', {})
            remove_static = pressure_static.get('remove_static', {})
            
            if remove_pressure and remove_static:
                p_drop = remove_pressure.get('drop_percent', 0)
                s_drop = remove_static.get('drop_percent', 0)
                
                print(f"\n  Pressure vs Static importance:")
                print(f"    Removing pressure features: {p_drop:+.2f}%")
                print(f"    Removing static features:   {s_drop:+.2f}%")
                
                if p_drop > s_drop + 5:
                    results['findings'] = results.get('findings', [])
                    results['findings'].append("Pressure features more important than static (supports hypothesis)")
                elif s_drop > p_drop + 5:
                    results['warnings'].append("Static features appear more important than pressure features")
        
        # Set confidence for ablation-only
        results['confidence'] = 'MEDIUM'
        results['confidence_reason'] = 'Ablation only - no SHAP cross-reference'
        results['validated'] = True
        
        print(f"\n  Validation: PASSED (ablation-only)")
        print(f"  Confidence: MEDIUM")
    
    # =========================================================================
    # FULL CROSS-VALIDATION (SHAP + Ablation)
    # =========================================================================
    elif results['validation_mode'] == 'full_cross_validation':
        print("\n  ── Full Cross-Validation ──")
        
        # Get individual ablation results
        individual = ablation_results.get('rf_individual', ablation_results.get('individual', {}))
        
        # Build rankings
        shap_ranking = {row['feature']: i+1 for i, row in shap_df.iterrows()}
        
        ablation_sorted = sorted(
            [(f, d.get('drop_percent', 0)) for f, d in individual.items() if isinstance(d, dict)],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        ablation_ranking = {f: i+1 for i, (f, _) in enumerate(ablation_sorted)}
        
        # Cross-validate
        mismatches = []
        agreements = []
        
        print("\n  Comparing SHAP vs Ablation rankings:")
        
        for feat in list(shap_ranking.keys())[:10]:  # Top 10 by SHAP
            shap_rank = shap_ranking.get(feat, 999)
            abl_rank = ablation_ranking.get(feat, 999)
            abl_drop = individual.get(feat, {}).get('drop_percent', 0) if feat in individual else 0
            
            # Calculate rank difference
            n_features = len(shap_ranking)
            rank_diff = abs(shap_rank - abl_rank) / n_features if n_features > 0 else 0
            
            status = "✓" if rank_diff < mismatch_threshold else "?"
            
            if rank_diff >= mismatch_threshold:
                mismatches.append({
                    'feature': feat,
                    'shap_rank': shap_rank,
                    'ablation_rank': abl_rank,
                    'ablation_drop': abl_drop,
                    'rank_diff': rank_diff
                })
            else:
                agreements.append({'feature': feat, 'shap_rank': shap_rank, 'ablation_rank': abl_rank})
            
            print(f"    {feat[:30]:<30}: SHAP #{shap_rank:<3} vs Abl #{abl_rank:<3} (Δ={abl_drop:+5.2f}%) {status}")
        
        results['agreements'] = agreements
        results['mismatches'] = mismatches
        
        # Generate flags for significant mismatches
        for m in mismatches:
            if m['shap_rank'] <= 5 and abs(m['ablation_drop']) < 1.0:
                results['flags'].append(
                    f"⚠ SHAP says {m['feature']} is top-5, but ablation shows only {m['ablation_drop']:.2f}% impact"
                )
            elif m['ablation_rank'] <= 5 and m['shap_rank'] > 20:
                results['flags'].append(
                    f"⚠ Ablation shows {m['feature']} is important, but SHAP ranks it #{m['shap_rank']}"
                )
        
        # Set validation status
        n_flags = len(results['flags'])
        if n_flags == 0:
            results['validated'] = True
            results['confidence'] = 'HIGH'
            results['confidence_reason'] = 'SHAP and ablation rankings agree'
            print(f"\n  ✓ Validation: PASSED")
            print(f"  Confidence: HIGH")
        elif n_flags <= 2:
            results['validated'] = True
            results['confidence'] = 'MEDIUM'
            results['confidence_reason'] = f'{n_flags} minor discrepancies found'
            print(f"\n  ✓ Validation: PASSED with {n_flags} warnings")
            print(f"  Confidence: MEDIUM")
        else:
            results['validated'] = False
            results['confidence'] = 'LOW'
            results['confidence_reason'] = f'{n_flags} significant discrepancies'
            print(f"\n  ⚠ Validation: CONCERNS FOUND ({n_flags} flags)")
            print(f"  Confidence: LOW")
        
        # Check category consistency
        if 'category' in ablation_results:
            shap_category = shap_df.groupby('category')['importance'].sum() if 'category' in shap_df.columns else None
            
            if shap_category is not None:
                print("\n  Category-level comparison:")
                for cat in ablation_results['category']:
                    abl_drop = ablation_results['category'][cat].get('drop_percent', 0)
                    shap_pct = 100 * shap_category.get(cat, 0) / shap_category.sum() if shap_category.sum() > 0 else 0
                    
                    print(f"    {cat:<20}: SHAP {shap_pct:5.1f}% | Ablation Δ={abl_drop:+5.1f}%")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Print flags and warnings
    if results['flags']:
        print("\n  FLAGS (review required):")
        for flag in results['flags']:
            print(f"    {flag}")
    
    if results['warnings']:
        print("\n  WARNINGS (review recommended):")
        for warning in results['warnings'][:5]:
            print(f"    {warning}")
        if len(results['warnings']) > 5:
            print(f"    ... and {len(results['warnings']) - 5} more")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'validation_flags.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n  ✓ Saved: {filepath}")
    
    return results


def generate_confidence_labels(validation_results, shap_df=None, ablation_results=None):
    """
    Generate confidence labels for synthesis claims.
    
    HIGH: SHAP and ablation agree strongly
    MEDIUM: Partial agreement or ablation-only
    LOW: Significant discrepancies or no validation
    
    Args:
        validation_results: Result from cross_validate_importance()
        shap_df: Optional SHAP dataframe
        ablation_results: Optional ablation results dict
    
    Returns:
        Dict mapping feature names to confidence levels
    """
    confidence = {}
    
    # If no validation available
    if validation_results.get('validation_mode') == 'none':
        return confidence
    
    # Get mismatches
    mismatches = {m['feature'] for m in validation_results.get('mismatches', [])}
    flags = ' '.join(validation_results.get('flags', []))
    
    # Get features from SHAP or ablation
    if shap_df is not None and len(shap_df) > 0:
        features = shap_df['feature'].tolist()
    elif ablation_results:
        individual = ablation_results.get('rf_individual', ablation_results.get('individual', {}))
        features = list(individual.keys())
    else:
        return confidence
    
    # Assign confidence labels
    for feat in features:
        if feat in mismatches or feat in flags:
            confidence[feat] = 'LOW'
        elif validation_results.get('validation_mode') == 'ablation_only':
            confidence[feat] = 'MEDIUM'
        else:
            confidence[feat] = 'HIGH'
    
    return confidence


if __name__ == '__main__':
    # Test run
    results = cross_validate_importance()
    print(f"\nValidation mode: {results['validation_mode']}")
    print(f"Confidence: {results['confidence']}")
