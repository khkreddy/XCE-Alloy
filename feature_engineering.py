# feature_engineering.py
"""
Feature Engineering for Binary Alloy Mixing Prediction (V2.3)
==============================================================

Generates three feature sets for hypothesis testing:
  - AMBIENT: P=0 values only (Hume-Rothery baseline)
  - PRESSURE: Pressure dynamics only (gradients, curvatures, ranges)
  - FULL: All features combined

V2.3: Streamlined, deterministic, serves hypothesis testing directly.

Author: XCE Framework
Version: 2.3
"""

import numpy as np
import pandas as pd
import json
import os
import re
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


def parse_property_file(filename):
    """Parse SHARC-format property file."""
    elements = []
    data_lines = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and 'Column' in line and ':' in line:
                match = re.search(r'Column\s+(\d+):\s+(\w+)', line)
                if match:
                    col_num = int(match.group(1))
                    element = match.group(2)
                    if col_num > 1:
                        elements.append(element)
            elif line and not line.startswith('#'):
                values = line.split()
                if len(values) > 1:
                    data_lines.append([float(v) for v in values])
    
    data_array = np.array(data_lines)
    pressures = data_array[:, 0]
    values = data_array[:, 1:]
    
    data_dict = {elem: values[:, i] for i, elem in enumerate(elements) if i < values.shape[1]}
    return elements, pressures, data_dict


def compute_ambient_features(f1, f2, prop_name):
    """Compute AMBIENT-ONLY features (P=0 values)."""
    features = {}
    features[f"{prop_name}_A_initial"] = f1[0]
    features[f"{prop_name}_B_initial"] = f2[0]
    features[f"{prop_name}_delta_initial"] = f1[0] - f2[0]
    features[f"{prop_name}_delta_abs_initial"] = np.abs(f1[0] - f2[0])
    if np.abs(f2[0]) > 1e-10:
        features[f"{prop_name}_ratio_initial"] = f1[0] / f2[0]
    else:
        features[f"{prop_name}_ratio_initial"] = 1.0
    return features


def compute_pressure_features(f1, f2, prop_name, pressures):
    """Compute PRESSURE-DEPENDENT features (dynamics only)."""
    features = {}
    dx = np.mean(np.diff(pressures)) if len(pressures) > 1 else 1.0
    
    # Element A dynamics
    features[f"{prop_name}_A_range"] = np.ptp(f1)
    features[f"{prop_name}_A_std"] = np.std(f1)
    grad1 = np.gradient(f1, dx)
    features[f"{prop_name}_A_grad_mean"] = np.mean(grad1)
    features[f"{prop_name}_A_grad_max"] = np.max(np.abs(grad1))
    if len(f1) >= 3:
        curv1 = np.gradient(grad1, dx)
        features[f"{prop_name}_A_curv_mean"] = np.mean(curv1)
        features[f"{prop_name}_A_curv_max"] = np.max(np.abs(curv1))
    
    # Element B dynamics
    features[f"{prop_name}_B_range"] = np.ptp(f2)
    features[f"{prop_name}_B_std"] = np.std(f2)
    grad2 = np.gradient(f2, dx)
    features[f"{prop_name}_B_grad_mean"] = np.mean(grad2)
    features[f"{prop_name}_B_grad_max"] = np.max(np.abs(grad2))
    if len(f2) >= 3:
        curv2 = np.gradient(grad2, dx)
        features[f"{prop_name}_B_curv_mean"] = np.mean(curv2)
        features[f"{prop_name}_B_curv_max"] = np.max(np.abs(curv2))
    
    # Pairwise dynamics
    diff = f1 - f2
    features[f"{prop_name}_delta_range"] = np.ptp(diff)
    features[f"{prop_name}_delta_std"] = np.std(diff)
    features[f"{prop_name}_delta_squared_mean"] = np.mean(diff**2)
    features[f"{prop_name}_delta_rmse"] = np.sqrt(np.mean(diff**2))
    
    # Gradient interactions
    features[f"{prop_name}_grad_delta_mean"] = np.mean(grad1 - grad2)
    features[f"{prop_name}_grad_product_mean"] = np.mean(grad1 * grad2)
    
    # Crossings (elements swap relative values)
    features[f"{prop_name}_n_crossings"] = float(np.sum(np.diff(np.sign(diff)) != 0))
    
    # AUC
    try:
        auc = np.trapezoid(np.abs(diff), pressures)
    except AttributeError:
        auc = np.trapz(np.abs(diff), pressures)
    features[f"{prop_name}_delta_auc"] = auc / (pressures[-1] - pressures[0] + 1e-10)
    
    return features


def compute_spin_ambient(f1, f2):
    """Compute ambient spin features."""
    return {
        "SP_A_initial": f1[0],
        "SP_B_initial": f2[0],
        "SP_delta_initial": f1[0] - f2[0]
    }


def compute_spin_pressure(f1, f2, threshold=0.5):
    """Compute pressure-dependent spin features."""
    features = {}
    
    # Transitions
    changes1 = np.abs(np.diff(f1))
    changes2 = np.abs(np.diff(f2))
    
    features["SP_A_has_transition"] = float(np.max(changes1) >= threshold)
    features["SP_B_has_transition"] = float(np.max(changes2) >= threshold)
    features["SP_A_n_transitions"] = float(np.sum(changes1 >= threshold))
    features["SP_B_n_transitions"] = float(np.sum(changes2 >= threshold))
    
    if np.max(changes1) >= threshold:
        features["SP_A_transition_pressure_norm"] = np.argmax(changes1 >= threshold) / max(len(changes1), 1)
    else:
        features["SP_A_transition_pressure_norm"] = -1.0
    
    if np.max(changes2) >= threshold:
        features["SP_B_transition_pressure_norm"] = np.argmax(changes2 >= threshold) / max(len(changes2), 1)
    else:
        features["SP_B_transition_pressure_norm"] = -1.0
    
    features["SP_both_transition"] = float(features["SP_A_has_transition"] and features["SP_B_has_transition"])
    features["SP_either_transition"] = float(features["SP_A_has_transition"] or features["SP_B_has_transition"])
    features["SP_total_change_A"] = f1[-1] - f1[0]
    features["SP_total_change_B"] = f2[-1] - f2[0]
    
    return features


def compute_features_for_mode(prop_data, elements, pressures, mode):
    """Compute features for specified mode."""
    feature_rows = []
    element_pairs = []
    feature_names = None
    
    n_elements = len(elements)
    
    for i in range(n_elements):
        for j in range(i + 1, n_elements):
            e1, e2 = elements[i], elements[j]
            row = {}
            
            for prop_name, elem_data in prop_data.items():
                if e1 not in elem_data or e2 not in elem_data:
                    continue
                
                f1, f2 = elem_data[e1], elem_data[e2]
                
                if prop_name == 'SP':
                    if mode == 'ambient':
                        row.update(compute_spin_ambient(f1, f2))
                    elif mode == 'pressure':
                        row.update(compute_spin_pressure(f1, f2))
                    else:  # full
                        row.update(compute_spin_ambient(f1, f2))
                        row.update(compute_spin_pressure(f1, f2))
                else:
                    if mode == 'ambient':
                        row.update(compute_ambient_features(f1, f2, prop_name))
                    elif mode == 'pressure':
                        row.update(compute_pressure_features(f1, f2, prop_name, pressures))
                    else:  # full
                        row.update(compute_ambient_features(f1, f2, prop_name))
                        row.update(compute_pressure_features(f1, f2, prop_name, pressures))
            
            if feature_names is None:
                feature_names = sorted(row.keys())
            
            feature_rows.append([row.get(fn, 0.0) for fn in feature_names])
            element_pairs.append((e1, e2))
    
    return np.array(feature_rows), feature_names, element_pairs


def create_pair_groups(element_pairs):
    """Create canonical pair-based groups for CV."""
    pair_to_group = {}
    groups = []
    
    for e1, e2 in element_pairs:
        canonical = tuple(sorted([e1, e2]))
        if canonical not in pair_to_group:
            pair_to_group[canonical] = len(pair_to_group)
        groups.append(pair_to_group[canonical])
    
    return np.array(groups)


def build_feature_metadata(feature_names, mode):
    """Build metadata for each feature."""
    pressure_keywords = ['grad', 'curv', 'range', 'std', 'auc', 'transition',
                        'crossings', 'rmse', 'squared', 'product', 'n_transitions']
    
    metadata = {}
    for fn in feature_names:
        fn_lower = fn.lower()
        
        # Determine category
        if fn_lower.startswith('en_'):
            category = 'Electronegativity'
        elif fn_lower.startswith('re_'):
            category = 'Relative_EN'
        elif fn_lower.startswith('ra_'):
            category = 'Atomic_Radius'
        elif fn_lower.startswith('sp_'):
            category = 'Spin_State'
        else:
            category = 'Other'
        
        # Determine if pressure-dependent
        is_pressure = any(kw in fn_lower for kw in pressure_keywords)
        if '_initial' in fn_lower and 'delta' not in fn_lower:
            is_pressure = False
        
        metadata[fn] = {
            'category': category,
            'pressure_dependent': is_pressure
        }
    
    return metadata


def run_feature_engineering(mode='full', save_outputs=True):
    """
    Main feature engineering function.
    
    Args:
        mode: 'ambient' | 'pressure' | 'full'
        save_outputs: Whether to save to disk
    
    Returns:
        X, y, groups, feature_names, metadata_dict
    """
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING (Mode: {mode.upper()})")
    print('='*60)
    
    # Property files
    prop_files = {
        'EN': 'electronegativity_40_elements.txt',
        'RE': 'relative_electronegativity_40-elements.txt',
        'RA': 'radius_single_map_40_elements.txt',
        'SP': 'spin_40-elements.txt'
    }
    
    # Load properties
    print("\nLoading property data...")
    prop_data = {}
    elements = pressures = None
    
    for prop_name, filename in prop_files.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing: {filename}")
        
        elem_list, press, data_dict = parse_property_file(filename)
        prop_data[prop_name] = data_dict
        
        if elements is None:
            elements = sorted(elem_list)
            pressures = press
            print(f"  {len(elements)} elements, {len(pressures)} pressure points")
        
        print(f"  Loaded {prop_name}: {len(data_dict)} elements")
    
    # Compute features
    print(f"\nComputing {mode} features...")
    X, feature_names, element_pairs = compute_features_for_mode(
        prop_data, elements, pressures, mode
    )
    
    # Filter constant features
    variances = np.var(X, axis=0)
    non_constant = variances > 1e-10
    n_removed = np.sum(~non_constant)
    if n_removed > 0:
        print(f"  Removed {n_removed} constant features")
        X = X[:, non_constant]
        feature_names = [fn for fn, nc in zip(feature_names, non_constant) if nc]
    
    print(f"  Generated {X.shape[1]} features for {X.shape[0]} pairs")
    
    # Load targets
    print("\nLoading targets...")
    target_df = pd.read_csv('binary_mixing_enthalpies_780_pairs.csv')
    
    y = []
    matched_indices = []
    enthalpies = []
    
    for idx, (e1, e2) in enumerate(element_pairs):
        match = target_df[
            ((target_df['Element1'] == e1) & (target_df['Element2'] == e2)) |
            ((target_df['Element1'] == e2) & (target_df['Element2'] == e1))
        ]
        if len(match) > 0:
            enthalpy = match['avg mixing enthalpy in kJ per mole'].values[0]
            y.append(1 if enthalpy < 0 else 0)
            matched_indices.append(idx)
            enthalpies.append(enthalpy)
    
    X = X[matched_indices]
    element_pairs = [element_pairs[i] for i in matched_indices]
    y = np.array(y)
    enthalpies = np.array(enthalpies)
    
    # Create groups
    groups = create_pair_groups(element_pairs)
    
    # Build metadata
    feature_metadata = build_feature_metadata(feature_names, mode)
    
    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Summary
    n_pressure = sum(1 for fn in feature_names if feature_metadata[fn]['pressure_dependent'])
    print(f"\nDataset summary:")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Pressure-dependent: {n_pressure} ({100*n_pressure/len(feature_names):.1f}%)")
    print(f"  Static: {len(feature_names) - n_pressure} ({100*(len(feature_names)-n_pressure)/len(feature_names):.1f}%)")
    print(f"  Favorable (ΔH<0): {np.sum(y==1)} ({100*np.sum(y==1)/len(y):.1f}%)")
    
    # Save outputs
    if save_outputs:
        np.save(f'X_{mode}.npy', X)
        np.save('y_labels.npy', y)
        np.save('groups.npy', groups)
        np.save('enthalpies.npy', enthalpies)
        np.save('element_pairs.npy', np.array(element_pairs))
        
        with open(f'feature_names_{mode}.json', 'w') as f:
            json.dump(feature_names, f, indent=2, cls=NumpyEncoder)
        
        with open(f'feature_metadata_{mode}.json', 'w') as f:
            json.dump({
                'mode': mode,
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'n_pressure_dependent': int(n_pressure),
                'n_static': int(len(feature_names) - n_pressure),
                'feature_info': feature_metadata,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ Saved: X_{mode}.npy, feature_names_{mode}.json, feature_metadata_{mode}.json")
    
    return X, y, groups, feature_names, feature_metadata


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    run_feature_engineering(mode=mode)
