# portable_generator.py
"""
Portable Predictor Generator (V2.6)
====================================

Generates a fully self-contained prediction script from the best trained model.
The generated script includes:
- Embedded model (base64 serialized)
- Complete feature engineering (exact 83 features)
- Command-line interface for predictions

Usage:
    from portable_generator import generate_portable_predictor
    generate_portable_predictor(fitted_model, X, y, output_dir='results')

Author: XCE Framework
Version: 2.6
"""

import pickle
import base64
import numpy as np
import os
import json
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


# Exact 83 feature names from training pipeline (sorted alphabetically)
# EN: 25 features, RA: 25 features, RE: 20 features (no ambient), SP: 13 features
FEATURE_NAMES_83 = [
    "EN_A_curv_max", "EN_A_curv_mean", "EN_A_grad_max", "EN_A_grad_mean", "EN_A_initial",
    "EN_A_range", "EN_A_std", "EN_B_curv_max", "EN_B_curv_mean", "EN_B_grad_max",
    "EN_B_grad_mean", "EN_B_initial", "EN_B_range", "EN_B_std", "EN_delta_abs_initial",
    "EN_delta_auc", "EN_delta_initial", "EN_delta_range", "EN_delta_rmse", "EN_delta_squared_mean",
    "EN_delta_std", "EN_grad_delta_mean", "EN_grad_product_mean", "EN_n_crossings", "EN_ratio_initial",
    "RA_A_curv_max", "RA_A_curv_mean", "RA_A_grad_max", "RA_A_grad_mean", "RA_A_initial",
    "RA_A_range", "RA_A_std", "RA_B_curv_max", "RA_B_curv_mean", "RA_B_grad_max",
    "RA_B_grad_mean", "RA_B_initial", "RA_B_range", "RA_B_std", "RA_delta_abs_initial",
    "RA_delta_auc", "RA_delta_initial", "RA_delta_range", "RA_delta_rmse", "RA_delta_squared_mean",
    "RA_delta_std", "RA_grad_delta_mean", "RA_grad_product_mean", "RA_n_crossings", "RA_ratio_initial",
    "RE_A_curv_max", "RE_A_curv_mean", "RE_A_grad_max", "RE_A_grad_mean",
    "RE_A_range", "RE_A_std", "RE_B_curv_max", "RE_B_curv_mean", "RE_B_grad_max",
    "RE_B_grad_mean", "RE_B_range", "RE_B_std",
    "RE_delta_auc", "RE_delta_range", "RE_delta_rmse", "RE_delta_squared_mean",
    "RE_delta_std", "RE_grad_delta_mean", "RE_grad_product_mean", "RE_n_crossings",
    "SP_A_has_transition", "SP_A_initial", "SP_A_n_transitions", "SP_A_transition_pressure_norm",
    "SP_B_has_transition", "SP_B_initial", "SP_B_n_transitions", "SP_B_transition_pressure_norm",
    "SP_both_transition", "SP_delta_initial", "SP_either_transition", "SP_total_change_A", "SP_total_change_B"
]


def generate_portable_predictor(fitted_model, X, y, output_dir='results', 
                                 model_name=None, metrics=None):
    """
    Generate a portable prediction script from a trained model.
    
    Args:
        fitted_model: Trained sklearn model or Pipeline
        X: Training feature matrix (for validation)
        y: Training labels (for validation)
        output_dir: Output directory for the script
        model_name: Name of the model (e.g., 'GradientBoostingClassifier')
        metrics: Dict with performance metrics (optional)
    
    Returns:
        Dict with generation status and file path
    """
    print("\n" + "="*60)
    print("GENERATING PORTABLE PREDICTOR")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate inputs
    if fitted_model is None:
        print("  ✗ Error: No fitted model provided")
        return {'success': False, 'error': 'No fitted model'}
    
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    print(f"\n  Training data: {n_samples} samples, {n_features} features")
    
    # Validate feature count
    if n_features != 83:
        print(f"  ⚠ Warning: Expected 83 features, got {n_features}")
        print(f"  Will use actual feature count: {n_features}")
    
    # Detect model type
    if model_name is None:
        if hasattr(fitted_model, 'named_steps'):
            # Pipeline - get classifier name
            clf_name = list(fitted_model.named_steps.keys())[-1]
            clf = fitted_model.named_steps[clf_name]
            model_name = type(clf).__name__
        else:
            model_name = type(fitted_model).__name__
    
    print(f"  Model type: {model_name}")
    
    # Serialize model to base64
    print("\n  Serializing model...")
    try:
        model_bytes = pickle.dumps(fitted_model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        print(f"  ✓ Model size: {len(model_b64):,} chars ({len(model_bytes)/1024:.1f} KB)")
    except Exception as e:
        print(f"  ✗ Serialization failed: {e}")
        return {'success': False, 'error': f'Serialization failed: {e}'}
    
    # Compute training accuracy
    try:
        train_acc = fitted_model.score(X, y)
        print(f"  ✓ Training accuracy: {train_acc:.1%}")
    except:
        train_acc = None
    
    # Format metrics string
    metrics_str = ""
    if metrics:
        metrics_str = f"F1={metrics.get('f1_mean', 0):.3f}, AUC={metrics.get('auc_mean', 0):.3f}"
    elif train_acc:
        metrics_str = f"Accuracy={train_acc:.3f}"
    
    # Generate the script
    print("\n  Generating predict_portable.py...")
    
    script = _generate_script_content(
        model_b64=model_b64,
        n_samples=n_samples,
        n_features=n_features,
        model_name=model_name,
        metrics_str=metrics_str
    )
    
    # Write the file
    output_path = os.path.join(output_dir, 'predict_portable.py')
    with open(output_path, 'w') as f:
        f.write(script)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"  ✓ Generated: predict_portable.py ({file_size:.1f} KB)")
    
    # Save generation metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_name,
        'n_features': n_features,
        'n_samples': n_samples,
        'model_size_kb': len(model_bytes) / 1024,
        'training_accuracy': train_acc,
        'metrics': metrics
    }
    
    with open(os.path.join(output_dir, 'portable_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    
    print(f"  ✓ Saved: portable_metadata.json")
    
    print("\n" + "="*60)
    print("PORTABLE PREDICTOR GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"""
  Usage:
    python {output_path} Fe La
    python {output_path} --all --folder ./sharc_data/
    python {output_path} --list --folder ./sharc_data/
""")
    
    return {
        'success': True,
        'path': output_path,
        'model_type': model_name,
        'size_kb': file_size
    }


def _generate_script_content(model_b64, n_samples, n_features, model_name, metrics_str):
    """Generate the full script content."""
    
    feature_names_str = repr(FEATURE_NAMES_83)
    
    script = f'''#!/usr/bin/env python3
"""
XCE V2.6 - PORTABLE Binary Alloy Mixing Predictor
===================================================

Fully self-contained predictor. Works on ANY computer.

REQUIREMENTS:
    pip install numpy scipy scikit-learn

USAGE:
    python predict_portable.py Fe La
    python predict_portable.py Fe La --folder ./sharc_data/
    python predict_portable.py --all --folder ./sharc_data/
    python predict_portable.py --list --folder ./sharc_data/

FILE NAMING (SHARC format):
    electronegativity_La.txt
    relative_electronegativity_La.txt
    radius_La.txt
    spin_La.txt

MODEL: {model_name} ({metrics_str})
TRAINING: {n_samples} pairs, {n_features} features
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import numpy as np
from scipy.interpolate import interp1d
import pickle
import base64
import re
import os
import glob
import json
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# =============================================================================
# EMBEDDED MODEL
# =============================================================================

MODEL_B64 = """{model_b64}"""

_CACHED_MODEL = None

def load_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = pickle.loads(base64.b64decode(MODEL_B64))
    return _CACHED_MODEL

# =============================================================================
# CONSTANTS
# =============================================================================

STANDARD_PRESSURES = np.linspace(0, 300, 601)

# Exact 83 feature names (sorted alphabetically as in training)
# EN: 25 features, RA: 25 features, RE: 20 features (no ambient), SP: 13 features
FEATURE_NAMES = {feature_names_str}

PROPERTY_FILES = {{
    'EN': 'electronegativity_{{elem}}.txt',
    'RE': 'relative_electronegativity_{{elem}}.txt',
    'RA': 'radius_{{elem}}.txt',
    'SP': 'spin_{{elem}}.txt'
}}

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def get_element_files(element, folder='.'):
    files = {{}}
    for prop, pattern in PROPERTY_FILES.items():
        filename = pattern.format(elem=element)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing: {{filepath}}")
        files[prop] = filepath
    return files


def list_available_elements(folder='.'):
    pattern = os.path.join(folder, 'electronegativity_*.txt')
    en_files = glob.glob(pattern)
    
    elements = []
    for f in en_files:
        match = re.search(r'electronegativity_([A-Za-z]+)\\.txt', os.path.basename(f))
        if match:
            elem = match.group(1)
            try:
                get_element_files(elem, folder)
                elements.append(elem)
            except FileNotFoundError:
                pass
    return sorted(elements)


def parse_sharc_file(filepath):
    """Parse SHARC file and interpolate to standard grid."""
    pressures = []
    values = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pressures.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    
    if len(pressures) == 0:
        raise ValueError(f"No data in {{filepath}}")
    
    # Handle duplicates by averaging
    unique_p, unique_v = [], []
    i = 0
    while i < len(pressures):
        p = pressures[i]
        same = [values[i]]
        j = i + 1
        while j < len(pressures) and pressures[j] == p:
            same.append(values[j])
            j += 1
        unique_p.append(p)
        unique_v.append(np.mean(same))
        i = j
    
    f = interp1d(unique_p, unique_v, bounds_error=False, 
                 fill_value=(unique_v[0], unique_v[-1]))
    return f(STANDARD_PRESSURES)

# =============================================================================
# FEATURE ENGINEERING (EXACT MATCH TO TRAINING)
# =============================================================================

def compute_ambient_features(f1, f2, prop_name):
    """Compute ambient features (5 total) - ONLY for EN and RA, NOT for RE."""
    features = {{}}
    features[f"{{prop_name}}_A_initial"] = f1[0]
    features[f"{{prop_name}}_B_initial"] = f2[0]
    features[f"{{prop_name}}_delta_initial"] = f1[0] - f2[0]
    features[f"{{prop_name}}_delta_abs_initial"] = np.abs(f1[0] - f2[0])
    if np.abs(f2[0]) > 1e-10:
        features[f"{{prop_name}}_ratio_initial"] = f1[0] / f2[0]
    else:
        features[f"{{prop_name}}_ratio_initial"] = 1.0
    return features


def compute_pressure_features(f1, f2, prop_name, pressures):
    """Compute pressure-dependent features (20 total)."""
    features = {{}}
    dx = np.mean(np.diff(pressures)) if len(pressures) > 1 else 1.0
    
    # Element A (6)
    features[f"{{prop_name}}_A_range"] = np.ptp(f1)
    features[f"{{prop_name}}_A_std"] = np.std(f1)
    grad1 = np.gradient(f1, dx)
    features[f"{{prop_name}}_A_grad_mean"] = np.mean(grad1)
    features[f"{{prop_name}}_A_grad_max"] = np.max(np.abs(grad1))
    curv1 = np.gradient(grad1, dx)
    features[f"{{prop_name}}_A_curv_mean"] = np.mean(curv1)
    features[f"{{prop_name}}_A_curv_max"] = np.max(np.abs(curv1))
    
    # Element B (6)
    features[f"{{prop_name}}_B_range"] = np.ptp(f2)
    features[f"{{prop_name}}_B_std"] = np.std(f2)
    grad2 = np.gradient(f2, dx)
    features[f"{{prop_name}}_B_grad_mean"] = np.mean(grad2)
    features[f"{{prop_name}}_B_grad_max"] = np.max(np.abs(grad2))
    curv2 = np.gradient(grad2, dx)
    features[f"{{prop_name}}_B_curv_mean"] = np.mean(curv2)
    features[f"{{prop_name}}_B_curv_max"] = np.max(np.abs(curv2))
    
    # Pairwise (8)
    diff = f1 - f2
    features[f"{{prop_name}}_delta_range"] = np.ptp(diff)
    features[f"{{prop_name}}_delta_std"] = np.std(diff)
    features[f"{{prop_name}}_delta_squared_mean"] = np.mean(diff**2)
    features[f"{{prop_name}}_delta_rmse"] = np.sqrt(np.mean(diff**2))
    features[f"{{prop_name}}_grad_delta_mean"] = np.mean(grad1 - grad2)
    features[f"{{prop_name}}_grad_product_mean"] = np.mean(grad1 * grad2)
    features[f"{{prop_name}}_n_crossings"] = float(np.sum(np.diff(np.sign(diff)) != 0))
    
    try:
        auc = np.trapezoid(np.abs(diff), pressures)
    except AttributeError:
        auc = np.trapz(np.abs(diff), pressures)
    features[f"{{prop_name}}_delta_auc"] = auc / (pressures[-1] - pressures[0] + 1e-10)
    
    return features


def compute_spin_features(f1, f2, threshold=0.5):
    """Compute spin features (13 total: 3 ambient + 10 pressure)."""
    features = {{}}
    
    # Ambient (3)
    features["SP_A_initial"] = f1[0]
    features["SP_B_initial"] = f2[0]
    features["SP_delta_initial"] = f1[0] - f2[0]
    
    # Pressure (10)
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


def compute_all_features(data_A, data_B):
    """
    Compute all 83 features for a pair of elements.
    
    Feature breakdown:
    - EN: 5 ambient + 20 pressure = 25
    - RA: 5 ambient + 20 pressure = 25
    - RE: 0 ambient + 20 pressure = 20 (NO AMBIENT for RE!)
    - SP: 3 ambient + 10 pressure = 13
    Total: 25 + 25 + 20 + 13 = 83
    """
    pressures = STANDARD_PRESSURES
    row = {{}}
    
    # EN: ambient + pressure (25 features)
    row.update(compute_ambient_features(data_A['EN'], data_B['EN'], 'EN'))
    row.update(compute_pressure_features(data_A['EN'], data_B['EN'], 'EN', pressures))
    
    # RA: ambient + pressure (25 features)
    row.update(compute_ambient_features(data_A['RA'], data_B['RA'], 'RA'))
    row.update(compute_pressure_features(data_A['RA'], data_B['RA'], 'RA', pressures))
    
    # RE: pressure ONLY, NO ambient (20 features)
    row.update(compute_pressure_features(data_A['RE'], data_B['RE'], 'RE', pressures))
    
    # SP: ambient + pressure (13 features)
    row.update(compute_spin_features(data_A['SP'], data_B['SP']))
    
    # Extract features in EXACT order (sorted alphabetically)
    features = [row[fn] for fn in FEATURE_NAMES]
    
    return np.array(features)

# =============================================================================
# PREDICTION
# =============================================================================

def predict_pair(elem_A, elem_B, folder='.'):
    """Predict mixing behavior for a pair of elements."""
    # Alphabetical ordering (must match training)
    if elem_A > elem_B:
        elem_A, elem_B = elem_B, elem_A
    
    # Get files
    files_A = get_element_files(elem_A, folder)
    files_B = get_element_files(elem_B, folder)
    
    # Load data
    data_A = {{prop: parse_sharc_file(path) for prop, path in files_A.items()}}
    data_B = {{prop: parse_sharc_file(path) for prop, path in files_B.items()}}
    
    # Compute features (exact 83 in correct order)
    features = compute_all_features(data_A, data_B)
    X = features.reshape(1, -1)
    X = np.nan_to_num(X)
    
    # Validate
    if X.shape[1] != 83:
        raise ValueError(f"Feature mismatch: got {{X.shape[1]}}, expected 83")
    
    # Predict
    model = load_model()
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    
    return {{
        'pair': f'{{elem_A}}-{{elem_B}}',
        'element_A': elem_A,
        'element_B': elem_B,
        'prediction': 'Favorable' if pred == 1 else 'Unfavorable',
        'prob_favorable': round(float(prob[1]), 4),
        'prob_unfavorable': round(float(prob[0]), 4),
        'confidence': round(float(max(prob)), 4)
    }}


def predict_all_pairs(folder='.'):
    """Predict all possible pairs."""
    elements = list_available_elements(folder)
    if len(elements) < 2:
        raise ValueError(f"Need at least 2 elements, found: {{elements}}")
    
    n_pairs = len(elements) * (len(elements) - 1) // 2
    print(f"\\nPredicting {{n_pairs}} pairs from {{len(elements)}} elements...")
    print("-" * 60)
    
    results = []
    for i, (e1, e2) in enumerate(combinations(elements, 2)):
        try:
            result = predict_pair(e1, e2, folder)
            results.append(result)
            status = "✓" if result['prediction'] == 'Favorable' else "✗"
            print(f"  {{i+1:3d}}/{{n_pairs}}: {{result['pair']:8s}} → {{result['prediction']:11s}} ({{result['confidence']:.1%}}) {{status}}")
        except Exception as ex:
            print(f"  {{i+1:3d}}/{{n_pairs}}: {{e1}}-{{e2}} → ERROR: {{ex}}")
    
    return results

# =============================================================================
# OUTPUT
# =============================================================================

def print_result(r):
    symbol = "✓" if r['prediction'] == 'Favorable' else "✗"
    print()
    print("=" * 60)
    print(f"  PREDICTION: {{r['pair']}}")
    print("=" * 60)
    print(f"""
  Element A:      {{r['element_A']}}
  Element B:      {{r['element_B']}}
  
  ┌────────────────────────────────────────────────────┐
  │  RESULT:     {{r['prediction'].upper():^38}} │
  │  CONFIDENCE: {{f"{{r['confidence']:.1%}}":^38}} │
  └────────────────────────────────────────────────────┘
  
  P(Favorable):   {{r['prob_favorable']:.1%}}
  P(Unfavorable): {{r['prob_unfavorable']:.1%}}
  
  {{symbol}} {{"Elements predicted to MIX (ΔH < 0)" if r['prediction'] == 'Favorable' else "Elements predicted NOT to mix (ΔH ≥ 0)"}}
""")
    print("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='XCE Portable Binary Alloy Mixing Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_portable.py Fe La
  python predict_portable.py Fe La --folder ./data/
  python predict_portable.py --list --folder ./data/
  python predict_portable.py --all --folder ./data/
"""
    )
    
    parser.add_argument('elements', nargs='*', help='Two element symbols')
    parser.add_argument('--folder', '-f', default='.', help='Folder with SHARC files')
    parser.add_argument('--list', '-l', action='store_true', help='List available elements')
    parser.add_argument('--all', '-a', action='store_true', help='Predict all pairs')
    parser.add_argument('--output', '-o', help='Output JSON file')
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("  XCE V2.6 PORTABLE ALLOY PREDICTOR")
    print("=" * 60)
    
    if args.list:
        elements = list_available_elements(args.folder)
        print(f"\\nFound {{len(elements)}} elements in '{{args.folder}}':")
        for i, elem in enumerate(elements, 1):
            print(f"  {{i:2d}}. {{elem}}")
        return
    
    if args.all:
        results = predict_all_pairs(args.folder)
        if results:
            favorable = sum(1 for r in results if r['prediction'] == 'Favorable')
            print()
            print("=" * 60)
            print(f"  SUMMARY: {{favorable}}/{{len(results)}} favorable ({{100*favorable/len(results):.1f}}%)")
            print("=" * 60)
            
            output_file = args.output or 'all_predictions.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\nSaved: {{output_file}}")
            
            csv_file = output_file.replace('.json', '.csv')
            with open(csv_file, 'w') as f:
                f.write("pair,element_A,element_B,prediction,prob_favorable,prob_unfavorable,confidence\\n")
                for r in results:
                    f.write(f"{{r['pair']}},{{r['element_A']}},{{r['element_B']}},{{r['prediction']}},{{r['prob_favorable']}},{{r['prob_unfavorable']}},{{r['confidence']}}\\n")
            print(f"Saved: {{csv_file}}")
        return
    
    if len(args.elements) != 2:
        print("\\nUsage: python predict_portable.py Fe La")
        print("Run with --help for more options.")
        return
    
    elem_A, elem_B = args.elements
    print(f"\\nPredicting: {{elem_A}}-{{elem_B}}")
    
    try:
        result = predict_pair(elem_A, elem_B, args.folder)
        print_result(result)
        
        output_file = args.output or f"prediction_{{result['pair']}}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {{output_file}}")
        
    except FileNotFoundError as e:
        print(f"\\n  ✗ Error: {{e}}")
        print(f"\\n  Available elements:")
        for elem in list_available_elements(args.folder):
            print(f"    - {{elem}}")
    except Exception as e:
        print(f"\\n  ✗ Error: {{e}}")


if __name__ == '__main__':
    main()
'''
    
    return script
