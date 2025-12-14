# verifier.py
"""
Model Verification and Evolutionary Loop (V2.5)
================================================

Handles:
- Safe code execution
- Model evaluation with GroupShuffleSplit
- Evolutionary convergence tracking with diversity metrics
- Fitted model storage for SHAP analysis
- Feedback generation

V2.5 Updates:
- Added diversity-based early stopping
- Store FITTED model (not factory) for SHAP
- Better error capture and reporting
- RF baseline comparison

Author: XCE Framework
Version: 2.5
"""

import numpy as np
import json
import os
import uuid
import traceback
import importlib.util
from datetime import datetime
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


class EvolutionTracker:
    """
    Track evolutionary progress and detect convergence.
    
    V2.5 Convergence criteria:
    - Ceiling: composite > ceiling_score
    - Plateau: < min_improvement for patience rounds
    - Diversity: agent score std < diversity_threshold for diversity_patience rounds
    - Maximum: max_rounds reached
    """
    
    def __init__(self, max_rounds=8, patience=3, min_improvement=0.01, 
                 ceiling_score=2.4, diversity_threshold=0.05, diversity_patience=2):
        self.max_rounds = max_rounds
        self.patience = patience
        self.min_improvement = min_improvement
        self.ceiling_score = ceiling_score
        self.diversity_threshold = diversity_threshold
        self.diversity_patience = diversity_patience
        
        self.history = []
        self.best_score = -np.inf
        self.best_agent = None
        self.best_round = None
        self.best_result = None  # Store full result including fitted model
        self.rounds_without_improvement = 0
        self.low_diversity_rounds = 0
    
    def record(self, round_num, results):
        """Record round results and track diversity."""
        successful = {
            agent: data['metrics']['composite']
            for agent, data in results.items()
            if data.get('success')
        }
        
        if not successful:
            self.history.append({
                'round': round_num,
                'best_score': 0,
                'best_agent': None,
                'n_successful': 0,
                'agent_std': 0,
                'global_best': self.best_score
            })
            self.rounds_without_improvement += 1
            return
        
        best_score = max(successful.values())
        best_agent = max(successful, key=successful.get)
        agent_std = np.std(list(successful.values())) if len(successful) > 1 else 0
        
        # Track diversity
        if agent_std < self.diversity_threshold and len(successful) > 1:
            self.low_diversity_rounds += 1
        else:
            self.low_diversity_rounds = 0
        
        # Check improvement and store best result
        if best_score > self.best_score + self.min_improvement:
            self.best_score = best_score
            self.best_agent = best_agent
            self.best_round = round_num
            self.best_result = results[best_agent]  # Store full result with model
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        
        self.history.append({
            'round': round_num,
            'best_score': best_score,
            'best_agent': best_agent,
            'mean_score': np.mean(list(successful.values())),
            'agent_std': agent_std,
            'n_successful': len(successful),
            'global_best': self.best_score
        })
    
    def should_continue(self):
        """Check if evolution should continue."""
        if not self.history:
            return True
        
        current_round = len(self.history)
        
        # Ceiling reached
        if self.best_score >= self.ceiling_score:
            print(f"  ★ Ceiling reached: {self.best_score:.3f} >= {self.ceiling_score}")
            return False
        
        # Plateau detected
        if self.rounds_without_improvement >= self.patience:
            print(f"  ★ Plateau: {self.patience} rounds without improvement > {self.min_improvement}")
            return False
        
        # Diversity convergence (V2.5)
        if self.low_diversity_rounds >= self.diversity_patience:
            print(f"  ★ Agent convergence: std < {self.diversity_threshold} for {self.diversity_patience} rounds")
            return False
        
        # Max rounds
        if current_round >= self.max_rounds:
            print(f"  ★ Maximum rounds reached: {current_round}")
            return False
        
        return True
    
    def get_best_fitted_model(self):
        """Get the best fitted model (V2.5 - critical for SHAP)."""
        if self.best_result and self.best_result.get('fitted_model'):
            return self.best_result['fitted_model']
        elif self.best_result and self.best_result.get('model'):
            return self.best_result['model']
        return None
    
    def get_summary(self):
        """Get evolution summary for synthesis."""
        if not self.history:
            return "No evolution rounds completed."
        
        lines = [
            f"Evolution completed in {len(self.history)} rounds.",
            f"Best composite: {self.best_score:.3f} ({self.best_agent}, round {self.best_round})",
            "",
            "Round progression:"
        ]
        
        for h in self.history:
            diversity_note = f" [std={h['agent_std']:.3f}]" if h.get('agent_std', 0) > 0 else ""
            lines.append(f"  R{h['round']}: best={h['best_score']:.3f}, winner={h['best_agent']}{diversity_note}")
        
        return '\n'.join(lines)


def execute_code_safely(code):
    """Execute agent code and return model."""
    # Basic safety check
    dangerous = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__']
    for pattern in dangerous:
        if pattern in code:
            return None, f"Dangerous pattern: {pattern}"
    
    # Create temp module
    module_name = f"agent_{uuid.uuid4().hex[:8]}"
    temp_file = f"/tmp/{module_name}.py"
    
    try:
        with open(temp_file, 'w') as f:
            f.write(code)
        
        spec = importlib.util.spec_from_file_location(module_name, temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'get_model'):
            return None, "No get_model() function found"
        
        model = module.get_model()
        return model, None
        
    except Exception as e:
        return None, f"Execution error: {str(e)}"
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def evaluate_model(model, X, y, groups, n_splits=5, return_fitted=True):
    """
    Evaluate model with GroupShuffleSplit CV.
    
    V2.5: Optionally returns a fitted model for SHAP analysis.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=RANDOM_STATE)
    
    f1_scores, mcc_scores, auc_scores = [], [], []
    fold_errors = []
    best_fold_model = None
    best_fold_score = -np.inf
    
    for fold_idx, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        try:
            fold_model = clone(model)
            fold_model.fit(X[train_idx], y[train_idx])
            
            y_pred = fold_model.predict(X[test_idx])
            
            if hasattr(fold_model, 'predict_proba'):
                y_prob = fold_model.predict_proba(X[test_idx])[:, 1]
            else:
                y_prob = fold_model.decision_function(X[test_idx])
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
            
            f1 = f1_score(y[test_idx], y_pred)
            mcc = matthews_corrcoef(y[test_idx], y_pred)
            auc = roc_auc_score(y[test_idx], y_prob)
            composite = f1 + mcc + auc
            
            f1_scores.append(f1)
            mcc_scores.append(mcc)
            auc_scores.append(auc)
            
            # Track best fold model for SHAP (V2.5)
            if composite > best_fold_score:
                best_fold_score = composite
                best_fold_model = fold_model
            
        except Exception as e:
            fold_errors.append(f"Fold {fold_idx}: {type(e).__name__}: {str(e)[:100]}")
            continue
    
    if len(f1_scores) < 2:
        error_detail = "; ".join(fold_errors[:3]) if fold_errors else "Unknown error"
        return None, f"Too few successful folds ({len(f1_scores)}/{n_splits}). Errors: {error_detail}", None
    
    # Bootstrap CI
    composites = np.array(f1_scores) + np.array(mcc_scores) + np.array(auc_scores)
    bootstrap = [np.mean(np.random.choice(composites, len(composites), replace=True)) 
                 for _ in range(1000)]
    
    metrics = {
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'mcc_mean': float(np.mean(mcc_scores)),
        'mcc_std': float(np.std(mcc_scores)),
        'auc_mean': float(np.mean(auc_scores)),
        'auc_std': float(np.std(auc_scores)),
        'composite': float(np.mean(composites)),
        'composite_std': float(np.std(composites)),
        'composite_ci_95': [float(np.percentile(bootstrap, 2.5)),
                           float(np.percentile(bootstrap, 97.5))],
        'n_folds': len(f1_scores)
    }
    
    # Return fitted model if requested (V2.5)
    if return_fitted:
        return metrics, None, best_fold_model
    return metrics, None, None


def fit_model_on_full_data(model, X, y):
    """
    Fit model on full dataset for SHAP analysis.
    
    V2.5: This ensures we have a properly fitted model for explainability.
    """
    try:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        fitted = clone(model)
        fitted.fit(X, y)
        return fitted, None
    except Exception as e:
        return None, f"Failed to fit on full data: {str(e)}"


def verify_and_evaluate(code, agent, X, y, groups):
    """
    Full verification pipeline.
    
    V2.5: Returns fitted model for SHAP analysis.
    """
    result = {
        'agent': agent,
        'success': False,
        'code': code,
        'detected_model': None,
        'metrics': None,
        'error': None,
        'model': None,
        'fitted_model': None  # V2.5: Fitted model for SHAP
    }
    
    # Detect model type
    model_types = [
        'RandomForestClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier',
        'LogisticRegression', 'RidgeClassifier', 'SGDClassifier',
        'SVC', 'KNeighborsClassifier', 'GaussianProcessClassifier'
    ]
    for mt in model_types:
        if mt in code:
            result['detected_model'] = mt
            break
    
    # Execute
    model, error = execute_code_safely(code)
    if error:
        result['error'] = error
        return result
    
    # Evaluate with fitted model return (V2.5)
    metrics, error, fitted_model = evaluate_model(model, X, y, groups, return_fitted=True)
    if error:
        result['error'] = error
        return result
    
    result['success'] = True
    result['metrics'] = metrics
    result['model'] = model
    result['fitted_model'] = fitted_model  # V2.5: Store fitted model
    
    return result


def generate_feedback(result, validation_result=None, baseline_score=None):
    """
    Generate feedback for next round.
    
    V2.5: Added baseline comparison.
    """
    lines = []
    
    # Validation feedback
    if validation_result and not validation_result['valid']:
        lines.append("VALIDATION ERRORS (FIX THESE FIRST):")
        for error in validation_result['errors']:
            lines.append(f"  ✗ {error}")
        lines.append("")
    
    if result['success']:
        m = result['metrics']
        lines.append(f"✓ Model executed: {result['detected_model']}")
        lines.append(f"  F1:  {m['f1_mean']:.3f} ± {m['f1_std']:.3f}")
        lines.append(f"  MCC: {m['mcc_mean']:.3f} ± {m['mcc_std']:.3f}")
        lines.append(f"  AUC: {m['auc_mean']:.3f} ± {m['auc_std']:.3f}")
        lines.append(f"  Composite: {m['composite']:.3f} ± {m['composite_std']:.3f}")
        lines.append(f"  95% CI: [{m['composite_ci_95'][0]:.3f}, {m['composite_ci_95'][1]:.3f}]")
        
        # Baseline comparison (V2.5)
        if baseline_score:
            diff = m['composite'] - baseline_score
            if diff > 0:
                lines.append(f"  vs RF Baseline: +{diff:.3f} (BETTER)")
            else:
                lines.append(f"  vs RF Baseline: {diff:.3f} (WORSE - baseline={baseline_score:.3f})")
        
        # Suggestions
        lines.append("\nSUGGESTIONS:")
        if m['mcc_mean'] < 0.5:
            lines.append("  - MCC is low: try stronger regularization or feature selection")
        if m['composite'] < 2.2:
            lines.append("  - Room for improvement: adjust hyperparameters")
        if m['composite'] >= 2.3:
            lines.append("  - Good performance! Fine-tune for marginal gains")
    else:
        lines.append(f"✗ Model FAILED: {result['error']}")
        lines.append("\nREQUIRED FIXES:")
        lines.append("  - Ensure get_model() returns valid sklearn estimator")
        lines.append("  - Check syntax and imports")
        lines.append("  - Use Pipeline for preprocessing")
        lines.append("  - Do NOT use lambda functions that require 'y' argument")
    
    return '\n'.join(lines)


def save_round_results(results, round_num, output_dir='results'):
    """Save round results."""
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {}
    for agent, data in results.items():
        save_data[agent] = {
            'success': data['success'],
            'detected_model': data.get('detected_model'),
            'error': data.get('error'),
            'metrics': data.get('metrics')
        }
    
    filepath = os.path.join(output_dir, f'round_{round_num}.json')
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    
    return filepath
