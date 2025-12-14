# xai_utils.py
"""
SHAP Explainability Analysis (V2.6)
====================================

Generates feature importance analysis using SHAP values.
Now handles complex Pipelines and includes DIRECTIONAL information.

V2.6 Updates:
- Added directional SHAP (signed mean values, not just absolute)
- Direction indicates: positive = favors favorable mixing, negative = favors unfavorable
- Output includes: importance (magnitude), direction (signed), effect (+/−)

Author: XCE Framework
Version: 2.6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.base import clone

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


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


def run_shap_analysis(model, X, feature_names, feature_metadata,
                      output_dir='results', max_samples=500, y=None, groups=None):
    """
    Run SHAP analysis on trained model with DIRECTIONAL information.
    
    V2.6: Now returns both magnitude (importance) and direction (signed effect).
    
    Strategy:
    1. Try Pipeline-aware SHAP (explains in original feature space)
    2. If that fails, try extracting classifier and explaining transformed space
    3. If all fails, fall back to permutation importance
    
    Args:
        model: Trained sklearn model or Pipeline
        X: Feature matrix (original features)
        feature_names: List of original feature names
        feature_metadata: Dict with feature info
        output_dir: Output directory
        max_samples: Max samples for SHAP computation
        y: Labels (needed for fallback permutation importance)
        groups: Group labels (needed for fallback)
    
    Returns:
        Dict with importance dataframe (including direction) and summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS (V2.6 - Directional)")
    print("="*60)
    
    # Subsample for efficiency
    if X.shape[0] > max_samples:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]
        y_sample = y[idx] if y is not None else None
    else:
        X_sample = X
        y_sample = y
    
    # Clean data
    X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Try different SHAP strategies
    shap_result = None
    method_used = None
    
    # Strategy 1: Pipeline-aware SHAP (best - explains original features)
    if SHAP_AVAILABLE:
        print("\n  Attempting Pipeline-aware SHAP...")
        shap_result = try_pipeline_aware_shap(model, X_sample, feature_names)
        if shap_result is not None:
            method_used = "pipeline_aware_shap"
            print("  ✓ Pipeline-aware SHAP successful")
    
    # Strategy 2: Direct classifier SHAP (may work for simple pipelines)
    if shap_result is None and SHAP_AVAILABLE:
        print("\n  Attempting direct classifier SHAP...")
        shap_result = try_direct_classifier_shap(model, X_sample, feature_names)
        if shap_result is not None:
            method_used = "direct_classifier_shap"
            print("  ✓ Direct classifier SHAP successful")
    
    # Strategy 3: Permutation importance (fallback - always works, but no direction)
    if shap_result is None:
        print("\n  Falling back to permutation importance...")
        if y_sample is not None:
            shap_result = compute_permutation_importance(
                model, X_sample, y_sample, feature_names
            )
            if shap_result is not None:
                method_used = "permutation_importance"
                print("  ✓ Permutation importance computed")
                print("  ⚠ Note: Permutation importance does not provide direction")
        else:
            print("  ✗ Cannot compute permutation importance without labels")
    
    # If all strategies failed
    if shap_result is None:
        error_msg = "All importance methods failed. Check model compatibility."
        print(f"\n  ✗ {error_msg}")
        return {'error': error_msg, 'method': 'none'}
    
    # Process results - V2.6: now includes direction
    mean_importance, mean_direction, selected_features = shap_result
    
    print(f"\n  Method: {method_used}")
    print(f"  Features analyzed: {len(selected_features)}")
    
    # Build importance dataframe with direction
    feature_info = feature_metadata.get('feature_info', {})
    importance_data = []
    
    for i, fn in enumerate(selected_features):
        if i >= len(mean_importance):
            break
        info = feature_info.get(fn, {})
        
        # Direction: positive means higher value → more favorable
        direction = mean_direction[i] if mean_direction is not None else 0.0
        effect = "+" if direction > 0 else "-" if direction < 0 else "0"
        
        importance_data.append({
            'feature': fn,
            'importance': float(mean_importance[i]),
            'direction': float(direction),
            'effect': effect,
            'category': info.get('category', get_category(fn)),
            'pressure_dependent': info.get('pressure_dependent', is_pressure(fn))
        })
    
    imp_df = pd.DataFrame(importance_data)
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Save CSV
    imp_df.to_csv(os.path.join(output_dir, 'shap_importance.csv'), index=False)
    print(f"\n  ✓ Saved: shap_importance.csv")
    
    # Generate plots
    try:
        generate_importance_plots(imp_df, output_dir, method_used)
        generate_direction_plot(imp_df, output_dir)  # V2.6: New directional plot
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")
    
    # Summary statistics
    total_imp = imp_df['importance'].sum()
    pressure_imp = imp_df[imp_df['pressure_dependent']]['importance'].sum()
    
    # V2.6: Directional summary
    favorable_features = imp_df[imp_df['direction'] > 0]['feature'].tolist()[:5]
    unfavorable_features = imp_df[imp_df['direction'] < 0]['feature'].tolist()[:5]
    
    summary = {
        'method': method_used,
        'n_features': len(imp_df),
        'top_10_features': imp_df.head(10)[['feature', 'importance', 'direction', 'effect', 'category']].to_dict('records'),
        'category_importance': imp_df.groupby('category')['importance'].sum().to_dict(),
        'pressure_dependent_pct': float(100 * pressure_imp / total_imp) if total_imp > 0 else 0,
        'static_pct': float(100 * (total_imp - pressure_imp) / total_imp) if total_imp > 0 else 0,
        'direction_available': mean_direction is not None,
        'top_favorable_features': favorable_features,
        'top_unfavorable_features': unfavorable_features
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'shap_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"  ✓ Saved: shap_summary.json")
    print(f"\n  Pressure-dependent: {summary['pressure_dependent_pct']:.1f}%")
    print(f"  Static: {summary['static_pct']:.1f}%")
    
    if mean_direction is not None:
        print(f"\n  Top features favoring FAVORABLE mixing (+):")
        for fn in favorable_features[:3]:
            print(f"    + {fn}")
        print(f"  Top features favoring UNFAVORABLE mixing (-):")
        for fn in unfavorable_features[:3]:
            print(f"    - {fn}")
    
    return {
        'importance_df': imp_df,
        'feature_names': selected_features,
        'summary': summary,
        'method': method_used
    }


def try_pipeline_aware_shap(model, X, feature_names):
    """
    Try Pipeline-aware SHAP that explains in original feature space.
    V2.6: Returns both magnitude and direction.
    """
    try:
        # Create prediction function for the full pipeline
        if hasattr(model, 'predict_proba'):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            predict_fn = model.predict
        
        # Use independent masker for tabular data
        masker = shap.maskers.Independent(X, max_samples=100)
        
        # Create explainer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.Explainer(predict_fn, masker, feature_names=feature_names)
            
            # Compute SHAP values (limit samples for speed)
            n_explain = min(100, X.shape[0])
            shap_values = explainer(X[:n_explain])
        
        # Extract values
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = np.array(shap_values)
        
        # V2.6: Compute both magnitude (importance) and direction
        mean_importance = np.mean(np.abs(values), axis=0)
        mean_direction = np.mean(values, axis=0)  # Signed mean
        
        # Validate shape matches features
        if len(mean_importance) != len(feature_names):
            return None
        
        return mean_importance, mean_direction, feature_names
        
    except Exception as e:
        print(f"    Pipeline-aware SHAP failed: {str(e)[:100]}")
        return None


def try_direct_classifier_shap(model, X, feature_names):
    """
    Try extracting classifier from Pipeline and applying SHAP directly.
    V2.6: Returns both magnitude and direction.
    """
    try:
        # Find classifier and transform data
        if hasattr(model, 'named_steps'):
            X_transformed = X.copy()
            classifier = None
            selected_features = feature_names.copy()
            
            for name, step in model.named_steps.items():
                # Check if this is the final classifier
                if hasattr(step, 'predict') and hasattr(step, 'fit'):
                    # Check if it's a transformer or classifier
                    if hasattr(step, 'transform') and name != list(model.named_steps.keys())[-1]:
                        # It's a transformer, apply it
                        X_transformed = step.transform(X_transformed)
                        
                        # Handle feature selection
                        if hasattr(step, 'get_support'):
                            mask = step.get_support()
                            selected_features = [f for f, m in zip(selected_features, mask) if m]
                        # Handle feature expansion (PolynomialFeatures)
                        elif hasattr(step, 'get_feature_names_out'):
                            try:
                                selected_features = list(step.get_feature_names_out(selected_features))
                            except:
                                selected_features = [f"transformed_{i}" for i in range(X_transformed.shape[1])]
                    else:
                        classifier = step
                        break
                elif hasattr(step, 'transform'):
                    X_transformed = step.transform(X_transformed)
            
            if classifier is None:
                classifier = model.named_steps[list(model.named_steps.keys())[-1]]
        else:
            classifier = model
            X_transformed = X
            selected_features = feature_names
        
        # Check for feature dimension mismatch
        if X_transformed.shape[1] != len(selected_features):
            selected_features = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        # Apply appropriate SHAP explainer
        model_type = type(classifier).__name__
        
        if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier',
                         'ExtraTreesClassifier', 'HistGradientBoostingClassifier']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed[:100])
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (favorable)
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        else:
            background = shap.sample(X_transformed, min(50, X_transformed.shape[0]))
            if hasattr(classifier, 'predict_proba'):
                explainer = shap.KernelExplainer(
                    lambda x: classifier.predict_proba(x)[:, 1], background
                )
            else:
                explainer = shap.KernelExplainer(classifier.predict, background)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_transformed[:50])
        
        # V2.6: Calculate both magnitude and direction
        mean_importance = np.mean(np.abs(shap_values), axis=0)
        mean_direction = np.mean(shap_values, axis=0)  # Signed mean
        
        return mean_importance, mean_direction, selected_features
        
    except Exception as e:
        print(f"    Direct classifier SHAP failed: {str(e)[:100]}")
        return None


def compute_permutation_importance(model, X, y, feature_names, n_repeats=10):
    """
    Compute permutation importance as fallback.
    Note: Permutation importance does not provide direction.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sklearn_perm_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1,
                scoring='f1'
            )
        
        mean_importance = result.importances_mean
        
        # Normalize to positive values
        mean_importance = np.maximum(mean_importance, 0)
        
        # No direction available for permutation importance
        mean_direction = None
        
        return mean_importance, mean_direction, feature_names
        
    except Exception as e:
        print(f"    Permutation importance failed: {str(e)[:100]}")
        return None


def generate_importance_plots(imp_df, output_dir, method_used):
    """Generate importance visualization plots."""
    
    method_label = {
        'pipeline_aware_shap': 'SHAP (Pipeline-aware)',
        'direct_classifier_shap': 'SHAP (Classifier)',
        'permutation_importance': 'Permutation Importance'
    }.get(method_used, 'Feature Importance')
    
    # 1. Bar plot (top 15)
    plt.figure(figsize=(10, 8))
    top_15 = imp_df.head(15)
    colors = [get_color(cat) for cat in top_15['category']]
    plt.barh(range(len(top_15)), top_15['importance'].values, color=colors)
    plt.yticks(range(len(top_15)), top_15['feature'].values)
    plt.xlabel(f'Mean |{method_label}|')
    plt.title(f'Top 15 Features by {method_label}')
    plt.gca().invert_yaxis()
    
    # Legend
    categories = top_15['category'].unique()
    handles = [plt.Rectangle((0,0),1,1, color=get_color(c)) for c in categories]
    plt.legend(handles, categories, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_bar.png")
    
    # 2. Category plot
    plt.figure(figsize=(8, 6))
    cat_imp = imp_df.groupby('category')['importance'].sum().sort_values()
    colors = [get_color(c) for c in cat_imp.index]
    cat_imp.plot(kind='barh', color=colors)
    plt.xlabel(f'Total {method_label}')
    plt.title('Importance by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_category.png")
    
    # 3. Pressure vs Static pie chart
    plt.figure(figsize=(8, 6))
    pressure_imp = imp_df[imp_df['pressure_dependent']]['importance'].sum()
    static_imp = imp_df[~imp_df['pressure_dependent']]['importance'].sum()
    
    if pressure_imp + static_imp > 0:
        plt.pie([pressure_imp, static_imp], 
                labels=['Pressure-dependent', 'Static'],
                colors=['#E74C3C', '#3498DB'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Importance: Pressure-dependent vs Static Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_pressure_vs_static.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: shap_pressure_vs_static.png")


def generate_direction_plot(imp_df, output_dir):
    """V2.6: Generate directional SHAP plot showing +/− effects."""
    
    if 'direction' not in imp_df.columns or imp_df['direction'].isna().all():
        print("  ⚠ Direction data not available for plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Get top 20 by importance, then sort by direction for visualization
    top_20 = imp_df.head(20).copy()
    top_20 = top_20.sort_values('direction')
    
    # Color based on direction
    colors = ['#2ECC71' if d > 0 else '#E74C3C' for d in top_20['direction']]
    
    plt.barh(range(len(top_20)), top_20['direction'].values, color=colors)
    plt.yticks(range(len(top_20)), top_20['feature'].values)
    plt.xlabel('SHAP Direction (+ = favors favorable mixing, − = favors unfavorable)')
    plt.title('Feature Direction: Effect on Mixing Prediction')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', label='Favors Favorable (+)'),
        Patch(facecolor='#E74C3C', label='Favors Unfavorable (-)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_direction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_direction.png")


def get_category(fn):
    """Get category from feature name."""
    fn = fn.lower()
    if fn.startswith('en_'): return 'Electronegativity'
    if fn.startswith('re_'): return 'Relative_EN'
    if fn.startswith('ra_'): return 'Atomic_Radius'
    if fn.startswith('sp_'): return 'Spin_State'
    return 'Other'


def is_pressure(fn):
    """Check if feature is pressure-dependent."""
    keywords = ['grad', 'curv', 'range', 'std', 'auc', 'transition', 
                'crossings', 'rmse', 'squared', 'product']
    fn = fn.lower()
    if '_initial' in fn and 'delta' not in fn:
        return False
    return any(kw in fn for kw in keywords)


def get_color(category):
    """Get color for category."""
    colors = {
        'Electronegativity': '#E74C3C',
        'Relative_EN': '#3498DB',
        'Atomic_Radius': '#2ECC71',
        'Spin_State': '#9B59B6',
        'Other': '#95A5A6'
    }
    return colors.get(category, '#95A5A6')
