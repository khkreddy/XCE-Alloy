# deterministic_validator.py
"""
Deterministic Code Validation (V2.5)
====================================

Rule-based validation for agent-generated code.
NO LLM calls - all checks are deterministic regex/AST patterns.

V2.5 Key Improvements:
- AST check for parameterless get_model()
- Detection of y-dependent lambdas (GPT's main failure mode)
- Pipeline structure validation
- KNeighbors with weights='distance' allowed
- Better error messages with fix hints

Author: XCE Framework
Version: 2.5
"""

import re
import ast
from typing import Dict, Tuple, List

# =============================================================================
# AGENT MODEL FAMILY CONFIGURATION
# =============================================================================

AGENT_MODEL_FAMILIES = {
    'gpt': {
        'name': 'GPT-4o',
        'family': 'Tree-based (RF)',
        'allowed': [
            'RandomForestClassifier',  # V2.5.1: Restricted to RF only for reliability
        ],
        'requires_scaler': False  # RF doesn't require scaling but we include it for consistency
    },
    'claude': {
        'name': 'Claude Sonnet 4.5',
        'family': 'Linear',
        'allowed': [
            'LogisticRegression',
            'RidgeClassifier',
            'SGDClassifier',
            'Perceptron',
            'PassiveAggressiveClassifier',
            'LinearSVC'
        ],
        'requires_scaler': True
    },
    'gemini': {
        'name': 'Gemini 2.0 Flash',
        'family': 'Kernel/Distance',
        'allowed': [
            'SVC',
            'NuSVC',
            'KNeighborsClassifier',
            'RadiusNeighborsClassifier',
            'GaussianProcessClassifier'
        ],
        'requires_scaler': True
    }
}


# =============================================================================
# AST-BASED VALIDATION CHECKS (V2.5)
# =============================================================================

def check_get_model_parameterless(code: str) -> Tuple[bool, str]:
    """
    V2.5: Check that get_model() takes NO arguments using AST.
    
    GPT often generates get_model(X, y) or lambdas that need y.
    """
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_model':
                # Check arguments
                args = node.args
                total_args = (
                    len(args.args) + 
                    len(args.posonlyargs) + 
                    len(args.kwonlyargs)
                )
                
                # Allow 'self' for class methods, but nothing else
                if total_args > 0:
                    arg_names = [a.arg for a in args.args]
                    if arg_names != ['self']:
                        return False, f"get_model() must take NO arguments. Found: {arg_names}"
                
                return True, "get_model() is parameterless"
        
        return False, "No get_model() function found"
        
    except SyntaxError as e:
        return False, f"Syntax error prevents AST parsing: {e}"


def check_lambda_y_dependency(code: str) -> Tuple[bool, str]:
    """
    V2.5: Detect lambdas that depend on 'y' argument.
    
    This is GPT's main failure mode - creating lambdas like:
    FunctionTransformer(lambda X, y: SMOTE().fit_resample(X, y))
    
    sklearn Pipeline doesn't pass y to intermediate transformers.
    """
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                # Check lambda arguments
                args = [a.arg for a in node.args.args]
                if 'y' in args:
                    return False, "Lambda function takes 'y' argument. sklearn Pipeline does NOT pass y to transformers. Use imblearn.pipeline.Pipeline for SMOTE."
        
        return True, "No problematic y-dependent lambdas"
        
    except SyntaxError:
        return True, "Could not parse (syntax check will catch this)"


def check_function_transformer_with_y(code: str) -> Tuple[bool, str]:
    """
    V2.5: Check for FunctionTransformer patterns that need y.
    """
    # Pattern: FunctionTransformer with lambda that takes y
    if 'FunctionTransformer' in code:
        # Check for lambda with y
        pattern = r'FunctionTransformer\s*\([^)]*lambda[^:]*,\s*y[^:]*:'
        if re.search(pattern, code):
            return False, "FunctionTransformer with y-dependent lambda. sklearn Pipeline doesn't pass y. Use imblearn.pipeline.Pipeline for resampling."
    
    return True, "No problematic FunctionTransformer"


# =============================================================================
# STANDARD VALIDATION CHECKS
# =============================================================================

def check_syntax(code: str) -> Tuple[bool, str]:
    """Check Python syntax validity."""
    try:
        ast.parse(code)
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e.msg} at line {e.lineno}"


def check_dangerous_patterns(code: str) -> Tuple[bool, str]:
    """Check for dangerous code patterns."""
    dangerous = [
        ('os.system', 'System command execution'),
        ('subprocess', 'Subprocess execution'),
        ('eval(', 'Eval execution'),
        ('exec(', 'Exec execution'),
        ('__import__', 'Dynamic import'),
        ('open(', 'File operations'),
        ('requests.', 'Network requests'),
        ('urllib', 'Network requests'),
    ]
    
    for pattern, description in dangerous:
        if pattern in code:
            return False, f"Dangerous pattern detected: {description}"
    
    return True, "No dangerous patterns"


def check_get_model_function(code: str) -> Tuple[bool, str]:
    """Check for required get_model() function."""
    if 'def get_model(' in code or 'def get_model()' in code:
        return True, "get_model() function found"
    return False, "Missing get_model() function. Your code must define: def get_model():"


def check_model_family(code: str, agent: str) -> Tuple[bool, str]:
    """Check that agent uses models from allowed family."""
    config = AGENT_MODEL_FAMILIES.get(agent)
    if not config:
        return True, ""
    
    allowed = config['allowed']
    found_models = []
    
    for model in allowed:
        if model in code:
            found_models.append(model)
    
    if found_models:
        return True, f"Using allowed model(s): {', '.join(found_models)}"
    
    # Check if they're using a disallowed model
    all_models = []
    for cfg in AGENT_MODEL_FAMILIES.values():
        all_models.extend(cfg['allowed'])
    
    used_wrong = [m for m in all_models if m in code and m not in allowed]
    if used_wrong:
        return False, f"Model {used_wrong[0]} not allowed for {agent}. Use: {', '.join(allowed[:3])}"
    
    return False, f"No valid model found. {agent} must use: {', '.join(allowed[:3])}"


def check_class_imbalance_handling(code: str, agent: str = None) -> Tuple[bool, str]:
    """
    Check for class imbalance handling.
    
    V2.5: KNeighbors allowed with weights='distance'.
    V2.5.1: GradientBoosting allowed with subsample or sample_weight.
    """
    # Check for KNeighbors with distance weighting (valid approach)
    if 'KNeighborsClassifier' in code:
        if re.search(r"weights\s*=\s*['\"]distance['\"]", code):
            return True, "KNeighbors with distance weighting (valid imbalance handling)"
        if 'SMOTE' in code:
            return True, "SMOTE resampling detected for KNeighbors"
        return False, "KNeighbors requires weights='distance' for imbalance handling (or use SMOTE with imblearn Pipeline)"
    
    # V2.5.1: GradientBoosting doesn't support class_weight - accept alternatives
    if 'GradientBoostingClassifier' in code or 'HistGradientBoostingClassifier' in code:
        # These are valid for GradientBoosting
        gb_patterns = [
            r"subsample\s*=\s*0\.[5-9]",  # subsample=0.5 to 0.9
            r"subsample\s*=\s*0\.8",
            r"sample_weight",
            r"SMOTE",
            r"RandomOverSampler",
            r"class_weight",  # HistGradientBoosting supports it
        ]
        for pattern in gb_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True, "GradientBoosting with valid imbalance handling (subsample/sample_weight)"
        
        # GradientBoosting without explicit handling - give specific guidance
        return False, "GradientBoostingClassifier doesn't support class_weight. Use subsample=0.8 or HistGradientBoostingClassifier with class_weight='balanced'"
    
    # Standard class_weight patterns for other models
    patterns = [
        r"class_weight\s*=\s*['\"]balanced['\"]",
        r"class_weight\s*=\s*'balanced'",
        r'class_weight\s*=\s*"balanced"',
        r"SMOTE",
        r"RandomOverSampler",
        r"scale_pos_weight",
        r"compute_class_weight"
    ]
    
    for pattern in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return True, "Class imbalance handling detected"
    
    return False, "Missing class imbalance handling. Add class_weight='balanced' or use SMOTE."


def check_pipeline_usage(code: str) -> Tuple[bool, str]:
    """Check for sklearn Pipeline usage (prevents data leakage)."""
    if 'Pipeline' in code and ('sklearn.pipeline' in code or 'from sklearn' in code):
        return True, "Pipeline usage detected"
    
    if 'make_pipeline' in code:
        return True, "make_pipeline usage detected"
    
    # Also accept imblearn Pipeline
    if 'imblearn.pipeline' in code:
        return True, "imblearn Pipeline usage detected"
    
    return False, "Must use sklearn Pipeline to prevent data leakage during CV."


def check_scaler_for_linear(code: str, agent: str) -> Tuple[bool, str]:
    """Check that linear/kernel models have StandardScaler."""
    config = AGENT_MODEL_FAMILIES.get(agent)
    if not config or not config.get('requires_scaler'):
        return True, ""
    
    # Check if any model from this family is used
    uses_model = any(m in code for m in config['allowed'])
    
    if not uses_model:
        return True, ""
    
    # Must have StandardScaler (or MinMaxScaler)
    if 'StandardScaler' in code or 'MinMaxScaler' in code:
        return True, "Scaler found for model"
    
    return False, f"{config['family']} models REQUIRE StandardScaler in Pipeline."


def check_random_state(code: str) -> Tuple[bool, str]:
    """Check for random_state (reproducibility)."""
    if re.search(r'random_state\s*=\s*\d+', code):
        return True, "random_state set"
    
    return False, "Missing random_state for reproducibility. Add random_state=42."


def check_svc_probability(code: str, agent: str) -> Tuple[bool, str]:
    """Check that SVC has probability=True for AUC calculation."""
    if 'SVC' not in code or 'LinearSVC' in code:
        return True, ""
    
    if re.search(r'probability\s*=\s*True', code):
        return True, "SVC probability=True found"
    
    return False, "SVC REQUIRES probability=True for AUC calculation."


def check_pipeline_structure(code: str) -> Tuple[bool, str]:
    """
    Check Pipeline structure validity.
    All intermediate steps must be transformers, final step must be estimator.
    """
    try:
        tree = ast.parse(code)
        
        # Find Pipeline calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.name
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name == 'Pipeline' and node.args:
                    if isinstance(node.args[0], ast.List):
                        steps = node.args[0].elts
                        
                        # Check each step except the last
                        classifier_names = []
                        for cfg in AGENT_MODEL_FAMILIES.values():
                            classifier_names.extend(cfg['allowed'])
                        
                        for i, step in enumerate(steps[:-1]):
                            if isinstance(step, ast.Tuple) and len(step.elts) >= 2:
                                transformer = step.elts[1]
                                if isinstance(transformer, ast.Call):
                                    if isinstance(transformer.func, ast.Name):
                                        name = transformer.func.name
                                        if name in classifier_names:
                                            return False, f"Pipeline error: {name} in position {i+1}. Classifiers must be LAST step only."
        
        return True, "Pipeline structure valid"
        
    except Exception:
        return True, "Could not verify pipeline structure"


def check_column_transformer_numpy(code: str) -> Tuple[bool, str]:
    """
    V2.5: Check for ColumnTransformer with string column names on numpy arrays.
    """
    if 'ColumnTransformer' in code:
        # Check for string column specifications
        if re.search(r"ColumnTransformer\s*\([^)]*\[['\"][a-zA-Z]", code):
            return False, "ColumnTransformer with string column names requires DataFrame. Data is numpy array. Use integer indices or remove ColumnTransformer."
    
    return True, "No problematic ColumnTransformer"


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_agent_code(code: str, agent: str) -> Dict:
    """
    Run all validation checks on agent code.
    
    V2.5: Added AST checks for GPT failure modes.
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    # Critical checks (must pass)
    critical_checks = [
        ('syntax', lambda: check_syntax(code)),
        ('dangerous_patterns', lambda: check_dangerous_patterns(code)),
        ('get_model_function', lambda: check_get_model_function(code)),
        ('get_model_parameterless', lambda: check_get_model_parameterless(code)),  # V2.5
        ('lambda_y_dependency', lambda: check_lambda_y_dependency(code)),  # V2.5
        ('function_transformer_y', lambda: check_function_transformer_with_y(code)),  # V2.5
        ('column_transformer_numpy', lambda: check_column_transformer_numpy(code)),  # V2.5
        ('model_family', lambda: check_model_family(code, agent)),
        ('class_imbalance', lambda: check_class_imbalance_handling(code, agent)),
        ('pipeline', lambda: check_pipeline_usage(code)),
        ('scaler_for_linear', lambda: check_scaler_for_linear(code, agent)),
        ('svc_probability', lambda: check_svc_probability(code, agent)),
        ('pipeline_structure', lambda: check_pipeline_structure(code)),
    ]
    
    # Warning checks (should pass but not blocking)
    warning_checks = [
        ('random_state', lambda: check_random_state(code)),
    ]
    
    # Run critical checks
    for check_name, check_func in critical_checks:
        passed, message = check_func()
        result['checks'][check_name] = {'passed': passed, 'message': message}
        
        if not passed:
            result['valid'] = False
            result['errors'].append(f"[{check_name}] {message}")
    
    # Run warning checks
    for check_name, check_func in warning_checks:
        passed, message = check_func()
        result['checks'][check_name] = {'passed': passed, 'message': message}
        
        if not passed:
            result['warnings'].append(f"[{check_name}] {message}")
    
    return result


def generate_feedback_from_validation(validation_result: Dict, agent: str) -> str:
    """
    Generate targeted feedback from validation results.
    
    V2.5: Improved hints for GPT-specific issues.
    """
    if validation_result['valid']:
        return "Code passed all validation checks."
    
    lines = ["VALIDATION FAILED - Fix these issues:\n"]
    
    for error in validation_result['errors']:
        lines.append(f"❌ {error}")
    
    for warning in validation_result['warnings']:
        lines.append(f"⚠️ {warning}")
    
    # Add helpful hints based on common errors
    error_text = '\n'.join(validation_result['errors'])
    
    if 'lambda' in error_text.lower() or 'y' in error_text:
        lines.append("\n" + "="*60)
        lines.append("HINT: Do NOT use lambda functions that need 'y' argument!")
        lines.append("sklearn Pipeline does NOT pass y to intermediate steps.")
        lines.append("")
        lines.append("WRONG:")
        lines.append("  FunctionTransformer(lambda X, y: SMOTE().fit_resample(X, y))")
        lines.append("")
        lines.append("RIGHT (if you need SMOTE):")
        lines.append("  from imblearn.pipeline import Pipeline")
        lines.append("  Pipeline([('smote', SMOTE()), ('clf', YourClassifier())])")
        lines.append("")
        lines.append("OR just use class_weight='balanced' (simpler!):")
        lines.append("  RandomForestClassifier(class_weight='balanced')")
        lines.append("="*60)
    
    if 'class_imbalance' in error_text:
        if 'KNeighbors' in error_text:
            lines.append("\nHINT: For KNeighborsClassifier, use weights='distance':")
            lines.append("  KNeighborsClassifier(n_neighbors=5, weights='distance')")
        else:
            lines.append("\nHINT: Add class_weight='balanced' to your classifier")
    
    if 'scaler' in error_text.lower():
        lines.append("\nHINT: Wrap your model in a Pipeline with StandardScaler:")
        lines.append("  Pipeline([('scaler', StandardScaler()), ('clf', YourModel(...))])")
    
    if 'svc_probability' in error_text.lower():
        lines.append("\nHINT: SVC needs probability=True for AUC calculation:")
        lines.append("  SVC(kernel='rbf', probability=True, class_weight='balanced')")
    
    if 'pipeline_structure' in error_text.lower():
        lines.append("\nHINT: Pipeline must be: transformers → classifier (in that order)")
        lines.append("  Pipeline([('scaler', StandardScaler()), ('clf', Classifier())])")
    
    if 'ColumnTransformer' in error_text:
        lines.append("\nHINT: Data is numpy array, not DataFrame.")
        lines.append("  Remove ColumnTransformer or use integer column indices.")
    
    if 'parameterless' in error_text.lower():
        lines.append("\nHINT: get_model() must take NO arguments:")
        lines.append("  def get_model():  # NO parameters!")
        lines.append("      return Pipeline([...])")
    
    return '\n'.join(lines)


def get_validation_summary(validation_result: Dict) -> str:
    """Get a one-line summary of validation result."""
    if validation_result['valid']:
        return "✓ Valid"
    else:
        first_error = validation_result['errors'][0] if validation_result['errors'] else "Unknown error"
        return f"✗ {first_error}"
