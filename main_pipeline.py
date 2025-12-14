# main_pipeline.py
"""
XCE Main Pipeline (V2.6)
=========================

Orchestrates Phases 1-4:
  Phase 1: Data Preparation (Feature Engineering)
  Phase 2: Baseline Validation (HARD GATE)
  Phase 3: Multi-Agent Model Exploration
  Phase 4: Explainability & Validation

V2.6 Updates:
- Auto-generates portable predictor after successful run
- Directional SHAP analysis (shows +/− effect of features)
- Enhanced output with direction information

Phase 5 (Synthesis) is decoupled - run post_synthesis.py separately.

Author: XCE Framework
Version: 2.6
"""

import os
import sys
import json
import numpy as np
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


def print_banner():
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██╗  ██╗ ██████╗███████╗    ██╗   ██╗██████╗     ██████╗                 ║
║     ╚██╗██╔╝██╔════╝██╔════╝    ██║   ██║╚════██╗   ██╔════╝                 ║
║      ╚███╔╝ ██║     █████╗      ██║   ██║ █████╔╝   ███████╗                 ║
║      ██╔██╗ ██║     ██╔══╝      ╚██╗ ██╔╝██╔═══╝    ██╔═══██╗                ║
║     ██╔╝ ██╗╚██████╗███████╗     ╚████╔╝ ███████╗██╗╚██████╔╝                ║
║     ╚═╝  ╚═╝ ╚═════╝╚══════╝      ╚═══╝  ╚══════╝╚═╝ ╚═════╝                 ║
║                                                                              ║
║              XAI-Curious Evolutionary Framework V2.6                         ║
║         Streamlined Scientific Workflow for Alloy Prediction                 ║
║                                                                              ║
║  NEW: Directional SHAP + Auto Portable Predictor Generation                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def run_pipeline():
    """
    Run the main pipeline (Phases 1-4).
    
    Phase 5 (Synthesis) must be run separately via post_synthesis.py
    after human review of results.
    """
    print_banner()
    
    start_time = datetime.now()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import modules
    from feature_engineering import run_feature_engineering
    from baseline_validation import run_baseline_validation
    from agents import (call_agent, generate_explorer_prompt, extract_code,
                       check_api_keys, save_interaction_log, AGENT_CONFIGS)
    from deterministic_validator import validate_agent_code, generate_feedback_from_validation
    from verifier import (EvolutionTracker, verify_and_evaluate, 
                         generate_feedback, save_round_results)
    from xai_utils import run_shap_analysis
    from ablation import run_ablation_experiments
    from cross_validator import cross_validate_importance
    from portable_generator import generate_portable_predictor  # V2.6
    
    os.makedirs('results', exist_ok=True)
    
    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION")
    print("="*70)
    
    # Generate all three feature sets
    X_ambient, y, groups, fn_ambient, meta_ambient = run_feature_engineering(mode='ambient')
    X_pressure, _, _, fn_pressure, meta_pressure = run_feature_engineering(mode='pressure')
    X_full, _, _, fn_full, meta_full = run_feature_engineering(mode='full')
    
    # =========================================================================
    # PHASE 2: BASELINE VALIDATION (HARD GATE)
    # =========================================================================
    baseline_results = run_baseline_validation(X_ambient, X_pressure, X_full, y, groups)
    
    # Check decision gate
    if not baseline_results['decision']['proceed_with_pipeline']:
        print("\n" + "!"*70)
        print("PIPELINE STOPPED: Hypothesis not supported by baseline validation")
        print("!"*70)
        print("\nReview results/baseline_results.json for details.")
        print("Options:")
        print("  1. Investigate feature engineering")
        print("  2. Consider alternative hypotheses")
        print("  3. Prepare negative results paper")
        
        # Save what we have
        with open('results/pipeline_status.json', 'w') as f:
            json.dump({
                'status': 'stopped',
                'reason': 'baseline_validation_failed',
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, cls=NumpyEncoder)
        
        return {'status': 'stopped', 'phase': 2, 'baseline': baseline_results}
    
    # =========================================================================
    # PHASE 3: MULTI-AGENT MODEL EXPLORATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: MULTI-AGENT MODEL EXPLORATION")
    print("="*70)
    
    # Check API keys
    print("\nChecking API keys...")
    check_api_keys()
    
    # Load researcher notes
    researcher_notes = ""
    if os.path.exists('researcher_notes.txt'):
        with open('researcher_notes.txt') as f:
            researcher_notes = f.read()
        print(f"✓ Loaded researcher notes")
    
    # Initialize tracker
    tracker = EvolutionTracker(
        max_rounds=8,
        patience=3,
        min_improvement=0.01,
        ceiling_score=2.4
    )
    
    # Explorer agents
    explorer_agents = ['gpt', 'claude', 'gemini']
    agent_results = {agent: None for agent in explorer_agents}
    agent_feedback = {agent: None for agent in explorer_agents}
    agent_code = {agent: None for agent in explorer_agents}
    
    round_num = 1
    
    while tracker.should_continue():
        print(f"\n{'─'*70}")
        print(f"ROUND {round_num}")
        print(f"{'─'*70}")
        
        round_results = {}
        
        for agent in explorer_agents:
            print(f"\n  [{agent.upper()}] {AGENT_CONFIGS[agent]['name']}...")
            
            # Generate prompt
            prompt = generate_explorer_prompt(
                agent=agent,
                researcher_notes=researcher_notes,
                baseline_results=baseline_results,
                feedback=agent_feedback.get(agent),
                previous_code=agent_code.get(agent),
                round_num=round_num
            )
            
            try:
                # Get response
                response = call_agent(agent, prompt, temperature=0.2, round_num=round_num)
                
                # Extract code
                code = extract_code(response)
                if not code:
                    round_results[agent] = {
                        'success': False,
                        'error': 'No code block found',
                        'agent': agent
                    }
                    print(f"      ✗ No code block found")
                    continue
                
                # Deterministic validation
                validation = validate_agent_code(code, agent)
                
                if not validation['valid']:
                    round_results[agent] = {
                        'success': False,
                        'error': '; '.join(validation['errors']),
                        'agent': agent,
                        'code': code
                    }
                    agent_feedback[agent] = generate_feedback_from_validation(validation, agent)
                    print(f"      ✗ Validation failed: {validation['errors'][0][:50]}...")
                    continue
                
                # Execute and evaluate
                result = verify_and_evaluate(code, agent, X_full, y, groups)
                round_results[agent] = result
                
                if result['success']:
                    m = result['metrics']
                    print(f"      ✓ {result['detected_model']}: composite={m['composite']:.3f}")
                    agent_code[agent] = code
                    agent_feedback[agent] = generate_feedback(result, validation)
                else:
                    print(f"      ✗ Execution failed: {result['error'][:50]}...")
                    agent_feedback[agent] = generate_feedback(result, validation)
                
            except Exception as e:
                round_results[agent] = {
                    'success': False,
                    'error': str(e),
                    'agent': agent
                }
                print(f"      ✗ Error: {str(e)[:50]}...")
        
        # Record round
        tracker.record(round_num, round_results)
        
        # Update best results
        for agent, result in round_results.items():
            if result.get('success'):
                current_best = agent_results[agent]
                if current_best is None or \
                   result['metrics']['composite'] > current_best.get('metrics', {}).get('composite', 0):
                    agent_results[agent] = result
        
        # Save round
        save_round_results(round_results, round_num)
        
        round_num += 1
    
    # Evolution complete
    print(f"\n{'─'*70}")
    print("EVOLUTION COMPLETE")
    print(f"{'─'*70}")
    print(f"  Best agent: {tracker.best_agent}")
    print(f"  Best composite: {tracker.best_score:.3f}")
    print(f"  Total rounds: {len(tracker.history)}")
    
    # Save agent results
    save_agent_results = {}
    for agent, result in agent_results.items():
        if result:
            save_agent_results[agent] = {
                'success': result['success'],
                'detected_model': result.get('detected_model'),
                'metrics': result.get('metrics'),
                'code': result.get('code')
            }
        else:
            save_agent_results[agent] = {'success': False}
    
    with open('results/agent_results.json', 'w') as f:
        json.dump(save_agent_results, f, indent=2, cls=NumpyEncoder)
    
    # Save evolution summary
    with open('results/evolution_summary.json', 'w') as f:
        json.dump({
            'best_agent': tracker.best_agent,
            'best_score': float(tracker.best_score) if tracker.best_score != -np.inf else None,
            'best_round': tracker.best_round,
            'total_rounds': len(tracker.history),
            'history': tracker.history,
            'summary': tracker.get_summary()
        }, f, indent=2, cls=NumpyEncoder)
    
    # Save best model code
    best_result = agent_results.get(tracker.best_agent)
    if best_result and best_result.get('code'):
        with open('best_model_code.py', 'w') as f:
            f.write(best_result['code'])
        print(f"  ✓ Saved: best_model_code.py")
    
    # Save interaction log
    save_interaction_log()
    
    # =========================================================================
    # PHASE 4: EXPLAINABILITY & VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 4: EXPLAINABILITY & VALIDATION")
    print("="*70)
    
    # V2.5: Get the FITTED model from tracker (not the factory function)
    fitted_model = tracker.get_best_fitted_model()
    
    if fitted_model is None and best_result and best_result.get('fitted_model'):
        fitted_model = best_result['fitted_model']
    
    # If still no fitted model, fit on full data (V2.5 fallback)
    if fitted_model is None and best_result and best_result.get('model'):
        print("\n  Fitting model on full data for SHAP analysis...")
        from verifier import fit_model_on_full_data
        fitted_model, fit_error = fit_model_on_full_data(best_result['model'], X_full, y)
        if fit_error:
            print(f"  ⚠ Could not fit model: {fit_error}")
    
    # SHAP Analysis (V2.6: now includes direction)
    shap_results = None
    if fitted_model is not None:
        with open('feature_metadata_full.json') as f:
            feature_metadata = json.load(f)
        
        print("\n  Running SHAP analysis with fitted model (V2.6: Directional)...")
        shap_results = run_shap_analysis(
            model=fitted_model,
            X=X_full,
            feature_names=fn_full,
            feature_metadata=feature_metadata,
            output_dir='results',
            y=y,
            groups=groups
        )
    else:
        print("\n  ⚠ No fitted model available for SHAP analysis")
    
    # Ablation Experiments (dual strategy: RF baseline + winning model)
    with open('feature_metadata_full.json') as f:
        feature_metadata = json.load(f)
    
    # V2.5: Use fitted model for model-specific ablation
    ablation_results = run_ablation_experiments(
        X_full, y, groups, fn_full, feature_metadata,
        shap_file='results/shap_importance.csv',
        output_dir='results',
        winning_model=fitted_model
    )
    
    # Cross-Validation (SHAP vs Ablation)
    validation_results = cross_validate_importance(
        shap_file='results/shap_importance.csv',
        ablation_file='results/ablation_results.json',
        output_dir='results'
    )
    
    # V2.5: Baseline comparison
    rf_baseline_score = baseline_results['full']['composite']
    evolved_best_score = tracker.best_score if tracker.best_score != -np.inf else 0
    baseline_comparison = {
        'rf_baseline': rf_baseline_score,
        'evolved_best': evolved_best_score,
        'difference': evolved_best_score - rf_baseline_score,
        'evolved_beats_baseline': evolved_best_score > rf_baseline_score
    }
    
    print(f"\n  ── Baseline Comparison ──")
    print(f"  RF Baseline (Full):    {rf_baseline_score:.3f}")
    print(f"  Best Evolved Model:    {evolved_best_score:.3f}")
    diff = baseline_comparison['difference']
    if diff > 0:
        print(f"  Evolved vs Baseline:   +{diff:.3f} (BETTER)")
    else:
        print(f"  Evolved vs Baseline:   {diff:.3f} (WORSE)")
    
    # =========================================================================
    # V2.6: GENERATE PORTABLE PREDICTOR
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING PORTABLE PREDICTOR (V2.6)")
    print("="*70)
    
    portable_result = None
    if fitted_model is not None:
        # Get metrics from best result
        best_metrics = None
        if best_result and best_result.get('metrics'):
            best_metrics = best_result['metrics']
        
        # Get model name
        model_name = None
        if best_result and best_result.get('detected_model'):
            model_name = best_result['detected_model']
        
        portable_result = generate_portable_predictor(
            fitted_model=fitted_model,
            X=X_full,
            y=y,
            output_dir='results',
            model_name=model_name,
            metrics=best_metrics
        )
    else:
        print("  ⚠ No fitted model available for portable predictor generation")
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save pipeline status (V2.6: include portable predictor info)
    pipeline_status = {
        'status': 'completed',
        'version': '2.6',
        'phases_completed': [1, 2, 3, 4],
        'phase_5_pending': True,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'best_agent': tracker.best_agent,
        'best_composite': float(tracker.best_score) if tracker.best_score != -np.inf else None,
        'hypothesis_supported': bool(baseline_results['decision']['hypothesis_supported']),
        'validation_passed': bool(validation_results.get('validated', False)),
        'baseline_comparison': baseline_comparison,
        'shap_succeeded': shap_results is not None and 'error' not in shap_results,
        'shap_directional': shap_results is not None and shap_results.get('summary', {}).get('direction_available', False),
        'portable_predictor_generated': portable_result is not None and portable_result.get('success', False)
    }
    
    with open('results/pipeline_status.json', 'w') as f:
        json.dump(pipeline_status, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE (Phases 1-4)")
    print("="*70)
    print(f"\n  Duration: {duration:.1f} seconds")
    print(f"  Best model: {tracker.best_agent} ({tracker.best_score:.3f})")
    print(f"  Hypothesis: {'SUPPORTED' if baseline_results['decision']['hypothesis_supported'] else 'NOT SUPPORTED'}")
    print(f"  Validation: {'PASSED' if validation_results.get('validated') else 'CONCERNS FOUND'}")
    
    print(f"\n  Output files in results/:")
    print(f"    - baseline_results.json")
    print(f"    - agent_results.json")
    print(f"    - evolution_summary.json")
    print(f"    - shap_importance.csv          (V2.6: includes direction)")
    print(f"    - shap_summary.json            (V2.6: includes direction)")
    print(f"    - shap_direction.png           (V2.6: new plot)")
    print(f"    - ablation_results.json")
    print(f"    - validation_flags.json")
    print(f"    - pipeline_status.json")
    if portable_result and portable_result.get('success'):
        print(f"    - predict_portable.py          (V2.6: auto-generated)")
        print(f"    - portable_metadata.json       (V2.6: predictor info)")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         NEXT STEP: HUMAN CHECKPOINT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Review the results in results/ directory, then run:                         ║
║                                                                              ║
║    python post_synthesis.py [human_notes.txt]                                ║
║                                                                              ║
║  Optionally create human_notes.txt with your interpretations.                ║
║                                                                              ║
║  V2.6: Portable predictor available at results/predict_portable.py           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    return {
        'status': 'completed',
        'baseline': baseline_results,
        'agent_results': save_agent_results,
        'tracker': tracker,
        'validation': validation_results,
        'portable_predictor': portable_result
    }


if __name__ == '__main__':
    run_pipeline()
