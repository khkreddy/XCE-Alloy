# post_synthesis.py
"""
Decoupled Synthesis Report Generation (V2.3)
=============================================

Runs AFTER main pipeline, with explicit human checkpoint.
Uses fixed JSON inputs for reproducibility.

WORKFLOW:
1. Load saved results (baseline, agent, SHAP, ablation, validation)
2. [HUMAN CHECKPOINT] Review and optionally add notes
3. Generate synthesis with Grok
4. Save report

Author: XCE Framework
Version: 2.3
"""

import os
import json
from datetime import datetime

# Grok API
def call_grok(prompt, temperature=0.3):
    """Call xAI Grok-4 API."""
    import openai
    client = openai.OpenAI(
        api_key=os.getenv("GROK_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
        model="grok-4",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def load_results(results_dir='results'):
    """Load all saved results."""
    results = {}
    
    files = {
        'baseline': 'baseline_results.json',
        'agent': 'agent_results.json',
        'shap': 'shap_summary.json',
        'ablation': 'ablation_results.json',
        'validation': 'validation_flags.json',
        'evolution': 'evolution_summary.json'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                results[key] = json.load(f)
            print(f"  ✓ Loaded: {filename}")
        else:
            print(f"  - Missing: {filename}")
            results[key] = None
    
    return results


def format_baseline_section(baseline):
    """Format baseline results for prompt."""
    if not baseline:
        return "Baseline results not available."
    
    return f"""
BASELINE VALIDATION RESULTS (Hypothesis Test):
  Ambient-only:    {baseline['ambient']['composite']:.3f} ± {baseline['ambient'].get('composite_std', 0):.3f} ({baseline['ambient']['n_features']} features)
  Pressure-only:   {baseline['pressure']['composite']:.3f} ± {baseline['pressure'].get('composite_std', 0):.3f} ({baseline['pressure']['n_features']} features)
  Full model:      {baseline['full']['composite']:.3f} ± {baseline['full'].get('composite_std', 0):.3f} ({baseline['full']['n_features']} features)

  PRIMARY COMPARISON (Pressure vs Ambient):
    Δ Composite: {baseline['comparison_pressure_vs_ambient']['delta']:+.3f} ({baseline['comparison_pressure_vs_ambient']['delta_percent']:+.1f}%)
    p-value: {baseline['comparison_pressure_vs_ambient']['p_value']:.4f}
    Cohen's d: {baseline['comparison_pressure_vs_ambient']['cohens_d']:.2f}
    Significant: {baseline['comparison_pressure_vs_ambient']['significant']}

  DECISION: {baseline['decision']['recommendation']}
"""


def format_agent_section(agent_results):
    """Format agent results for prompt."""
    if not agent_results:
        return "Agent results not available."
    
    lines = ["MULTI-AGENT MODEL EXPLORATION RESULTS:"]
    
    for agent, data in agent_results.items():
        if data.get('success'):
            m = data['metrics']
            lines.append(f"""
  {agent.upper()} ({data.get('detected_model', 'Unknown')}):
    F1: {m['f1_mean']:.3f} ± {m['f1_std']:.3f}
    MCC: {m['mcc_mean']:.3f} ± {m['mcc_std']:.3f}
    AUC: {m['auc_mean']:.3f} ± {m['auc_std']:.3f}
    Composite: {m['composite']:.3f}
    95% CI: [{m['composite_ci_95'][0]:.3f}, {m['composite_ci_95'][1]:.3f}]""")
        else:
            lines.append(f"\n  {agent.upper()}: FAILED - {data.get('error', 'Unknown error')[:50]}")
    
    return '\n'.join(lines)


def format_shap_section(shap_summary):
    """Format SHAP summary for prompt."""
    if not shap_summary:
        return "SHAP results not available."
    
    lines = ["SHAP FEATURE IMPORTANCE:"]
    lines.append(f"\n  Pressure-dependent features: {shap_summary.get('pressure_dependent_pct', 0):.1f}%")
    lines.append(f"  Static features: {shap_summary.get('static_pct', 0):.1f}%")
    
    lines.append("\n  Top 10 Features:")
    for i, feat in enumerate(shap_summary.get('top_10_features', [])[:10]):
        lines.append(f"    {i+1}. {feat['feature']}: {feat['importance']:.4f} ({feat['category']})")
    
    lines.append("\n  Category Importance:")
    for cat, imp in sorted(shap_summary.get('category_importance', {}).items(), 
                          key=lambda x: x[1], reverse=True):
        lines.append(f"    {cat}: {imp:.4f}")
    
    return '\n'.join(lines)


def format_ablation_section(ablation):
    """Format ablation results for prompt."""
    if not ablation:
        return "Ablation results not available."
    
    lines = ["ABLATION EXPERIMENTS (Validated Feature Importance):"]
    lines.append(f"\n  Baseline composite: {ablation.get('baseline', {}).get('composite', 0):.3f}")
    
    # Individual ablation
    if 'individual' in ablation:
        lines.append("\n  Individual Feature Removal (Top 5 impact):")
        sorted_ind = sorted(ablation['individual'].items(), 
                          key=lambda x: abs(x[1]['drop_percent']), reverse=True)
        for feat, data in sorted_ind[:5]:
            lines.append(f"    Remove {feat[:30]:<30}: Δ = {data['drop_percent']:+.2f}%")
    
    # Category ablation
    if 'category' in ablation:
        lines.append("\n  Category Removal:")
        for cat, data in ablation['category'].items():
            lines.append(f"    Remove {cat:<20}: Δ = {data['drop_percent']:+.2f}%")
    
    # Pressure vs Static
    if 'pressure_vs_static' in ablation:
        pvs = ablation['pressure_vs_static']
        if 'remove_pressure' in pvs:
            lines.append(f"\n  Remove all pressure-dependent: Δ = {pvs['remove_pressure']['drop_percent']:+.2f}%")
        if 'remove_static' in pvs:
            lines.append(f"  Remove all static: Δ = {pvs['remove_static']['drop_percent']:+.2f}%")
    
    return '\n'.join(lines)


def format_validation_section(validation):
    """Format validation results for prompt."""
    if not validation:
        return "Validation results not available."
    
    lines = ["SHAP-ABLATION CROSS-VALIDATION:"]
    lines.append(f"\n  Status: {'VALIDATED' if validation.get('validated') else 'CONCERNS FOUND'}")
    lines.append(f"  Summary: {validation.get('summary', 'N/A')}")
    
    if validation.get('flags'):
        lines.append("\n  FLAGS (must address in synthesis):")
        for flag in validation['flags'][:5]:
            lines.append(f"    {flag}")
    
    if validation.get('warnings'):
        lines.append(f"\n  WARNINGS: {len(validation['warnings'])} items (review in details)")
    
    return '\n'.join(lines)


def generate_synthesis_prompt(results, researcher_notes=None, human_notes=None):
    """Generate the synthesis prompt."""
    
    baseline_section = format_baseline_section(results.get('baseline'))
    agent_section = format_agent_section(results.get('agent'))
    shap_section = format_shap_section(results.get('shap'))
    ablation_section = format_ablation_section(results.get('ablation'))
    validation_section = format_validation_section(results.get('validation'))
    
    human_notes_section = ""
    if human_notes:
        human_notes_section = f"""
═══════════════════════════════════════════════════════════════════════════════
PRINCIPAL INVESTIGATOR NOTES (Human Input)
═══════════════════════════════════════════════════════════════════════════════

{human_notes}
"""
    
    prompt = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SYNTHESIS REPORT GENERATION                                ║
║                    Binary Alloy Mixing Prediction                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

You are a senior materials scientist writing the Results and Discussion section
of a scientific paper on using pressure-dependent atomic properties to predict
binary alloy mixing enthalpy.

═══════════════════════════════════════════════════════════════════════════════
EXPERIMENTAL RESULTS (From Fixed JSON Inputs)
═══════════════════════════════════════════════════════════════════════════════

{baseline_section}

{agent_section}

{shap_section}

{ablation_section}

{validation_section}

{human_notes_section}

═══════════════════════════════════════════════════════════════════════════════
SYNTHESIS REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

1. NUMERICAL PRECISION
   - Use EXACT numbers from the data above
   - Do not round or approximate
   - Cite specific values when making claims

2. CONFIDENCE LABELING (Required for each major claim)
   - HIGH: Supported by both SHAP and ablation, validated, p < 0.01
   - MEDIUM: Supported by one method or minor validation flags
   - LOW: Unvalidated or significant discrepancies

3. VALIDATION FLAG HANDLING
   - If validation flags exist, explicitly address them
   - Do NOT make claims that contradict validation results

4. STRUCTURE YOUR REPORT AS:
   
   ## 1. Hypothesis Test Results
   - State the primary finding (pressure vs ambient)
   - Report statistical significance
   - Confidence: [HIGH/MEDIUM/LOW]
   
   ## 2. Model Performance Comparison
   - Which model family performed best?
   - What does this suggest about data structure?
   - Confidence: [HIGH/MEDIUM/LOW]
   
   ## 3. Feature Importance Analysis
   - Top contributing features (SHAP)
   - Ablation-validated importance
   - Flag any SHAP-ablation discrepancies
   - Pressure-dependent vs static contribution
   - Confidence: [HIGH/MEDIUM/LOW]
   
   ## 4. Physical Interpretation
   - What do results tell us about alloy mixing physics?
   - Connection to Miedema model or Hume-Rothery rules
   - Novel insights (if any)
   - Confidence: [HIGH/MEDIUM/LOW]
   
   ## 5. Limitations
   - Validation concerns
   - Data limitations
   - Generalization questions
   
   ## 6. Key Conclusions
   - 5-7 bullet points
   - Each labeled with confidence level

═══════════════════════════════════════════════════════════════════════════════

Generate the synthesis report now:
"""
    
    return prompt


def run_synthesis(results_dir='results', human_notes_file=None, output_file=None):
    """
    Run the synthesis generation.
    
    Args:
        results_dir: Directory containing result JSON files
        human_notes_file: Optional path to human notes file
        output_file: Output path for synthesis report
    """
    print("\n" + "="*70)
    print("PHASE 5: SYNTHESIS REPORT GENERATION")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)
    
    # Load researcher notes
    researcher_notes = None
    if os.path.exists('researcher_notes.txt'):
        with open('researcher_notes.txt') as f:
            researcher_notes = f.read()
        print(f"  ✓ Loaded: researcher_notes.txt")
    
    # Load human notes (optional)
    human_notes = None
    if human_notes_file and os.path.exists(human_notes_file):
        with open(human_notes_file) as f:
            human_notes = f.read()
        print(f"  ✓ Loaded: {human_notes_file}")
    
    # Check for validation concerns
    validation = results.get('validation', {})
    if validation and not validation.get('validated', True):
        print("\n" + "!"*70)
        print("WARNING: Validation flags detected!")
        print("Review results/validation_flags.json before proceeding.")
        print("!"*70)
        
        response = input("\nProceed with synthesis? (y/n): ")
        if response.lower() != 'y':
            print("Synthesis cancelled.")
            return None
    
    # Generate prompt
    print("\nGenerating synthesis prompt...")
    prompt = generate_synthesis_prompt(results, researcher_notes, human_notes)
    print(f"  Prompt length: {len(prompt)} characters")
    
    # Call Grok
    print("\nCalling Grok-4 synthesizer...")
    synthesis = call_grok(prompt, temperature=0.3)
    print(f"  Response length: {len(synthesis)} characters")
    
    # Save report
    if output_file is None:
        output_file = os.path.join(results_dir, 'synthesis_report.md')
    
    with open(output_file, 'w') as f:
        f.write(f"# XCE Synthesis Report (V2.3)\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        f.write(synthesis)
    
    print(f"\n✓ Saved: {output_file}")
    
    # Also save the prompt for reproducibility
    prompt_file = os.path.join(results_dir, 'synthesis_prompt.txt')
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"✓ Saved: {prompt_file}")
    
    return synthesis


if __name__ == '__main__':
    import sys
    
    human_notes_file = None
    if len(sys.argv) > 1:
        human_notes_file = sys.argv[1]
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         HUMAN CHECKPOINT                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Before running synthesis, please review:                                    ║
║                                                                              ║
║    □ results/baseline_results.json   - Hypothesis test                       ║
║    □ results/agent_results.json      - Model performance                     ║
║    □ results/shap_summary.json       - Feature importance                    ║
║    □ results/ablation_results.json   - Validated importance                  ║
║    □ results/validation_flags.json   - Cross-validation results              ║
║                                                                              ║
║  Optionally create: human_notes.txt with your interpretations                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    run_synthesis(human_notes_file=human_notes_file)
