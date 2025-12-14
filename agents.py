# agents.py
"""
Multi-Agent LLM Interface (V2.3)
=================================

Streamlined agent interface for explorer agents only.
Synthesis is handled separately in post_synthesis.py.

AGENTS:
- GPT-4o: Tree-based models
- Claude Sonnet 4.5: Linear models
- Gemini 2.0 Flash: Kernel/Distance models

Author: XCE Framework
Version: 2.3
"""

import os
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

# API clients (lazy import)
_openai_client = None
_anthropic_client = None
_genai_configured = False


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI()
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    return _anthropic_client


def _configure_genai():
    global _genai_configured
    if not _genai_configured:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _genai_configured = True


# Agent configurations
AGENT_CONFIGS = {
    'gpt': {
        'name': 'GPT-4o',
        'model_id': 'gpt-4o',
        'family': 'tree-based',
        'allowed_models': [
            'RandomForestClassifier',
            'GradientBoostingClassifier',
            'HistGradientBoostingClassifier',
            'ExtraTreesClassifier',
            'AdaBoostClassifier'
        ]
    },
    'claude': {
        'name': 'Claude Sonnet 4.5',
        'model_id': 'claude-sonnet-4-5-20250929',
        'family': 'linear',
        'allowed_models': [
            'LogisticRegression',
            'RidgeClassifier',
            'SGDClassifier',
            'LinearSVC'
        ]
    },
    'gemini': {
        'name': 'Gemini 2.0 Flash',
        'model_id': 'gemini-2.0-flash',
        'family': 'kernel/distance',
        'allowed_models': [
            'SVC',
            'NuSVC',
            'KNeighborsClassifier',
            'GaussianProcessClassifier'
        ]
    }
}

# Interaction log
INTERACTION_LOG = []


def log_interaction(agent, prompt, response, round_num=None):
    """Log interaction for reproducibility."""
    INTERACTION_LOG.append({
        'timestamp': datetime.now().isoformat(),
        'agent': agent,
        'round': round_num,
        'prompt_length': len(prompt),
        'response_length': len(response)
    })


def save_interaction_log(filepath='results/interaction_log.json'):
    """Save interaction log."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(INTERACTION_LOG, f, indent=2, cls=NumpyEncoder)


# =============================================================================
# API CALLS
# =============================================================================

def call_gpt(prompt, temperature=0.2):
    """Call OpenAI GPT-4o."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_claude(prompt, temperature=0.2):
    """Call Anthropic Claude Sonnet 4.5."""
    client = _get_anthropic_client()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_gemini(prompt, temperature=0.2):
    """Call Google Gemini 2.0 Flash."""
    _configure_genai()
    import google.generativeai as genai
    model = genai.GenerativeModel('gemini-2.0-flash')
    config = genai.GenerationConfig(temperature=temperature)
    response = model.generate_content(prompt, generation_config=config)
    return response.text


def call_agent(agent, prompt, temperature=0.2, round_num=None):
    """Call specified agent."""
    if agent == 'gpt':
        response = call_gpt(prompt, temperature)
    elif agent == 'claude':
        response = call_claude(prompt, temperature)
    elif agent == 'gemini':
        response = call_gemini(prompt, temperature)
    else:
        raise ValueError(f"Unknown agent: {agent}")
    
    log_interaction(agent, prompt, response, round_num)
    return response


# =============================================================================
# EXPLORER PROMPTS
# =============================================================================

def generate_explorer_prompt(agent, researcher_notes, baseline_results,
                            feedback=None, previous_code=None, round_num=1):
    """Generate prompt for explorer agent."""
    config = AGENT_CONFIGS[agent]
    
    # Format baseline results
    baseline_section = f"""
BASELINE VALIDATION RESULTS (Your model must beat this):
  Ambient-only baseline: {baseline_results['ambient']['composite']:.3f} composite
  Pressure-only model:   {baseline_results['pressure']['composite']:.3f} composite  
  Improvement needed:    {baseline_results['comparison_pressure_vs_ambient']['delta']:.3f} (+{baseline_results['comparison_pressure_vs_ambient']['delta_percent']:.1f}%)
  
Your target: Achieve composite > {baseline_results['pressure']['composite']:.3f}
"""
    
    # Feedback section
    feedback_section = ""
    if feedback and previous_code:
        feedback_section = f"""
═══════════════════════════════════════════════════════════════════════════════
FEEDBACK FROM ROUND {round_num - 1}
═══════════════════════════════════════════════════════════════════════════════

Your previous code:
```python
{previous_code[:1500]}{'...' if len(previous_code) > 1500 else ''}
```

Feedback:
{feedback}

FIX THE ISSUES ABOVE before proceeding.
═══════════════════════════════════════════════════════════════════════════════
"""
    
    # Model-family specific instructions
    model_specific = ""
    if agent == 'claude':
        model_specific = """
CRITICAL FOR LINEAR MODELS:
- MUST use StandardScaler in Pipeline (linear models fail without scaling!)
- Test regularization: C in [0.01, 0.1, 1.0, 10.0]
- Use max_iter=2000 to ensure convergence
- EXAMPLE:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  
  def get_model():
      return Pipeline([
          ('scaler', StandardScaler()),
          ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
      ])
  ```
"""
    elif agent == 'gpt':
        model_specific = """
YOUR TASK: Create a RandomForestClassifier model.

YOU MUST follow this EXACT pattern - only change the hyperparameter VALUES:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def get_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,      # Try: 50, 100, 150, 200
            max_depth=10,          # Try: 5, 10, 15, 20, None
            min_samples_split=2,   # Try: 2, 5, 10
            min_samples_leaf=1,    # Try: 1, 2, 4
            class_weight='balanced',  # REQUIRED - do not change
            random_state=42           # REQUIRED - do not change
        ))
    ])
```

RULES:
1. ONLY use RandomForestClassifier (no other models)
2. ONLY change the numeric hyperparameter values listed above
3. ALWAYS keep class_weight='balanced' and random_state=42
4. ALWAYS use the exact Pipeline structure shown
5. DO NOT add any other preprocessing steps
6. DO NOT use lambda functions, SMOTE, or ColumnTransformer

Your goal: Find the best hyperparameter combination for this dataset.
"""
    elif agent == 'gemini':
        model_specific = """
CRITICAL FOR KERNEL/DISTANCE MODELS:
- SVC: MUST use probability=True for AUC calculation, use class_weight='balanced'
- KNeighbors: Use weights='distance' for imbalance handling (NOT class_weight!)
- MUST use StandardScaler (kernel methods sensitive to scale!)

EXAMPLE (SVC):
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import SVC
  
  def get_model():
      return Pipeline([
          ('scaler', StandardScaler()),
          ('clf', SVC(
              kernel='rbf',
              C=1.0,
              probability=True,  # REQUIRED for AUC!
              class_weight='balanced',
              random_state=42
          ))
      ])
  ```

EXAMPLE (KNeighbors):
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import KNeighborsClassifier
  
  def get_model():
      return Pipeline([
          ('scaler', StandardScaler()),
          ('clf', KNeighborsClassifier(
              n_neighbors=7,
              weights='distance',  # REQUIRED for imbalance!
              metric='euclidean'
          ))
      ])
  ```
"""
    
    prompt = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EXPLORER AGENT: {config['name']:<20} - {config['family']:<20}         ║
╚══════════════════════════════════════════════════════════════════════════════╝

TASK: Create a classifier for binary alloy mixing enthalpy prediction.
      Predict: Will mixing be favorable (ΔH < 0) or unfavorable (ΔH ≥ 0)?

{baseline_section}

═══════════════════════════════════════════════════════════════════════════════
CONSTRAINTS (MUST FOLLOW)
═══════════════════════════════════════════════════════════════════════════════

1. MODEL FAMILY: {config['family']}
   ALLOWED MODELS: {', '.join(config['allowed_models'])}
   
2. CLASS IMBALANCE: Dataset is 60/40 split
   REQUIRED: Use class_weight='balanced' or SMOTE
   
3. DATA LEAKAGE PREVENTION:
   REQUIRED: Use sklearn Pipeline with preprocessing inside
   
4. REPRODUCIBILITY:
   REQUIRED: Set random_state=42

{model_specific}

═══════════════════════════════════════════════════════════════════════════════
RESEARCHER NOTES
═══════════════════════════════════════════════════════════════════════════════

{researcher_notes[:2000] if researcher_notes else "No additional notes."}

{feedback_section}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return ONLY a Python code block with this structure:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# ... imports ...

def get_model():
    \"\"\"
    Model: [Your model name]
    Rationale: [1-2 sentences on your design choices]
    \"\"\"
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', YourModel(...))
    ])
```

ROUND {round_num} - BEGIN YOUR RESPONSE:
"""
    
    return prompt


# =============================================================================
# UTILITIES
# =============================================================================

def check_api_keys():
    """Check API key status."""
    keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Claude': os.getenv('CLAUDE_API_KEY'),
        'Gemini': os.getenv('GEMINI_API_KEY'),
        'Grok': os.getenv('GROK_API_KEY')
    }
    
    print("API Key Status:")
    for name, key in keys.items():
        status = "✓ SET" if key else "✗ NOT SET"
        print(f"  {name:<10}: {status}")
    
    return all(keys.values())


def extract_code(response):
    """Extract Python code from response."""
    import re
    
    # Try python code block
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic code block
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


if __name__ == '__main__':
    check_api_keys()
