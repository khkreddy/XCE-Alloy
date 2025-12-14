# XCE Framework

### 1. Directional SHAP Analysis
SHAP output now includes **direction** information:
- `importance`: Magnitude of feature influence (always positive)
- `direction`: Signed effect (+/−) toward favorable mixing
- `effect`: Simple indicator ("+" or "−")

**Interpretation:**
- **Positive direction (+)**: Higher feature value → MORE likely favorable mixing
- **Negative direction (−)**: Higher feature value → LESS likely favorable mixing

### 2. Auto-Generated Portable Predictor
After successful pipeline completion, automatically generates `predict_portable.py`:
- Fully self-contained (no external dependencies except numpy, scipy, scikit-learn)
- Embedded trained model (base64 serialized)
- Command-line interface for predictions
- Works on any computer

## Output Files

| File | Description |
|------|-------------|
| `shap_importance.csv` | Now includes `direction` and `effect` columns |
| `shap_summary.json` | Now includes `top_favorable_features` and `top_unfavorable_features` |
| `shap_direction.png` | New plot showing directional effects |
| `predict_portable.py` | Auto-generated standalone predictor |
| `portable_metadata.json` | Metadata about the portable predictor |

## Example SHAP Output

```csv
feature,importance,direction,effect,category,pressure_dependent
EN_A_std,0.0222,+0.0187,+,Electronegativity,True
SP_delta_initial,0.0151,-0.0098,−,Spin_State,False
RE_A_std,0.0188,+0.0145,+,Relative_EN,True
```

## Usage

### Run Pipeline
```bash
python main_pipeline.py
```

### Use Portable Predictor
```bash
# After pipeline completes, in results/:
python predict_portable.py Fe La
python predict_portable.py --all --folder ./sharc_data/
python predict_portable.py --list --folder ./sharc_data/
```

## File Structure

```
xce_v2.6/
├── main_pipeline.py           # Main orchestration (UPDATED)
├── feature_engineering.py     # Feature computation
├── baseline_validation.py     # Hypothesis testing
├── agents.py                  # LLM agent interface
├── deterministic_validator.py # Code validation
├── verifier.py                # Model evaluation
├── xai_utils.py               # SHAP analysis (UPDATED - directional)
├── ablation.py                # Ablation experiments
├── cross_validator.py         # SHAP-ablation validation
├── portable_generator.py      # NEW - generates portable predictor
└── post_synthesis.py          # Human-in-loop synthesis
```

## Requirements
```
numpy
pandas
scikit-learn
scipy
shap
matplotlib
openai
anthropic
google-generativeai
```


