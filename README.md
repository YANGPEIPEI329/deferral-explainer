# Deferral Explainer

Data product of the research project **"An Uncertainty Attribution Model for Explaining Model Abstention in Clinical
Risk Prediction"** (YANG PEIPEI, Master of Data Science, Universiti Malaya, 2026; supervisor Assoc. Prof. Dr. Maizatul
Akmar binti Ismail).

When a conformal selective prediction model abstains, its prediction set contains both labels and the case is handed to
a human. The Deferral Explainer shows **why** a case is deferred, next to what conventional SHAP says about the
predicted risk. Everything is computed live from the fitted models:

- the recalibrated LightGBM probability of the selected or edited case;
- the class-conditional split conformal prediction set at 95, 90 or 80 per cent target coverage;
- the ambiguity score A = -|logit p - logit pi| and the abstention propensity R over 200 calibration bootstraps;
- grouped KernelSHAP attributions of A (proposed) and of p (conventional), with the coalition sample and background
  used in the study, so the values reproduce the report;
- a what-if: how often the deferral is resolved when the top-k feature groups are replaced by background values.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Contents

- `app.py`: the application
- `assets/<dataset>/model.txt`: fitted LightGBM model
- `assets/<dataset>/data.npz`: calibration-set probabilities and labels, calibration bootstraps, background sample,
  coalition sample, and the 600 held-out cases explained in the study
- `assets/<dataset>/meta.json`: feature names and groups, Platt parameters, prevalence, summary results

## Data sources

- Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, CC BY 4.0); Strack et al. (2014).
- PhysioNet/Computing in Cardiology Challenge 2019, early prediction of sepsis; Reyna et al. (2020).
- Open University Learning Analytics Dataset (CC BY 4.0); Kuzilek et al. (2017).

Research prototype. It is not a medical device and must not be used for clinical decisions.
