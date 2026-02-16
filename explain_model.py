import joblib
import shap
import pandas as pd

model = joblib.load("salary_model.pkl")

# Random example input
sample = pd.DataFrame([{
    "experience_years": 5,
    "education_level": 1,
    "num_skills": 7,
    "location_index": 2,
    "current_salary_lpa": 10
}])

# Extract RF model from pipeline
rf_model = model.named_steps["model"]

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(sample)

print("\nFeature Contributions:")

for i, col in enumerate(sample.columns):
    print(col, ":", shap_values[0][i])
