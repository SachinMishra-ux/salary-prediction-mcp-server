from fastmcp import FastMCP
import joblib
import pandas as pd
import shap
import numpy as np
from openai import OpenAI

# Initialize MCP Server
mcp = FastMCP("Salary Intelligence Server")

# Load trained ML pipeline
pipeline = joblib.load("salary_model.pkl")

# Extract RF model for SHAP
rf_model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)

# LLM Client
client = OpenAI()

# ------------------------------
# TOOL 1: Predict Salary
# ------------------------------

@mcp.tool()
def predict_salary(
    experience_years: int,
    education_level: int,
    num_skills: int,
    location_index: int,
    current_salary_lpa: float
):
    """
    Predict expected market salary based on profile.
    """

    input_df = pd.DataFrame([{
        "experience_years": experience_years,
        "education_level": education_level,
        "num_skills": num_skills,
        "location_index": location_index,
        "current_salary_lpa": current_salary_lpa
    }])

    prediction = pipeline.predict(input_df)[0]

    return {
        "predicted_market_salary_lpa": round(float(prediction), 2)
    }

# ------------------------------
# TOOL 2: Explain Prediction
# ------------------------------

@mcp.tool()
def explain_salary_prediction(
    experience_years: int,
    education_level: int,
    num_skills: int,
    location_index: int,
    current_salary_lpa: float
):
    """
    Explain why salary prediction was made.
    """

    input_df = pd.DataFrame([{
        "experience_years": experience_years,
        "education_level": education_level,
        "num_skills": num_skills,
        "location_index": location_index,
        "current_salary_lpa": current_salary_lpa
    }])

    shap_values = explainer.shap_values(input_df)

    contributions = {}

    for i, col in enumerate(input_df.columns):
        contributions[col] = round(float(shap_values[0][i]), 2)

    return {
        "feature_contributions": contributions
    }

# ------------------------------
# TOOL 3: Salary Gap Analysis
# ------------------------------

@mcp.tool()
def salary_gap_analysis(
    experience_years: int,
    education_level: int,
    num_skills: int,
    location_index: int,
    current_salary_lpa: float
):
    """
    Analyze how underpaid or overpaid user is.
    """

    input_df = pd.DataFrame([{
        "experience_years": experience_years,
        "education_level": education_level,
        "num_skills": num_skills,
        "location_index": location_index,
        "current_salary_lpa": current_salary_lpa
    }])

    predicted_salary = pipeline.predict(input_df)[0]

    gap = predicted_salary - current_salary_lpa
    percent = (gap / predicted_salary) * 100

    return {
        "predicted_salary_lpa": round(float(predicted_salary), 2),
        "salary_gap_lpa": round(float(gap), 2),
        "underpaid_percentage": round(float(percent), 2)
    }

# ------------------------------
# TOOL 4: Skill Recommendation
# ------------------------------

# @mcp.tool()
# def recommend_skills_for_target_salary(
#     experience_years: int,
#     current_salary_lpa: float,
#     target_salary_lpa: float,
#     current_skills: list[str]
# ):
#     """
#     Recommend skills needed to reach target salary.
#     """

#     prompt = f"""
#     A candidate has:
#     Experience: {experience_years} years
#     Current Salary: {current_salary_lpa} LPA
#     Target Salary: {target_salary_lpa} LPA
#     Current Skills: {current_skills}

#     Suggest 3-5 skills they should learn to reach target salary.
#     """

#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return {
#         "recommended_skills": response.choices[0].message.content
#     }

# Run MCP Server
if __name__ == "__main__":
    mcp.run(transport="http", host= "0.0.0.0",port=8002)
