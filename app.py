import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction")
st.caption("Machine Learning Classification Application")

st.info(
    "This model estimates attrition risk from the supplied employee attributes. "
    "It should support HR analysis, not replace human judgment or employment decisions."
)

@st.cache_resource
def load_model():
    return joblib.load("models/employee_attrition_model.joblib")

model = load_model()

c1, c2, c3 = st.columns(3)

with c1:
    Gender = st.selectbox("Gender", ["Female", "Male"])

    Age = st.selectbox(
        "Age",
        ["21 to 30", "31 to 40", "41 to 50", "51 to 60"]
    )

    Maritalstatus = st.selectbox(
        "Marital status",
        ["Married", "Single", "Divorced"]
    )

    Academic_degree = st.selectbox(
        "Academic degree",
        ["Master's", "Bachelor's", "Diploma or secondary", "Ph.D"]
    )

    Years_Experience = st.selectbox(
        "Years of experience",
        [
            "Less than 5 years",
            "From 5 to 10 years",
            "From 11 to 15 years",
            "From 16 to 20 years",
            "From 21 to 25 years",
            "From 26 to 30 years",
            "From 31 to 35 years"
        ]
    )

    Years_experience_lastorganization = st.selectbox(
        "Experience at last organization",
        [
            "Less than 5 years",
            "From 5 to 10 years",
            "From 11 to 15 years",
            "From 16 to 20 years",
            "From 21 to 25 years",
            "From 26 to 30 years",
            "From 31 to 35 years"
        ]
    )

    Sector = st.selectbox(
        "Sector",
        [
            "Medical sector",
            "Education sector",
            "Communications and IT sector",
            "Tourism sector",
            "Financial sector",
            "Economic sector",
            "Transport sector",
            "Food production sector",
            "Industry sector",
            "Media sector",
            "Engineering consulting companies",
            "Environment, Water, and Agriculture sector",
            "Law firm",
            "Energy sector",
            "Restaurant sector",
            "education sector"
        ]
    )

    Department = st.selectbox(
        "Department",
        [
            "Accounting",
            "Teaching",
            "Relations",
            "HR",
            "Administration",
            "Processes",
            "Banking operations",
            "Customers service",
            "Sales",
            "Safety & Security",
            "Technical support",
            "Training",
            "Information technology",
            "Patient Affairs",
            "Marketing",
            "Legal Affairs",
            "Treasury",
            "Production",
            "Medical service",
            "Studies and design",
            "patient Affairs",
            "Engineering",
            "technical support"
        ]
    )

with c2:
    JobTitle = st.text_input("Job title", "Accountant")

    MonthlySalary = st.selectbox(
        "Monthly salary",
        [
            "Less than 5000 SAR",
            "From 5000 to 10000 S.R",
            "From 11000 to 15000 S.R",
            "From 16000 to 20000 S.R",
            "From 21000 to 25000 S.R",
            "From 26000 to 30000 S.R",
            "S.R 31000 - and more"
        ]
    )

    Allowances = st.selectbox("Allowances", [0, 1, 2, 3, 4])

    MedicalInsurance = st.selectbox(
        "Medical insurance",
        ["No", "Yes"]
    )

    Bonus = st.selectbox(
        "Bonus",
        ["No", "Yes"]
    )

    OverTime = st.selectbox(
        "Overtime",
        ["No", "Yes"]
    )

    Payment_Overtime = st.selectbox(
        "Overtime payment",
        ["I don't have overtime", "No", "Yes"]
    )

    Rewards_Wages_Satisfaction = st.selectbox(
        "Rewards & wages satisfaction",
        ["Yes", "No"]
    )

    Get_Deserved_Promotion = st.selectbox(
        "Received deserved promotion",
        ["Yes", "No"]
    )

    Training_programs_During_last_three_years = st.selectbox(
        "Training in last 3 years",
        [
            "I did not receive any training",
            "From 1 to 3 training programs",
            "From 4 to 6 training programs",
            "From 7 training programs to more"
        ]
    )

    Useful_Training_Programs = st.selectbox(
        "Useful training programs",
        ["Yes", "No"]
    )

    Business_Travel = st.selectbox(
        "Business travel",
        [
            "I do not travel for work",
            "Travel rarely",
            "Travel frequently"
        ]
    )

with c3:
    Job_Support = st.selectbox(
        "Job support",
        ["Low", "Medium", "High"]
    )

    Recognition = st.selectbox(
        "Recognition",
        ["No", "Yes"]
    )

    Emotional_Commitment = st.selectbox(
        "Emotional commitment",
        ["Low", "Medium", "High"]
    )

    Job_Engagement = st.selectbox(
        "Job engagement",
        ["Difficult", "Medium", "Easy"]
    )

    Distance_to_work = st.selectbox(
        "Distance to work",
        ["Close", "Medium", "Far"]
    )

    Work_Live_Balance = st.selectbox(
        "Work-life balance",
        ["Difficult", "Medium", "Easy"]
    )

    Physical_Stress = st.selectbox(
        "Physical stress",
        ["No", "Sometimes", "Yes"]
    )

    Psychological_Exhaustion = st.selectbox(
        "Psychological exhaustion",
        ["No", "Sometimes", "Yes"]
    )

    Job_Stability = st.selectbox(
        "Job stability",
        ["No", "Yes"]
    )

    Health_Issues = st.selectbox(
        "Health issues",
        ["No", "Yes"]
    )

    Environment_Satisfaction = st.selectbox(
        "Environment satisfaction",
        ["Low", "Medium", "High"]
    )

    Job_Satisfaction = st.selectbox(
        "Job satisfaction",
        ["Not satisfied", "Satisfied", "Very satisfied"]
    )

    Job_Opportunities = st.selectbox(
        "Job opportunities",
        ["No", "Yes"]
    )

input_df = pd.DataFrame([{
    "Gender": Gender,
    "Age": Age,
    "Maritalstatus": Maritalstatus,
    "Academic_degree": Academic_degree,
    "Years_Experience": Years_Experience,
    "Years_experience_lastorganization": Years_experience_lastorganization,
    "Sector": Sector,
    "Department": Department,
    "JobTitle": JobTitle,
    "MonthlySalary": MonthlySalary,
    "Allowances": Allowances,
    "MedicalInsurance": MedicalInsurance,
    "Bonus": Bonus,
    "OverTime": OverTime,
    "Payment_Overtime": Payment_Overtime,
    "Rewards&Wages_Satisfaction": Rewards_Wages_Satisfaction,
    "Get_ Deserved_Promotion": Get_Deserved_Promotion,
    "Training_programs_ During_last_three_years": Training_programs_During_last_three_years,
    "Useful_Training_Programs": Useful_Training_Programs,
    "Business_Travel": Business_Travel,
    "Job_Support": Job_Support,
    "Recognition": Recognition,
    "Emotional_Commitment": Emotional_Commitment,
    "Job_Engagement": Job_Engagement,
    "Distance_to_work": Distance_to_work,
    "Work_Live_Balance": Work_Live_Balance,
    "Physical_Stress": Physical_Stress,
    "Psychological_Exhaustion": Psychological_Exhaustion,
    "Job_Stability": Job_Stability,
    "Health_Issues": Health_Issues,
    "Environment_Satisfaction": Environment_Satisfaction,
    "Job_Satisfaction": Job_Satisfaction,
    "Job_Opportunities": Job_Opportunities
}])

expected_columns = list(
    model.named_steps["preprocessor"].feature_names_in_
)

missing_columns = [
    column for column in expected_columns
    if column not in input_df.columns
]

if st.button(
    "🔮 Predict Attrition",
    type="primary",
    use_container_width=True
):

    if missing_columns:
        st.error("The following model features are missing:")
        st.code("\n".join(missing_columns))
        st.stop()

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Predicted outcome: Employee Attrition (Yes)")
        risk = "HIGH"
    else:
        st.success("✅ Predicted outcome: Employee Attrition (No)")
        risk = "LOW"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Attrition Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk
        )

    st.progress(float(probability))