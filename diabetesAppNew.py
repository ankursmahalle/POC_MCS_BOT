import streamlit as st
import pandas as pd
import json
from langchain import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import time
from langchain.prompts import PromptTemplate
import os
from constants import openai_key
from langchain.chains import LLMChain  # Importing a chain to link language models
import seaborn as sns
import plotly.express as px

# API_KEY = "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"

st.set_page_config(page_title="Page Title", layout="wide")

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#ADD8E6,#ADD8E6);
    color: Red;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #ADD8E6;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style>
""",
    unsafe_allow_html=True,
)


st.sidebar.image("logo_dark.png", use_column_width=True)

left_column, right_column = st.columns(2)
left_column.title("👨‍💻 POC: Cohort Builder")

data = st.sidebar.file_uploader("Upload a CSV")

# Insert your query(To be serached)
query = left_column.text_area(
    "Ask about diabetic related topics like diet, Blood Sugar Level, Complications"
)

os.environ["OPENAI_API_KEY"] = openai_key

first_input_prompt = PromptTemplate(
    input_variables=["prompt"],
    template="reply this question {prompt} in the context of a diabetic patient.",
)


# Create an instance of the OpenAI language model (LLM)
llm = OpenAI(temperature=0.8)

# Create a language model chain with the specified prompt template and memory
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

# Button to Submit your query
if left_column.button("Submit Query", type="primary"):
    if query:
        left_column.write(chain.run(query))


def group_age(row):
    if row["Age"] > 20 and row["Age"] <= 30:
        return "21-30"
    elif row["Age"] > 30 and row["Age"] <= 40:
        return "31-40"
    elif row["Age"] > 40 and row["Age"] <= 50:
        return "41-50"
    elif row["Age"] > 50 and row["Age"] <= 60:
        return "51-60"
    elif row["Age"] > 60 and row["Age"] <= 70:
        return "61-70"
    elif row["Age"] > 70 and row["Age"] <= 80:
        return "71-80"
    elif row["Age"] > 80 and row["Age"] <= 90:
        return "81-90"
    return "90+"


def group_bp(row):
    if row["BloodPressure"] > 80 and row["BloodPressure"] <= 89:
        return "Prehypertension"
    elif row["BloodPressure"] > 89 and row["BloodPressure"] <= 99:
        return "Stage 1 Hypertension"
    elif row["BloodPressure"] > 99 and row["BloodPressure"] <= 150:
        return "Stage 2 Hypertension"
    return "Normal"


def group_bmi(row):
    if row["BMI"] > 18.5 and row["BMI"] <= 24.9:
        return "Normal"
    elif row["BMI"] > 24.9 and row["BMI"] <= 29.9:
        return "Overweight"
    elif row["BMI"] > 29.9 and row["BMI"] <= 34.9:
        return "High Obesity"
    elif row["BMI"] > 34.9 and row["BMI"] <= 39.9:
        return "Very High Obesity"
    return "Extreme Obesity"


if data is not None:
    with right_column:
        df = pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
        df["AgeGroup"] = df.apply(lambda row: group_age(row), axis=1)
        df["BPGroup"] = df.apply(lambda row: group_bp(row), axis=1)
        df["BMIGroup"] = df.apply(lambda row: group_bmi(row), axis=1)
        df.loc[df["Outcome"] == 0, "Outcome"] = 2
        df_diabetic = df[df.Outcome == 1]
        df_non_diabetic = df[df.Outcome == 2]

        totalCount = sns.countplot(x="Outcome", data=df)
        totalCount.set_xticklabels(["Non-Diabetic", "Diabetic"])
        totalCount.bar_label(totalCount.containers[0])
        right_column.pyplot(totalCount.figure)

        # Radio buttons
        radio_btn = right_column.radio(
            "Choose any one to analyse",
            ["***Diabetic***", "***Non-Diabetic***"],
            captions=[
                "To Check diabetic data",
                "To Check non-diabetic data",
            ],
        )

        # Choose factors to analyse data
        option = right_column.selectbox(
            "Using which factor you would like to Analyse...",
            (
                "AgeGroup",
                "BPGroup",
                "BMIGroup",
                "DiabetesPedigreeFunction",
                "Glucose",
                "Insulin",
                "Pregnancies",
                "SkinThickness",
            ),
        )
        if option == option and radio_btn == "***Non-Diabetic***":
            right_column.write("Choose any one factor to analyse factor")
            groupedvalues = df_non_diabetic.groupby(option).sum().reset_index()
            fig, ax = plt.subplots(figsize=(12, 7))
            plot = sns.barplot(
                x=option,
                y="Outcome",
                data=groupedvalues,
                err_kws={"linewidth": 0},
                color="Green",
            )
            plot.set(
                xlabel=option,
                ylabel="Count of Non-Diabetic patients in the group " + option,
                title="Non Diabetic Patients within group " + option,
            )
            ax.bar_label(ax.containers[0])
            right_column.pyplot(plot.figure)
        else:
            groupedvalues = df_diabetic.groupby(option).sum().reset_index()

            fig, ax = plt.subplots(figsize=(15, 10))
            plot = sns.barplot(
                x=option,
                y="Outcome",
                data=groupedvalues,
                err_kws={"linewidth": 0},
                color="Tomato",
            )
            plot.set(
                xlabel=option,
                ylabel="Count of Diabetic patients in the group " + option,
                title="Diabetic Patients within group " + option,
            )
            ax.bar_label(ax.containers[0])
            right_column.pyplot(plot.figure)

        X = df.drop(["Outcome", "AgeGroup", "BPGroup", "BMIGroup"], axis=1)
        y = df["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=7
        )
        fill_values = SimpleImputer(missing_values=0, strategy="mean")
        X_train = fill_values.fit_transform(X_train)
        X_test = fill_values.fit_transform(X_test)
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train, y_train)
        # time.sleep(3)
        features = rfc.feature_importances_ * 100
        st.markdown(
            "<h6 style='text-align: center; color: grey;'>Below graph shows percent contribution towards diabetes</h6>",
            unsafe_allow_html=True,
        )
        right_column.bar_chart(pd.Series(features, X.columns))
