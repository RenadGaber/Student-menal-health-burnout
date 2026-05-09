import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Student Burnout Predictor", layout="wide")

# Load models and preprocessing objects
@st.cache_resource
def load_assets():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open('target_encoder.pkl', 'rb') as f:
        le_target = pickle.load(f)
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    return model, scaler, le_dict, le_target, results

try:
    model, scaler, le_dict, le_target, results = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Prediction", "Dataset Explorer", "Model Performance"])

if page == "Live Prediction":
    st.title("🎓 Student Burnout Predictor")
    st.write("Enter your details to predict your burnout level.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 17, 30, 20)
        gender = st.selectbox("Gender", le_dict['gender'].classes_)
        course = st.selectbox("Course", le_dict['course'].classes_)
        year = st.selectbox("Year", le_dict['year'].classes_)
        daily_study_hours = st.slider("Daily Study Hours", 0.0, 15.0, 5.0)
        daily_sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.0)
        screen_time_hours = st.slider("Daily Screen Time (Hours)", 0.0, 20.0, 4.0)
        attendance = st.slider("Attendance Percentage", 0, 100, 85)
        cgpa = st.slider("CGPA", 0.0, 10.0, 8.0)
        
    with col2:
        physical_activity = st.slider("Physical Activity (Hours)", 0.0, 10.0, 1.0)
        anxiety_score = st.slider("Anxiety Score (1-10)", 1, 10, 5)
        depression_score = st.slider("Depression Score (1-10)", 1, 10, 5)
        stress_level = st.selectbox("Stress Level", le_dict['stress_level'].classes_)
        extracurricular = st.slider("Extracurricular Hours", 0.0, 10.0, 2.0)
        financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 3)
        social_support = st.selectbox("Social Support", le_dict['social_support'].classes_)
        family_history = st.selectbox("Family History of Mental Illness", le_dict['family_history_mental_illness'].classes_)

    if st.button("Predict Burnout Level"):
        # Prepare input data
        input_data = {
            'age': age,
            'gender': gender,
            'course': course,
            'year': year,
            'daily_study_hours': daily_study_hours,
            'daily_sleep_hours': daily_sleep_hours,
            'screen_time_hours': screen_time_hours,
            'attendance_percentage': attendance,
            'cgpa': cgpa,
            'physical_activity_hours': physical_activity,
            'anxiety_score': anxiety_score,
            'depression_score': depression_score,
            'stress_level': stress_level,
            'extracurricular_hours': extracurricular,
            'financial_stress': financial_stress,
            'social_support': social_support,
            'family_history_mental_illness': family_history
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical
        for col, le in le_dict.items():
            input_df[col] = le.transform(input_df[col])
            
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        result = le_target.inverse_transform(prediction)[0]
        
        # Display result
        st.subheader("Prediction Result:")
        if result == 'Low':
            st.success(f"Your burnout level is: **{result}**")
        elif result == 'Medium':
            st.warning(f"Your burnout level is: **{result}**")
        else:
            st.error(f"Your burnout level is: **{result}**")

elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.write("Insights from the 150,000 student records.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Burnout Level Distribution")
        st.image('plots/burnout_distribution.png')
        
        st.subheader("Sleep vs Burnout")
        st.image('plots/sleep_vs_burnout.png')
        
    with col2:
        st.subheader("Study Hours vs Burnout")
        st.image('plots/study_vs_burnout.png')
        
        st.subheader("Anxiety vs Burnout")
        st.image('plots/anxiety_vs_burnout.png')
    
    st.subheader("Feature Correlations")
    st.image('plots/correlation_matrix.png')

elif page == "Model Performance":
    st.title("📈 Model Performance Comparison")
    st.write("Comparison of different machine learning models.")
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [m['accuracy'] for m in results.values()],
        'F1-Score': [m['f1_score'] for m in results.values()]
    })
    
    st.table(metrics_df)
    
    st.subheader("Confusion Matrices")
    cols = st.columns(len(results))
    for i, (name, metrics) in enumerate(results.items()):
        with cols[i]:
            st.write(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le_target.classes_, yticklabels=le_target.classes_)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            
    st.subheader("Classification Reports")
    for name, metrics in results.items():
        with st.expander(f"View Report for {name}"):
            st.text(metrics['report'])
