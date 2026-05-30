import streamlit as st
import numpy as np
import joblib
model=joblib.load("student_model.pkl")
st.set_page_config(page_title="Student Predictor", page_icon="🎓")
st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter student details below to predict their result using AI!")
st.divider()
st.subheader("📝 Enter Student Details")
name=st.text_input("Student name",placeholder="Enter Student Name")
marks=st.slider("Marks(out of 100)",0,100,65)
attendance=st.slider("Attendance %",0,100,75)
study_hours=st.slider("Study hours per week",0,10,4)
st.divider()
if st.button("🔮 Predict Result",use_container_width=True):
    features=np.array([[marks,attendance,study_hours]])
    prediction=model.predict(features)[0]
    confidence=model.predict_proba(features)[0]
    if prediction==1:
        conf=confidence[1]*100
        st.success(f"✅ {name if name else 'Student'} is predicted to PASS!")
        st.metric("Confidence", f"{conf:.0f}%")
    else:
        conf=confidence[0]*100
        st.error(f"❌ {name if name else 'Student'} is predicted to FAIL!")
        st.metric("Confidence", f"{conf:.0f}%")
    st.divider()
    st.subheader("💡 Advice")
    if marks < 65:
        st.warning("📚 Marks are below passing threshold. Focus on scoring above 65.")
        if attendance < 75:
            st.warning("🏫 Attendance is low. Try to attend at least 75% of classes.")
        if study_hours < 3:
            st.warning("⏰ Study hours are low. Aim for at least 3 hours daily.")
        if prediction == 1 and marks >= 65 and attendance >= 75:
            st.balloons()
            st.info("🌟 Keep it up! Great performance across all areas.")