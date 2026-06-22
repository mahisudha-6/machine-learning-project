import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\AI Training\Titanic.csv")
data['Age'] = data['Age'].fillna(data['Age'].mean())
le= LabelEncoder()
le= LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
# Select Features and Target
X = data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = data['Survived']

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ['Male', 
                           'Female'])
# 0 = Female, 1 = Male

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    step=1
)

sibsp = st.number_input(
    "Siblings/Spouses",
    min_value=0,
    step=1
)

parch = st.number_input(
    "Parents/Children",
    min_value=0,
    step=1
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    step=1.0
)
sex=1 if sex == 'Male' else 0
if st.button("Predict Survival"):

    prediction = model.predict(
        [[pclass, sex, age, sibsp, parch, fare]]
    )

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")