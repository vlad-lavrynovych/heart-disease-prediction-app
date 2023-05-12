import os

import joblib
import pandas as pd
import streamlit as st
from keras.saving.saving_api import load_model
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
st.title("Прогнозування серцево-судинних хвороб")

models = {}
for file in os.listdir("."):
    if file.endswith(".pkl"):
        models[file] = joblib.load(open(file, "rb"))

models["Нейронна мережа"] = load_model("my_model.h5")

YES_NO = ['Ні', 'Так']
CHOLEST = ["Норма - завжди в межах норми",
                                    "Вище в межах норми - від 4,7 ммоль/л до 5",
                                    "Іноді вище норми",
                                    "Завжди вище норми"]
SEX = ["Чоловіча", "Жіноча"]

sex = st.selectbox("Стать", options=SEX)
age = st.number_input("Вік", 0, 90, 30)
height = st.slider('Зріст (см)', value=180.0, min_value=100.0, max_value=220.0, step=1.0, format="%.1f")
weight = st.slider('Вага (кг)', value=75.0, min_value=0.0, max_value=150.0, step=1.0, format="%.1f")
ChSS = st.number_input("Пульс (в стані спокою)", 0, 120, 70)
ADsist = st.number_input("Систолічний тиск", 90, 160, 120)
ADdiast = st.number_input("Діастолічний тиск", 50, 120, 80)

cholesterin = st.selectbox("Показник холестерину: ",
                           options=CHOLEST)

diabetus = st.selectbox("Ви хворієте на діабет?", options=YES_NO)

OP = st.selectbox("Чи були оперативні втручання стосовно серцево-судинної системи?", options=YES_NO)

Shunt = st.selectbox("Чи робили Вам шунтування артерій?", options=YES_NO)

AGtherapia = st.selectbox("Чи проходили ви антигіпертензивну терапію?", options=YES_NO)

IMT = round((weight / height ** 2), 2)
data = {
    "OP": [YES_NO.index(OP)],
    "Shunt": [YES_NO.index(Shunt)],
    "age": [age],
    "height": [height],
    "weight": [weight],
    "IMT": [IMT],
    "sex": [SEX.index(sex)],
    "ChSS": [ChSS],
    "AD sist.": [ADsist],
    "AD diast": [ADdiast],
    "AG therapia": [YES_NO.index(AGtherapia)],
    "cholesterin": [CHOLEST.index(cholesterin)],
    "diabetus melitus": [YES_NO.index(diabetus)],
}
input_df = pd.DataFrame(data, index=[0])

st.divider()

col1, col2 = st.columns([1, 1])
with col1:
    submit = st.button("Передбачити")
with col2:
    model = st.selectbox("Оберіть модель: ", options=list(models.keys()))

if submit:
    print(input_df)
    if model == "Нейронна мережа":
        model = models[model]
        prediction = model.predict(input_df)
        prediction = round(prediction[0][0] * 100, 2)
    # prediction_prob = model.predict_proba(input_df)
    else:
        model = models[model]
        prediction = model.predict_proba(input_df)
        prediction = round(prediction[0][1] * 100, 2)
    st.markdown(f"**Вірогідність наявності серцево-судинної хвороби становить {prediction}%**")

