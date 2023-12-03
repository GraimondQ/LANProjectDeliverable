import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier



@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('catboost_funzioni.cbm', format='cbm')
    return model


model = load_model()

def predict(data):
    prediction = model.predict(data)
    return prediction


def main():
    st.title("CatBoost Prediction App")

    with st.form("manual_entry_form"):
        ruolo = st.text_input("Ruolo")
        posizione = st.text_input("Posizione")
        azienda = st.text_input("Azienda")
        genere = st.selectbox("Genere", ["Maschile", "Femminile"])
        studi = st.text_input("Studi")
        eta = st.number_input("Età", min_value=0, max_value=100, step=1)
        submitted = st.form_submit_button("Submit")
        if submitted:

            data = pd.DataFrame([[ruolo, posizione, azienda, genere, studi, eta]],
                                columns=["ruolo", "posizione", "azienda", "Genere", "studi", "Età"])

            prediction = predict(data)
            st.write("Prediction:", prediction)


    uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["csv", "xlsx"])
    if uploaded_file is not None:

        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Prediction
        if st.button("Predict"):
            prediction = predict(data)
            st.write("Predictions:", prediction)

if __name__ == "__main__":
    main()
