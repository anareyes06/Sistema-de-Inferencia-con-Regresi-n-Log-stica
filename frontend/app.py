import streamlit as st
import requests


# Dirección de nuestra API
API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦"
)

st.title("🏦 Estimación de propensión")
st.write(
    "Ingresa la información del cliente para estimar "
    "la probabilidad de contratación de un depósito a plazo."
)


# Formulario
with st.form("client_form"):

    age = st.number_input(
        "Edad",
        min_value=18,
        max_value=100,
        value=41
    )

    job = st.selectbox(
        "Ocupación",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

    marital = st.selectbox(
        "Estado civil",
        [
            "married",
            "single",
            "divorced"
        ]
    )

    education = st.selectbox(
        "Nivel educativo",
        [
            "primary",
            "secondary",
            "tertiary",
            "unknown"
        ]
    )

    balance = st.number_input(
        "Balance anual promedio",
        value=0
    )

    housing = st.selectbox(
        "Crédito hipotecario",
        ["yes", "no"]
    )

    loan = st.selectbox(
        "Préstamo personal",
        ["yes", "no"]
    )

    campaign = st.number_input(
        "Número de contactos durante la campaña",
        min_value=1,
        value=1
    )

    submitted = st.form_submit_button("Estimar propensión")


# Cuando el usuario presione el botón
if submitted:

    client_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "campaign": campaign
    }

    try:
        response = requests.post(
            API_URL,
            json=client_data,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["probability"]
            prediction = result["prediction"]
            classification = result["classification"]

            st.success("Predicción realizada correctamente")

            st.metric(
                "Probabilidad estimada de contratación",
                f"{probability:.1%}"
            )

            st.write(f"**Predicción:** {prediction}")
            st.write(f"**Clasificación:** {classification}")

            st.caption(
                "El resultado representa una estimación del modelo "
                "y no una certeza sobre el comportamiento futuro del cliente."
            )

        else:
            st.error("La API rechazó los datos enviados.")
            st.write(response.json())

    except requests.exceptions.RequestException:
        st.error(
            "No fue posible conectarse con la API. "
            "Verifica que FastAPI esté ejecutándose."
        )