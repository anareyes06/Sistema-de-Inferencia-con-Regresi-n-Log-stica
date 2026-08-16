import joblib
import pandas as pd
from pathlib import Path


# Ruta al modelo guardado
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "bank_marketing_pipeline.joblib"

# Cargar el pipeline entrenado
model = joblib.load(MODEL_PATH)


def predict_client(data: dict):

    # Convertir los datos recibidos a DataFrame
    client_df = pd.DataFrame([data])

    # Obtener predicción
    prediction = model.predict(client_df)[0]

    # Obtener probabilidades
    probabilities = model.predict_proba(client_df)[0]

    # Identificar la posición de la clase "yes"
    classes = model.named_steps["classifier"].classes_
    yes_index = list(classes).index("yes")

    probability = probabilities[yes_index]

    # Clasificación amigable
    if prediction == "yes":
        classification = "Potencialmente interesado"
    else:
        classification = "Baja propensión"

    return {
        "prediction": prediction,
        "probability": float(probability),
        "classification": classification
    }