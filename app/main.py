from fastapi import FastAPI

from app.schemas import ClientData
from app.inference import predict_client


# Crear aplicación FastAPI
app = FastAPI(
    title="Bank Marketing Predictor",
    description="API para estimar la propensión de contratación de un depósito a plazo"
)


# Endpoint sencillo para comprobar que la API funciona
@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}


# Endpoint de inferencia
@app.post("/predict")
def predict(data: ClientData):

    # Convertir el modelo de Pydantic a diccionario
    client_data = data.model_dump()

    # Realizar inferencia
    result = predict_client(client_data)

    return result