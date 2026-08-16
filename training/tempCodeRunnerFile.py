import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# cargamos el dataset
df = pd.read_csv('data/bank.csv', sep = ';')

print(df.head())
df.info()

# variables
features = ["age", "job", "marital", "education", "balance", "housing", "loan", "campaign"]

# creamos x e y
X = df[features]
y = df["y"]

print(X.head())
print(y.head())

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Tamaño X_train:", X_train.shape)
print("Tamaño X_test:", X_test.shape)

print("\nDistribución en entrenamiento:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en prueba:")
print(y_test.value_counts(normalize=True))


# identificar el tipo de variables
numeric_features = [
    "age",
    "balance",
    "campaign"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "housing",
    "loan"
]

# preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# crear pipeline completo
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)

# entrenar el modelo
model.fit(X_train, y_train)

# realizar predicciones
y_pred = model.predict(X_test)

print(y_pred[:10])

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="yes")
recall = recall_score(y_test, y_pred, pos_label="yes")
f1 = f1_score(y_test, y_pred, pos_label="yes")

print("\nMétricas del modelo:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nPredicciones del modelo:")
print(pd.Series(y_pred).value_counts())

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])

print("\nMatriz de confusión:")
print(cm)

print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, labels=["no", "yes"]))

# guardar el pipeline
joblib.dump(model, "models/bank_marketing_pipeline.joblib")

print("\nListo.")