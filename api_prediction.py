
# api_prediction.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import io
import base64

# -----------------------------
# 1️⃣ Initialisation FastAPI
# -----------------------------
app = FastAPI(title="API de Prédiction ENSPD", 
              description="Prédiction de la réussite académique avec Logistic Regression ou Decision Tree",
              version="1.0")

# -----------------------------
# 2️⃣ Définir le modèle de données (un étudiant)
# -----------------------------
class Student(BaseModel):
    age: float
    year_of_study: int
    attendance_rate: float
    num_absences: float
    assignments_avg: float
    midterm_score: float
    final_score: float
    project_participation: float
    lab_hours_per_week: float
    infra_usage_hours_month: float
    extracurricular_count: float
    promotion_size: float
    gender: str
    filiere: str
    scholarship: str
    socioeconomic_status: str
    lab_access: str

# -----------------------------
# 3️⃣ Charger les datasets et entraîner les modèles
# -----------------------------
df = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").values.ravel()

# Colonnes inutiles
cols_to_drop = ["student_id", "supervisor_assigned", "classroom_allocated"]
df.drop(columns=cols_to_drop, inplace=True)

# Encodage
cat_cols = ["gender", "filiere", "scholarship", "socioeconomic_status", "lab_access"]
df_encoded = pd.get_dummies(df, columns=cat_cols)
feature_names = df_encoded.columns

X_train = df_encoded.values
y_train = y

# ----- Logistic Regression à la main -----
class LogisticRegressionManual:
    def __init__(self, learningrate=0.1, iterations=1000):
        self.lr = learningrate
        self.iterations = iterations
        self.weights = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        m, n = X.shape
        self.weights = np.zeros(n)
        for i in range(self.iterations):
            z = X @ self.weights
            y_pred = self.sigmoid(z)
            gradient = (1/m)*(X.T @ (y_pred - y))
            self.weights -= self.lr * gradient

    def predict(self, X):
        X = np.array(X, dtype=float)
        return (self.sigmoid(X @ self.weights) >= 0.5).astype(int)

log_model = LogisticRegressionManual()
log_model.fit(X_train, y_train)

# ----- Decision Tree -----
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

# -----------------------------
# 4️⃣ Fonction pour transformer un étudiant en dataframe encodé
# -----------------------------
def preprocess_student(student: Student):
    data = pd.DataFrame([student.dict()])
    # encoder les mêmes colonnes que le training set
    data_encoded = pd.get_dummies(data)
    # Ajouter les colonnes manquantes avec 0
    for col in feature_names:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    # Reordonner les colonnes pour correspondre au training set
    data_encoded = data_encoded[feature_names]
    return data_encoded.values

# -----------------------------
# 5️⃣ Fonction pour générer graphique importance features
# -----------------------------
def plot_feature_importance(model_type="logistic"):
    plt.figure(figsize=(12,5))
    if model_type == "logistic":
        importance = np.abs(log_model.weights)
        plt.bar(range(len(importance)), importance)
    else:
        importance = tree_model.feature_importances_
        plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.title(f"Importance des features - {model_type}")
    # Sauver en base64 pour l'API
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img_base64

# -----------------------------
# 6️⃣ Fonction pour générer arbre Decision Tree
# -----------------------------
def plot_tree_graph():
    plt.figure(figsize=(20,10))
    plot_tree(tree_model, feature_names=feature_names, class_names=["Fail","Pass"], filled=True, rounded=True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img_base64

# -----------------------------
# 7️⃣ Endpoints API
# -----------------------------
@app.post("/predict")
def predict(student: Student, model: str = Query("logistic", enum=["logistic","tree"])):
    X_new = preprocess_student(student)
    if model == "logistic":
        y_pred = log_model.predict(X_new)[0]
        importance_img = plot_feature_importance("logistic")
        tree_img = None
    else:
        y_pred = tree_model.predict(X_new)[0]
        importance_img = plot_feature_importance("tree")
        tree_img = plot_tree_graph()
    result = {
        "prediction": "Pass" if y_pred==1 else "Fail",
        "importance_features": importance_img,
        "tree_graph": tree_img
    }
    return result

# -----------------------------
# 8️⃣ Run: uvicorn api_prediction:app --reload
# -----------------------------

uvicorn api_prediction:app --reload


