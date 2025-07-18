from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import joblib # *** CAMBIADO: Usamos joblib para cargar los modelos ***
import numpy as np
import pandas as pd
import os

# Crear la instancia de la aplicación FastAPI
app = FastAPI()

# Definir la ruta base para archivos estáticos
# Asegúrate de que esta ruta apunta a la carpeta donde tienes index.html
app.mount("/static", StaticFiles(directory="."), name="static")

# Ruta para servir el archivo HTML principal
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo index.html no encontrado. Asegúrate de que esté en la misma carpeta que main.py.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=500, detail="Error de codificación al leer index.html. Verifica la codificación del archivo.")


# --- Carga de modelos y escaladores ---
# Asegúrate de que la ruta a la carpeta 'models' sea correcta
MODELS_DIR = "models"

# Cargar modelos de clasificación
try:
    # *** CAMBIO IMPORTANTE: Ahora usamos joblib.load ***
    modelo_clasificacion = joblib.load(os.path.join(MODELS_DIR, 'modelo_clasificacion.pkl'))
    scaler_clasificacion = joblib.load(os.path.join(MODELS_DIR, 'scaler_clasificacion.pkl'))
    columnas_clasificacion = joblib.load(os.path.join(MODELS_DIR, 'columnas_clasificacion.pkl')) # Ahora sí las cargamos
    print("Modelos de clasificación cargados exitosamente.")
except FileNotFoundError as e:
    raise RuntimeError(f"Error al cargar archivos del modelo de clasificación: {e}. Asegúrate de que los archivos .pkl estén en la carpeta '{MODELS_DIR}'.")
except Exception as e:
    raise RuntimeError(f"Error inesperado al cargar el modelo de clasificación: {e}")

# Cargar modelos de regresión
try:
    # *** CAMBIO IMPORTANTE: Ahora usamos joblib.load ***
    modelo_regresion = joblib.load(os.path.join(MODELS_DIR, 'modelo_regresion.pkl'))
    scaler_regresion = joblib.load(os.path.join(MODELS_DIR, 'scaler_regresion.pkl'))
    columnas_regresion = joblib.load(os.path.join(MODELS_DIR, 'columnas_regresion.pkl')) # Ahora sí las cargamos
    print("Modelos de regresión cargados exitosamente.")
except FileNotFoundError as e:
    raise RuntimeError(f"Error al cargar archivos del modelo de regresión: {e}. Asegúrate de que los archivos .pkl estén en la carpeta '{MODELS_DIR}'.")
except Exception as e:
    raise RuntimeError(f"Error inesperado al cargar el modelo de regresión: {e}")


# --- Modelos Pydantic para la validación de entrada ---
# Los nombres de los campos deben coincidir con los que envía el JavaScript (camelCase)
# ¡Tipos de datos actualizados a float para consistencia con JavaScript y entrenamiento!
class ClasificacionInput(BaseModel):
    roundHeadshots: float = Field(..., example=5.0, description="Headshots del jugador en la ronda")
    teamEquipmentValue: float = Field(..., example=8000.0, description="Valor total del equipamiento del equipo")
    primaryAssaultRifle: int = Field(..., example=1, description="1 si usa rifle de asalto principal, 0 si no")
    timeAlive: float = Field(..., example=60.0, description="Tiempo vivo en la ronda en segundos")

class RegresionInput(BaseModel):
    roundHeadshots: float = Field(..., example=5.0, description="Headshots del jugador en la ronda")
    teamEquipmentValue: float = Field(..., example=8000.0, description="Valor total del equipamiento del equipo")
    primaryAssaultRifle: int = Field(..., example=1, description="1 si usa rifle de asalto principal, 0 si no")
    timeAlive: float = Field(..., example=60.0, description="Tiempo vivo en la ronda en segundos")

# Mapeo para la salida de clasificación (si 1 es Ganar y 0 es Perder)
outcome_map = {1: "Victoria del Equipo", 0: "Derrota del Equipo"}

# --- Rutas de Predicción ---

@app.post("/predict/clasificacion")
async def predict_clasificacion(data: ClasificacionInput):
    try:
        # Crear DataFrame con los datos de entrada, usando las columnas cargadas
        # Esto asegura que el orden de las columnas sea el mismo que el modelo espera
        input_data_dict = {
            'RoundHeadshots': data.roundHeadshots,
            'TeamStartingEquipmentValue': data.teamEquipmentValue,
            'PrimaryAssaultRifle': data.primaryAssaultRifle,
            'TimeAlive': data.timeAlive
        }
        input_df = pd.DataFrame([input_data_dict], columns=columnas_clasificacion)
        
        # Escalar los datos
        input_scaled = scaler_clasificacion.transform(input_df)

        # Realizar predicción de clase
        prediction_id = int(modelo_clasificacion.predict(input_scaled)[0])
        prediction_label = outcome_map.get(prediction_id, "Resultado Desconocido")

        # Obtener probabilidades (si el modelo lo permite)
        if hasattr(modelo_clasificacion, 'predict_proba'):
            probabilities = modelo_clasificacion.predict_proba(input_scaled)[0]
            win_probability = round(probabilities[1] * 100, 2)  # Probabilidad de clase 1 (Victoria)
            loss_probability = round(probabilities[0] * 100, 2) # Probabilidad de clase 0 (Derrota)
        else:
            win_probability = None
            loss_probability = None

        return {
            "success": True,
            "predicted_class_id": prediction_id,
            "prediction_label": prediction_label,
            "win_probability": win_probability,
            "loss_probability": loss_probability # <--- Esto se envía al frontend
        }
    except Exception as e:
        # Devolver success: False en caso de error para que el frontend lo maneje
        raise HTTPException(status_code=500, detail={"success": False, "error": f"Error en la predicción de clasificación: {e}"})

@app.post("/predict/regresion")
async def predict_regresion(data: RegresionInput):
    try:
        # Crear DataFrame con los datos de entrada, usando las columnas cargadas
        # Esto asegura que el orden de las columnas sea el mismo que el modelo espera
        input_data_dict = {
            'RoundHeadshots': data.roundHeadshots,
            'TeamStartingEquipmentValue': data.teamEquipmentValue,
            'PrimaryAssaultRifle': data.primaryAssaultRifle,
            'TimeAlive': data.timeAlive
        }
        input_df = pd.DataFrame([input_data_dict], columns=columnas_regresion)
        
        # Escalar los datos
        input_scaled = scaler_regresion.transform(input_df)

        # Realizar predicción de regresión
        predicted_kills = float(modelo_regresion.predict(input_scaled)[0])

        return {
            "success": True,
            "predicted_kills": round(predicted_kills, 2)
        }
    except Exception as e:
        # Devolver success: False en caso de error para que el frontend lo maneje
        raise HTTPException(status_code=500, detail={"success": False, "error": f"Error en la predicción de regresión: {e}"})

# Si ejecutas directamente este archivo
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) # Usar 0.0.0.0 para acceso externo, y reload=True para desarrollo