import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib # Usamos joblib para guardar y cargar los modelos
import matplotlib.pyplot as plt
import os # Importar os para crear la carpeta models si no existe

# 1. Cargar dataset
# Asegúrate de que la ruta a tu archivo CSV sea correcta.
# Si 'data' está en la misma carpeta que este script, es correcto.
df = pd.read_csv("data/AnexoET_RoundWinner_Limpio.csv", sep=",")
df_backup = df.copy()

# 2. Limpieza (conservando TimeAlive que necesitamos)
df_backup.drop(columns=['Unnamed: 0', 'AbnormalMatch', 'FirstKillTime', 'TravelledDistance', 
                        'InternalTeamId', 'MatchId', 'RLethalGrenadesThrown', 
                        'RNonLethalGrenadesThrown'], inplace=True)
df_backup.dropna(inplace=True)

# 3. Filtrado
df_backup = df_backup[df_backup['MatchKills'] <= 28]
df_backup = df_backup[df_backup['MatchAssists'] <= 8]
df_backup = df_backup[(df_backup['RoundId'] >= 1) & (df_backup['RoundId'] <= 30)]

# 4. Transformaciones
le = LabelEncoder()
df_backup['Team'] = le.fit_transform(df_backup['Team'])
df_backup['Map'] = le.fit_transform(df_backup['Map'])

# Convertir columnas booleanas a 1/0
df_backup['RoundWinner'] = df_backup['RoundWinner'].astype(int)
df_backup['MatchWinner'] = df_backup['MatchWinner'].astype(int)
df_backup['Survived'] = df_backup['Survived'].astype(int)

# Asegurar que las columnas de armas sean binarias
weapon_cols = ['PrimaryAssaultRifle', 'PrimarySniperRifle', 
                'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']
for col in weapon_cols:
    df_backup[col] = df_backup[col].astype(int)

# 5. Modelo de REGRESIÓN (predecir RoundKills)
# Las características deben ser las mismas y en el mismo orden que las enviadas por el frontend
X_reg = df_backup[['RoundHeadshots', 'TeamStartingEquipmentValue', 'PrimaryAssaultRifle', 'TimeAlive']]
y_reg = df_backup['RoundKills']

# Escalado y división de datos
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Entrenamiento
modelo_reg = LinearRegression()
modelo_reg.fit(X_train_reg, y_train_reg)

# 6. Modelo de CLASIFICACIÓN (predecir RoundWinner)
# Las características deben ser las mismas y en el mismo orden que las enviadas por el frontend
X_clf = df_backup[['RoundHeadshots', 'TeamStartingEquipmentValue', 'PrimaryAssaultRifle', 'TimeAlive']]
y_clf = df_backup['RoundWinner']

# Escalado y división
scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Entrenamiento
modelo_clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)
modelo_clf.fit(X_train_clf, y_train_clf)

# Guardar modelos y scalers
# Asegúrate de que la carpeta 'models' exista, si no, la crea.
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Guardamos con joblib.dump
joblib.dump(modelo_reg, os.path.join(MODELS_DIR, "modelo_regresion.pkl"))
joblib.dump(scaler_reg, os.path.join(MODELS_DIR, "scaler_regresion.pkl"))
joblib.dump(X_reg.columns.tolist(), os.path.join(MODELS_DIR, "columnas_regresion.pkl"))

joblib.dump(modelo_clf, os.path.join(MODELS_DIR, "modelo_clasificacion.pkl"))
joblib.dump(scaler_clf, os.path.join(MODELS_DIR, "scaler_clasificacion.pkl"))
joblib.dump(X_clf.columns.tolist(), os.path.join(MODELS_DIR, "columnas_clasificacion.pkl"))

print("\nModelos, scalers y listas de columnas guardados exitosamente en la carpeta 'models/'.")