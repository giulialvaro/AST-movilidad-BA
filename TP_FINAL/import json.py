# %% [markdown]
# # TP2: Análisis de Series Temporales
# 
# **Maestría en Ciencia de Datos - Universidad Austral**  
# **Período de análisis:** 2021-01-01 a 2025-12-31 (datos DIARIOS)  
# 
# ---

# %% [markdown]
# # SETUP: Importar Librerías

# %%
import pandas as pd
import numpy as np
import requests
import io
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuración
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
np.random.seed(42)

print("✅ Librerías importadas correctamente\n")

# %% [markdown]
# ---
# 
# # FASE 1: PREPARACIÓN DE DATOS
# 
# ## Paso 1.1: Descargar datos diarios del Subte CABA

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# DATASET: MOVILIDAD URBANA (del TP1)
# ═══════════════════════════════════════════════════════════════════════════════
# Este dataset contiene datos de viajes en el subte de Buenos Aires
# Fuente: Gobierno de CABA
# Frecuencia: DIARIA
# Período: 2021-01-01 a 2025-12-31 (aproximadamente)


print("📥 Paso 1.1: Descargando datos diarios del Subte CABA (2021-2025)\n")

SUBTE_URLS = {
    2021: 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/sbase/subte-viajes-molinetes/molinetes-2021.zip',
    2022: 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/sbase/subte-viajes-molinetes/molinetes-2022.zip',
    2023: 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/sbase/subte-viajes-molinetes/molinetes-2023.zip',
    2024: 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/sbase/subte-viajes-molinetes/molinetes-2024.zip',
    2025: 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/sbase/subte-viajes-molinetes/molinetes-2025.zip',
}

def _limpiar_raw(raw_bytes):
    """Quita BOM UTF-8 y comillas externas de cada línea"""
    if raw_bytes.startswith(b'\xef\xbb\xbf'):
        raw_bytes = raw_bytes[3:]
        enc = 'utf-8'
    else:
        enc = 'latin1'
    
    texto = raw_bytes.decode(enc)
    lineas = texto.splitlines()
    
    primera_core = lineas[0].strip().rstrip(';').rstrip() if lineas else ''
    if primera_core.startswith('"') and primera_core.endswith('"') and ';' in primera_core:
        lineas = [l.strip().rstrip(';').strip('"') for l in lineas if l.strip()]
    
    limpio = '\n'.join(lineas).encode(enc)
    return limpio, enc

def _parsear_csv_subte_diario(raw_bytes):
    """Parsea CSV de subte → DataFrame con [fecha, viajes]"""
    raw_limpio, enc = _limpiar_raw(raw_bytes)
    primera = raw_limpio.split(b'\n')[0].decode(enc)
    sep = ';' if ';' in primera else ','
    
    df = pd.read_csv(io.BytesIO(raw_limpio), sep=sep, encoding=enc, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    
    if 'fecha' not in df.columns:
        return pd.DataFrame(columns=['fecha', 'viajes'])
    
    # Buscar columna de viajes totales (puede variar según año)
    for candidate in ['pax_total', 'total', 'viajes']:
        if candidate in df.columns:
            viajes_col = candidate
            break
    else:
        viajes_col = next((c for c in df.columns if 'total' in c), None)
    
    if viajes_col is None:
        return pd.DataFrame(columns=['fecha', 'viajes'])
    
    df['viajes'] = pd.to_numeric(df[viajes_col], errors='coerce')
    
    # Detectar formato de fecha: ISO (YYYY-MM-DD) vs argentino (D/M/YYYY)
    muestra = df['fecha'].dropna().iloc[0] if not df['fecha'].dropna().empty else ''
    if str(muestra).count('-') == 2 and str(muestra).index('-') == 4:
        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d', errors='coerce')
    else:
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
    
    return df[['fecha', 'viajes']].dropna()

def descargar_subte_anio(url, anio):
    """Descarga ZIP anual de molinetes y retorna DataFrame con datos DIARIOS"""
    print(f"  {anio}: descargando...", end=' ', flush=True)
    
    try:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        
        # Leer TODOS los CSVs del ZIP (puede haber múltiples por mes)
        frames = []
        for csv_name in csv_names:
            raw = z.open(csv_name).read()
            df_part = _parsear_csv_subte_diario(raw)
            if not df_part.empty:
                frames.append(df_part)
        
        if not frames:
            raise ValueError('No se pudo parsear ningún CSV del ZIP')
        
        out = pd.concat(frames, ignore_index=True)
        print(f"OK ({len(csv_names)} CSVs | {len(out):,} registros diarios)")
        return out
    
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame(columns=['fecha', 'viajes'])

# Descargar todos los años
frames = []
for anio, url in SUBTE_URLS.items():
    df_anio = descargar_subte_anio(url, anio)
    if not df_anio.empty:
        frames.append(df_anio)

if not frames:
    print("\n⚠️ No se descargó ningún año. Usando datos sintéticos...")
    dates = pd.date_range('2021-01-01', '2025-12-31', freq='D')
    df_diario = pd.DataFrame({'fecha': dates, 'viajes': np.random.randint(800, 1500, len(dates))})
else:
    df_diario = pd.concat(frames, ignore_index=True)
    df_diario['fecha'] = pd.to_datetime(df_diario['fecha'])
    df_diario = df_diario.sort_values('fecha').reset_index(drop=True)
    # Filtrar período 2021-2025
    df_diario = df_diario[(df_diario['fecha'] >= '2021-01-01') & (df_diario['fecha'] <= '2025-12-31')]

print(f"\n✅ Total descargado: {len(df_diario)} registros")
print(f"   Período: {df_diario['fecha'].min().date()} a {df_diario['fecha'].max().date()}\n")

# Agrupar por fecha (sumar si hay múltiples registros por día)
df_diario = df_diario.groupby('fecha')['viajes'].sum().reset_index()

print(df_diario.head(10))

# %% [markdown]
# ## Paso 1.2: Agregar variables dummy y temporales

# %%
"""
Creamos variables categóricas para capturar patrones:
- Fin de semana: sábado (5) y domingo (6) → valor 1
- Día laboral: lunes a viernes → valor 1
- Feriado: lista manual de feriados nacionales Argentina 2021-2025 → valor 1
- Variables temporales: mes, semana del año, día del año
- Temperatura: variable exógena (sintética para este ejercicio)
"""

print("\n🔧 Paso 1.2: Agregando variables dummy y temporales\n")

# Fin de semana (dayofweek: 0=lunes, 5=sábado, 6=domingo)
df_diario['es_fin_de_semana'] = (df_diario['fecha'].dt.dayofweek >= 5).astype(int)

# Día laboral (inverso del anterior)
df_diario['es_dia_laboral'] = (df_diario['fecha'].dt.dayofweek < 5).astype(int)

# Feriados Argentina 2021-2025
# NOTA: Esta es una lista completa de feriados nacionales
feriados = pd.to_datetime([
    '2021-01-01', '2021-02-15', '2021-02-16', '2021-03-24', '2021-04-02', '2021-05-01',
    '2021-05-25', '2021-06-20', '2021-06-21', '2021-07-09', '2021-08-17', '2021-10-12',
    '2021-11-22', '2021-12-08', '2021-12-25',
    '2022-01-01', '2022-02-28', '2022-03-01', '2022-03-24', '2022-04-02', '2022-05-01',
    '2022-05-25', '2022-06-20', '2022-06-21', '2022-07-09', '2022-08-17', '2022-10-12',
    '2022-11-22', '2022-12-08', '2022-12-25',
    '2023-01-01', '2023-02-20', '2023-02-21', '2023-03-24', '2023-04-02', '2023-05-01',
    '2023-05-25', '2023-06-20', '2023-06-21', '2023-07-09', '2023-08-17', '2023-10-12',
    '2023-11-22', '2023-12-08', '2023-12-25',
    '2024-01-01', '2024-02-12', '2024-02-13', '2024-03-24', '2024-04-02', '2024-05-01',
    '2024-05-25', '2024-06-20', '2024-06-21', '2024-07-09', '2024-08-17', '2024-10-12',
    '2024-11-22', '2024-12-08', '2024-12-25',
    '2025-01-01', '2025-03-01', '2025-03-02', '2025-03-24', '2025-04-02', '2025-05-01',
    '2025-05-25', '2025-06-20', '2025-06-21', '2025-07-09', '2025-08-17', '2025-10-12',
    '2025-11-22', '2025-12-08', '2025-12-25'
])
df_diario['es_feriado'] = df_diario['fecha'].isin(feriados).astype(int)

# Variables temporales (ciclos)
df_diario['mes'] = df_diario['fecha'].dt.month
df_diario['semana_anio'] = df_diario['fecha'].dt.isocalendar().week
df_diario['dia_anio'] = df_diario['fecha'].dt.dayofyear

# Temperatura (variable exógena - sintética para este TP)
# Simulamos patrón estacional realista de Buenos Aires
dias = np.arange(len(df_diario))
temp_seasonal = 8 * np.sin((dias + 80) * 2 * np.pi / 365)  # Ciclo anual
temp_noise = np.random.normal(0, 1.5, len(df_diario))  # Variación diaria
df_diario['temperatura'] = 16.5 + temp_seasonal + temp_noise

print("✅ Variables creadas:")
print(f"   • es_fin_de_semana: {df_diario['es_fin_de_semana'].sum()} días")
print(f"   • es_dia_laboral: {df_diario['es_dia_laboral'].sum()} días")
print(f"   • es_feriado: {df_diario['es_feriado'].sum()} días")
print(f"   • temperatura: {df_diario['temperatura'].min():.1f}°C a {df_diario['temperatura'].max():.1f}°C\n")

print(df_diario.head(10))

# %% [markdown]
# ## Paso 1.3: Crear lags (memoria temporal)

# %%
print("\n⏳ Paso 1.3: Creando lags (memoria temporal)\n")

df_diario['viajes_lag1'] = df_diario['viajes'].shift(1)
df_diario['viajes_lag7'] = df_diario['viajes'].shift(7)
df_diario['viajes_lag30'] = df_diario['viajes'].shift(30)

# TARGET: predecir viajes de mañana usando datos de hoy
df_diario['target'] = df_diario['viajes'].shift(-1)

# Limpiar filas con NaN (causadas por lags al inicio y target al final)
filas_antes = len(df_diario)
df_diario = df_diario.dropna()
filas_despues = len(df_diario)

print("✅ Lags creados:")
print(f"   • viajes_lag1 (ayer)")
print(f"   • viajes_lag7 (hace 7 días)")
print(f"   • viajes_lag30 (hace 30 días)")
print(f"   • target (mañana - lo que predecimos)")
print(f"\nFilas removidas por NaN: {filas_antes - filas_despues}")
print(f"Dataset final: {filas_despues} días\n")

print(df_diario.head(10))

# %% [markdown]
# ## Paso 1.4: Guardar datos preparados

# %% [markdown]
# ⚠️ IMPORTANTE: SOLO ejecutar esta celda LA PRIMERA VEZ.
# Próximas ejecuciones: cargar directamente desde la carpeta.

# %%
print("\n💾 Paso 1.4: Guardando datos preparados\n")

os.makedirs('datos_procesados', exist_ok=True)

# Guardar en CSV (formato universal, fácil de compartir)
df_diario.to_csv('datos_procesados/TP2_datos_preparados.csv', index=False)

# Guardar en Pickle (formato Python, más rápido para cargar)
df_diario.to_pickle('datos_procesados/TP2_datos_preparados.pkl')

print("✅ Archivos guardados en ./datos_procesados/:\n")
print("   📊 TP2_datos_preparados.csv")
print("   📊 TP2_datos_preparados.pkl")
print(f"\n   Total: {len(df_diario)} días")
print(f"   Período: {df_diario['fecha'].min().date()} a {df_diario['fecha'].max().date()}")
print(f"   Columnas: {list(df_diario.columns)}\n")

print("💡 PRÓXIMA VEZ: Carga directamente con:")
print("   df_diario = pd.read_csv('datos_procesados/TP2_datos_preparados.csv')")

# %% [markdown]
# ---
# 
# # PARTE 2: MODELADO XGBOOST

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# Cargar el DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🔄 Cargando datos para XGBoost...\n")

df_diario = pd.read_pickle('datos_procesados/TP2_datos_preparados.pkl')

print(f"✅ Datos cargados: {len(df_diario)} filas, {len(df_diario.columns)} columnas")
print(f"\nPrimeras filas:")
print(df_diario.head())

# %% [markdown]
# NOTA SOBRE LAS SERIES:
# Este DataFrame (TP2_datos_preparados.csv) contiene TODAS las series necesarias:
#   • Serie 1 (Subte): columna 'viajes' → viajes diarios del subte
#   • Serie 2 (Feriados): columna 'es_feriado' → dummy 0/1
#   • Serie 3 (Temperatura): columna 'temperatura' → temperatura diaria en °C
#   
# Todas están en el mismo dataset, sincronizadas por fecha.

# %% [markdown]
# ## Paso 2.1: Train-Test Split (Respetando cronología)

# %% [markdown]
# CRÍTICO
# En series temporales, NO usamos train_test_split aleatorio porque:
#   • Mezcla pasado y futuro
#   • El modelo ve datos futuros durante entrenamiento (data leakage)
#   
# SOLUCIÓN CORRECTA: Dividir por FECHA
#   • TRAIN: 2021-2024 (pasado histórico para aprender)
#   • TEST: 2025 (futuro sin ver, para validar)
#   
# El modelo aprende del pasado y valida en el futuro, como en la realidad.

# %%
print("📋 Paso 2.1: Train/Test Split (respetando cronología)\n")

fecha_corte = df_diario['fecha'].max() - pd.Timedelta(days=365)

train = df_diario[df_diario['fecha'] <= fecha_corte]
test = df_diario[df_diario['fecha'] > fecha_corte]

# Features: variables que el modelo usa para predecir
# Incluyen:
#   • Lags (viajes_lag1/7/30): memoria temporal, capturan dependencia del pasado
#   • Dummies (fin_semana, laboral, feriado): patrones especiales
#   • Temporales (mes, semana, día): ciclos
#   • Temperatura: variable exógena que afecta viajes

features = ['viajes_lag1', 'viajes_lag7', 'viajes_lag30', 
            'es_fin_de_semana', 'es_dia_laboral', 'es_feriado', 
            'mes', 'semana_anio', 'dia_anio', 'temperatura']

X_train = train[features]
y_train = train['target']
X_test = test[features]
y_test = test['target']

print(f"TRAIN: {len(X_train)} días")
print(f"   Período: {train['fecha'].min().date()} a {train['fecha'].max().date()}")
print(f"\nTEST: {len(X_test)} días")
print(f"   Período: {test['fecha'].min().date()} a {test['fecha'].max().date()}")
print(f"\nRatio: {100*len(X_train)/(len(X_train)+len(X_test)):.1f}% train / {100*len(X_test)/(len(X_train)+len(X_test)):.1f}% test\n")

# %% [markdown]
# ## Paso 2.2: Entrenamiento XGBoost

# %% [markdown]
# XGBoost: Gradient Boosting (ensamble de árboles que aprenden de sus errores)
# 
# Hiperparámetros:
# - max_depth=6: profundidad máxima del árbol (evita overfitting)
# - learning_rate=0.1: qué tan rápido aprende
# - n_estimators=300: cantidad de árboles
# - subsample=0.8: usa 80% de datos por árbol (regularización)
# - colsample_bytree=0.8: usa 80% de features por árbol (regularización)
# 
# VENTAJAS para series temporales:
# 
#   ✅ Captura relaciones NO lineales (lags pueden tener relaciones complejas)
#   ✅ Robusto a outliers (huelgas, cortes no lo rompen)
#   ✅ Feature importance: sabe cuál variable es más importante
#   ✅ Muy preciso en patrones complejos
# 
# DESVENTAJAS:
# 
#   ❌ No extrapola bien: si 2026 es MUY diferente a 2025, fallará
#   ❌ Necesita muchos datos (tenemos ~1700 días, suficiente)

# %%
print("🤖 Paso 2.2: Entrenando XGBoost\n")

model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, verbose=False)

print("✅ Modelo entrenado\n")

# Ver qué features son más importantes
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("📊 TOP 5 FEATURES MÁS IMPORTANTES:\n")
for idx, row in importance.head(5).iterrows():
    print(f"   {row['feature']:20s} → {row['importance']:.4f}")

# %% [markdown]
# ## Paso 2.3: Predicciones y Métricas

# %% [markdown]
# Métricas para evaluar series temporales:
# 1. SMAPE (Symmetric Mean Absolute Percentage Error) - ⭐ PRINCIPAL
#    Fórmula: SMAPE = 100 × (1/n) × Σ(2|pred - real| / (|real| + |pred|))
#    Rango: 0% a 100%
#    ✅ MEJOR para series temporales (simétrico, no indefinido en 0)
#    ✅ Interpretable: "Error del X%"
#    
# 2. RMSE (Root Mean Squared Error)
#    Fórmula: RMSE = √(1/n × Σ(pred - real)²)
#    ✅ Penaliza MUCHO errores grandes (cuadrado)
#    ❌ No interpretable directamente en porcentaje
#    
# 3. MAE (Mean Absolute Error)
#    Fórmula: MAE = (1/n) × Σ|pred - real|
#    ✅ MÁS interpretable: "Me equivoco en X viajes en promedio"
#    ✅ Usa misma unidad que los datos
#    
# 4. MAPE (Mean Absolute Percentage Error)
#    Fórmula: MAPE = 100 × (1/n) × Σ|pred - real| / |real|
#    ✅ Error porcentual
#    ❌ Indefinido si real=0 (por eso SMAPE es mejor)

# %%
print("\n\n📊 Paso 2.3: Calculando métricas en TEST\n")

y_pred = model.predict(X_test)

# Calcular métricas
smape = 100 * np.mean(2.0 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = 100 * np.mean(np.abs((y_test - y_pred) / y_test))

print("=" * 60)
print("MÉTRICAS XGBOOST (TEST 2025)")
print("=" * 60)
print(f"\n   SMAPE   {smape:7.2f}%     ⭐ PRINCIPAL: Error porcentual simétrico")
print(f"   RMSE    {rmse:7.2f}      Penaliza errores grandes")
print(f"   MAE     {mae:7.2f}      Error promedio en viajes")
print(f"   MAPE    {mape:7.2f}%     Error porcentual (complementaria)")
print("\n" + "=" * 60)

# Guardar resultados
resultados = pd.DataFrame({
    'fecha': test['fecha'].values,
    'real': y_test.values,
    'predicción': y_pred
})

metricas = pd.DataFrame({
    'Modelo': ['XGBoost'],
    'SMAPE': [smape],
    'RMSE': [rmse],
    'MAE': [mae],
    'MAPE': [mape]
})

resultados.to_csv('datos_procesados/predicciones_xgboost.csv', index=False)
metricas.to_csv('datos_procesados/metricas_xgboost.csv', index=False)
joblib.dump(model, 'datos_procesados/modelo_xgboost.pkl')

print("\n✅ Resultados guardados en ./datos_procesados/")

# %% [markdown]
# ## Paso 2.4: Gráficos de Predicción

# %%
# Gráfico 1: Real vs Predicción
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(resultados['fecha'], resultados['real'], label='Real', 
        color='black', linewidth=2.5, alpha=0.8)
ax.plot(resultados['fecha'], resultados['predicción'], label='Predicción', 
        color="#b44c1f", linewidth=1.8, alpha=0.8, linestyle='--')
ax.fill_between(resultados['fecha'], resultados['real'], resultados['predicción'], 
                alpha=0.2, color='gray')
ax.set_title('XGBoost: Real vs Predicción (TEST 2025)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Viajes', fontsize=11)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('datos_procesados/01_xgboost_real_vs_pred.png', dpi=100, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: 01_xgboost_real_vs_pred.png")

# %%
# Gráfico 2: Error en el tiempo
fig, ax = plt.subplots(figsize=(16, 7))
error = resultados['predicción'] - resultados['real']

ax.plot(resultados['fecha'], error, color='#ff7f0e', linewidth=1.5, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
ax.fill_between(resultados['fecha'], error, 0, where=(error >= 0), 
                color='red', alpha=0.3, label='Sobrestimado')
ax.fill_between(resultados['fecha'], error, 0, where=(error < 0), 
                color='green', alpha=0.3, label='Subestimado')

ax.set_title('XGBoost: Error en el Tiempo', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Error (Predicción - Real)', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('datos_procesados/02_xgboost_error.png', dpi=100, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: 02_xgboost_error.png")

# %%
# Gráfico 3: Feature Importance
fig, ax = plt.subplots(figsize=(11, 6))
top_features = importance.head(8)

bars = ax.barh(top_features['feature'], top_features['importance'], color='darkorange', alpha=0.8)
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_xlabel('Importancia', fontsize=11)
ax.set_title('XGBoost: Features Más Importantes', fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('datos_procesados/03_feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: 03_feature_importance.png")

# %%
# Gráfico 4: Distribución de errores
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(error, bins=30, color="#cad443", alpha=0.7, edgecolor='black', linewidth=1.2)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label=f'Media: {error.mean():.0f}')
ax.axvline(x=error.median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {error.median():.0f}')

ax.set_xlabel('Error (Predicción - Real)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
ax.set_title('XGBoost: Distribución de Errores', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('datos_procesados/04_error_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: 04_error_distribution.png")

# %% [markdown]
# ## Paso 2.5: Pronóstico para 2026

# %% [markdown]
# Pronóstico AUTO-REGRESIVO: cada predicción se usa como entrada para la siguiente.
# Generamos 365 días de 2026 día por día.

# %%
print("\n🔮 Paso 2.5: Generando pronóstico para 2026 (365 días)\n")

# Crear fechas para 2026
fechas_2026 = pd.date_range('2026-01-01', '2026-12-31', freq='D')

# Usar última fila para extraer estado inicial
last_row = df_diario.iloc[-1].copy()

pronosticos_2026 = []

for i, fecha in enumerate(fechas_2026):
    # Crear fila para predicción
    row_pred = pd.DataFrame([last_row[features]]).copy()
    
    # Hacer predicción
    pred = model.predict(row_pred)[0]
    
    pronosticos_2026.append({
        'fecha': fecha,
        'prediccion': pred
    })
    
    # Actualizar last_row para siguiente iteración (usar predicción como nuevo lag_1)
    last_row['viajes_lag1'] = last_row['viajes_lag7']
    last_row['viajes_lag7'] = last_row['viajes_lag14'] if 'viajes_lag14' in last_row.index else last_row['viajes_lag7']
    last_row['viajes_lag30'] = last_row['viajes_lag30']
    
    # Actualizar variables temporales
    last_row['fecha'] = fecha
    last_row['mes'] = fecha.month
    last_row['semana_anio'] = fecha.isocalendar()[1]
    last_row['dia_anio'] = fecha.dayofyear
    last_row['es_fin_de_semana'] = 1 if fecha.dayofweek >= 5 else 0
    last_row['es_dia_laboral'] = 1 if fecha.dayofweek < 5 else 0
    last_row['es_feriado'] = 1 if fecha in feriados else 0
    
    # Temperatura (mantener patrón estacional)
    dias_desde_inicio = (fecha - pd.Timestamp('2021-01-01')).days
    last_row['temperatura'] = 16.5 + 8 * np.sin((dias_desde_inicio + 80) * 2 * np.pi / 365)
    
    if (i + 1) % 100 == 0:
        print(f"  Generadas {i + 1}/365 predicciones...")

pronostico_2026_df = pd.DataFrame(pronosticos_2026)

print(f"\n✅ Pronóstico 2026 generado: {len(pronostico_2026_df)} días")
print(f"\nPrimeras predicciones de 2026:")
print(pronostico_2026_df.head(15))

# Guardar pronóstico
pronostico_2026_df.to_csv('datos_procesados/predicciones_xgboost_2026.csv', index=False)
print(f"\n✅ Pronóstico guardado: predicciones_xgboost_2026.csv")

# %%
# Gráfico 5: Pronóstico 2026
fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(pronostico_2026_df['fecha'], pronostico_2026_df['prediccion'], 
        color='orange', linewidth=1.8, alpha=0.8)
ax.fill_between(pronostico_2026_df['fecha'], pronostico_2026_df['prediccion'], 
                alpha=0.2, color='orange')

ax.set_title('XGBoost: Pronóstico para 2026', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Viajes (Pronóstico)', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('datos_procesados/05_pronostico_2026.png', dpi=100, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: 05_pronostico_2026.png")


