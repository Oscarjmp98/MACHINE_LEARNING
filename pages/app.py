# ============================================================
# Aplicación Web - Examen Final Machine Learning Aplicado
# Streamlit: Predicción Telco (Logística y KNN) + Clustering Credit Card (K-Means)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

# Cargar estilos para mejorar la presentación visual
with open("assets/estilo.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Cargar modelos y recursos necesarios para las predicciones
# ------------------------------------------------------------
@st.cache_resource
def cargar_modelos_y_recursos():
    modelos = {}
    try:
        # Modelos supervisados y su escalador (normalización de datos)
        modelos["logistica"] = joblib.load("modelos/modelo_logistica.pkl")
        modelos["knn"] = joblib.load("modelos/modelo_knn.pkl")
        scaler = joblib.load("modelos/scaler.pkl")
    except Exception as e:
        st.error(f"No se pudieron cargar modelos o escalador: {e}")
        scaler = None
    
    # Modelo no supervisado (K-Means) para tarjetas de crédito
    try:
        modelos["kmeans"] = joblib.load("modelos/modelo_kmeans.pkl")
    except Exception:
        pass

    # Columnas usadas en el entrenamiento del modelo Telco (tras get_dummies)
    feature_columns = None
    try:
        with open("modelos/feature_columns.json", "r") as f:
            feature_columns = json.load(f)
    except Exception:
        st.warning("No se encontró 'feature_columns.json'. Se usarán columnas estimadas según el formulario.")
        feature_columns = None

    # Descripciones de perfiles por cluster para tarjetas de crédito
    cluster_profiles = {}
    try:
        with open("modelos/cluster_profiles.json", "r") as f:
            cluster_data = json.load(f)
            for key, value in cluster_data.items():
                if key.startswith("Cluster "):
                    cluster_num = key.replace("Cluster ", "")
                    cluster_profiles[cluster_num] = value
                else:
                    cluster_profiles[key] = value
    except Exception:
        # Si no hay archivo de perfiles, se cargan descripciones básicas
        cluster_profiles = {
            "0": {"nombre": "Cluster 0", "descripcion": "Clientes con bajo balance y compras ocasionales.", "recomendacion": "N/A"},
            "1": {"nombre": "Cluster 1", "descripcion": "Clientes con balance medio y uso moderado de crédito.", "recomendacion": "N/A"},
            "2": {"nombre": "Cluster 2", "descripcion": "Clientes con alto balance y alto uso de adelantos.", "recomendacion": "N/A"},
        }

    return modelos, scaler, feature_columns, cluster_profiles

modelos, scaler, feature_columns, cluster_profiles = cargar_modelos_y_recursos()

# ------------------------------------------------------------
# Funciones de apoyo para preparar datos de Telco
# ------------------------------------------------------------

# Estas listas son de referencia si no hay archivo de columnas del entrenamiento
DEFAULT_TELCO_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
DEFAULT_TELCO_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

def construir_dataframe_telco(inputs_dict):
    """
    Convierte los datos del formulario en un DataFrame de una sola fila.
    Esto facilita aplicar las transformaciones y pasar los datos al modelo.
    """
    df = pd.DataFrame([inputs_dict])
    return df

def preparar_telco_para_modelo(df, feature_columns, scaler):
    """
    Prepara los datos para que el modelo los use correctamente:
    1) Convierte variables categóricas a columnas numéricas.
    2) Alinea las columnas con las usadas en el entrenamiento.
    3) Ordena las columnas en el mismo orden del entrenamiento.
    4) Aplica el escalador para dejar todo en una escala comparable.
    """
    # Convertir categorías a columnas
    df_dum = pd.get_dummies(df, drop_first=True)

    # Si no hay referencia de columnas del entrenamiento, se usa lo que surja del formulario
    if feature_columns is None:
        feature_columns = list(df_dum.columns)
    
    # Asegurar que existan todas las columnas esperadas
    for col in feature_columns:
        if col not in df_dum.columns:
            df_dum[col] = 0

    # Mantener solo las columnas esperadas
    columns_to_keep = [col for col in feature_columns if col in df_dum.columns]
    df_dum = df_dum[columns_to_keep]

    # Por seguridad, volver a agregar cualquier faltante con valor 0
    for col in feature_columns:
        if col not in df_dum.columns:
            df_dum[col] = 0
    
    # Ordenar columnas como en el entrenamiento
    df_dum = df_dum[feature_columns]

    # Aplicar escalado (normalización). Si no hay escalador, se usan los valores tal cual.
    if scaler is None:
        st.warning("No se encontró escalador. El KNN puede no funcionar bien sin normalización.")
        X_scaled = df_dum.values.astype(float)
    else:
        try:
            # Pasar el DataFrame con nombres de columnas ayuda a evitar avisos y errores
            X_scaled = scaler.transform(df_dum.astype(float))
        except Exception as e:
            st.error(f"Error al aplicar escalador: {str(e)}")
            st.write(f"Forma de los datos: {df_dum.shape}, Columnas esperadas: {len(feature_columns)}")
            raise

    return X_scaled, feature_columns

# ------------------------------------------------------------
# Configuración de la interfaz
# ------------------------------------------------------------
st.set_page_config(page_title="ML Aplicado - Telco y Credit Card", layout="centered")

st.title("Aplicación Web: Telco Churn (Logística y KNN) + Credit Card Clustering (K-Means)")
st.write("Esta aplicación permite probar los modelos entrenados usando formularios simples.")

# Crear dos pestañas para separar los dos análisis
tab_telco, tab_credit = st.tabs(["Telco: Churn", "Credit Card: K-Means"])

# ------------------------------------------------------------
# Pestaña Telco: predicción con Regresión Logística y KNN
# ------------------------------------------------------------
with tab_telco:
    st.header("Predicción de Churn (Telco)")

    # Formulario para ingresar características del cliente
    with st.form("form_telco"):
        # Variables numéricas
        senior_citizen = st.selectbox("Ciudadano de la tercera edad (SeniorCitizen)", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        tenure = st.number_input("Antigüedad del cliente (tenure)", min_value=0, max_value=100, value=12)
        monthly = st.number_input("Cargos mensuales (MonthlyCharges)", min_value=0.0, max_value=1000.0, value=70.0)
        total = st.number_input("Cargos totales (TotalCharges)", min_value=0.0, max_value=100000.0, value=840.0)

        # Variables categóricas
        gender = st.selectbox("Género", ["Female", "Male"])
        partner = st.selectbox("Pareja", ["Yes", "No"])
        dependents = st.selectbox("Dependientes", ["Yes", "No"])
        phone = st.selectbox("Servicio de teléfono", ["Yes", "No"])
        multiple = st.selectbox("Líneas múltiples", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
        onsec = st.selectbox("Seguridad en línea", ["No", "Yes", "No internet service"])
        onback = st.selectbox("Backup en línea", ["No", "Yes", "No internet service"])
        devprot = st.selectbox("Protección de dispositivos", ["No", "Yes", "No internet service"])
        techsup = st.selectbox("Soporte técnico", ["No", "Yes", "No internet service"])
        stv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        smov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Factura sin papel", ["Yes", "No"])
        paymethod = st.selectbox("Método de pago", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        submitted = st.form_submit_button("Predecir")

    if submitted:
        try:
            # Construir los datos con lo ingresado
            telco_inputs = {
                "SeniorCitizen": senior_citizen,
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "gender": gender,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": onsec,
                "OnlineBackup": onback,
                "DeviceProtection": devprot,
                "TechSupport": techsup,
                "StreamingTV": stv,
                "StreamingMovies": smov,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": paymethod
            }

            df_telco = construir_dataframe_telco(telco_inputs)
            X_scaled, cols_used = preparar_telco_para_modelo(df_telco, feature_columns, scaler)

            # Verificación de que el número de columnas coincide con lo esperado
            if feature_columns and X_scaled.shape[1] != len(feature_columns):
                st.error(f"El número de características ({X_scaled.shape[1]}) no coincide con las esperadas ({len(feature_columns)}).")
            else:
                # Slider para ajustar el umbral de clasificación (probabilidad a partir de la cual se califica como Sí)
                umbral = st.slider("Umbral de clasificación (0 a 1)", 0.0, 1.0, 0.5, help="Baje el umbral para que el modelo sea más sensible y clasifique más casos como Sí.")

                # Predicción con Regresión Logística
                if "logistica" in modelos and modelos["logistica"] is not None:
                    try:
                        prob_log = modelos["logistica"].predict_proba(X_scaled)[0, 1]
                        pred_log = "Yes" if prob_log >= umbral else "No"
                        st.subheader("Regresión Logística")
                        st.write(f"Probabilidad de churn: {prob_log:.3f}")
                        st.write(f"Clasificación (umbral={umbral:.2f}): {pred_log}")
                    except Exception as e:
                        st.error(f"Error en predicción de Regresión Logística: {str(e)}")
                        st.exception(e)
                else:
                    st.error("Modelo de Regresión Logística no disponible.")

                # Predicción con KNN
                if "knn" in modelos and modelos["knn"] is not None:
                    try:
                        # Si el KNN fue entrenado para dar probabilidades, se usa el umbral.
                        # Si no, se usa la clase directa.
                        if hasattr(modelos["knn"], "predict_proba"):
                            prob_knn = modelos["knn"].predict_proba(X_scaled)[0, 1]
                            pred_knn_str = "Yes" if prob_knn >= umbral else "No"
                            st.subheader("K-Nearest Neighbors (KNN)")
                            st.write(f"Probabilidad de churn: {prob_knn:.3f}")
                            st.write(f"Clasificación (umbral={umbral:.2f}): {pred_knn_str}")
                        else:
                            pred_knn = modelos["knn"].predict(X_scaled)[0]
                            pred_knn_str = "Yes" if int(pred_knn) == 1 else "No"
                            st.subheader("K-Nearest Neighbors (KNN)")
                            st.write(f"Clasificación: {pred_knn_str}")
                            st.caption("Nota: este KNN no entrega probabilidades. Para usar umbral, el modelo debe reentrenarse con salida probabilística.")
                    except Exception as e:
                        st.error(f"Error en predicción de KNN: {str(e)}")
                        st.exception(e)
                else:
                    st.error("Modelo KNN no disponible.")
        except Exception as e:
            st.error(f"Error al procesar los datos: {str(e)}")
            st.exception(e)

# ------------------------------------------------------------
# Pestaña Credit Card: asignación de cluster con K-Means
# ------------------------------------------------------------
with tab_credit:
    st.header("Asignación de Cluster (Tarjetas de Crédito)")

    # Formulario básico con las variables principales
    st.write("Ingrese los datos del cliente para asignar un cluster:")
    with st.form("form_credit"):
        balance = st.number_input("BALANCE (Balance)", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0)
        purchases = st.number_input("PURCHASES (Compras totales)", min_value=0.0, max_value=1000000.0, value=500.0, step=100.0)
        cash_adv = st.number_input("CASH_ADVANCE (Adelantos en efectivo)", min_value=0.0, max_value=1000000.0, value=200.0, step=100.0)
        credit_lim = st.number_input("CREDIT_LIMIT (Límite de crédito)", min_value=0.0, max_value=1000000.0, value=3000.0, step=100.0)
        payments = st.number_input("PAYMENTS (Pagos)", min_value=0.0, max_value=1000000.0, value=600.0, step=100.0)
        min_pay = st.number_input("MINIMUM_PAYMENTS (Pagos mínimos)", min_value=0.0, max_value=1000000.0, value=50.0, step=10.0)
        tenure = st.number_input("TENURE (Antigüedad en meses)", min_value=0, max_value=100, value=12)

        submitted_c = st.form_submit_button("Asignar cluster")

    if submitted_c:
        try:
            # Columnas esperadas por el modelo (guardadas desde el entrenamiento)
            credit_feature_columns = None
            try:
                with open("modelos/feature_columns_credit.json", "r") as f:
                    credit_feature_columns = json.load(f)
            except Exception as e:
                st.warning(f"No se encontró 'feature_columns_credit.json': {e}")
            
            # Construir los datos base con las variables ingresadas
            credit_inputs = {
                "BALANCE": balance,
                "PURCHASES": purchases,
                "CASH_ADVANCE": cash_adv,
                "CREDIT_LIMIT": credit_lim,
                "PAYMENTS": payments,
                "MINIMUM_PAYMENTS": min_pay,
                "TENURE": tenure
            }
            
            # Si hay lista de columnas del entrenamiento, completar las faltantes con valores razonables
            if credit_feature_columns:
                df_credit = pd.DataFrame([credit_inputs])
                
                # Valores calculados para columnas derivadas, basados en lo ingresado
                total_purchases = purchases
                balance_freq_default = 0.5 if balance > 0 else 0.0
                purchases_freq_default = 0.5 if purchases > 0 else 0.0
                oneoff_purchases_default = total_purchases * 0.3 if total_purchases > 0 else 0.0
                installments_purchases_default = total_purchases * 0.7 if total_purchases > 0 else 0.0
                oneoff_freq_default = 0.3 if oneoff_purchases_default > 0 else 0.0
                installments_freq_default = 0.4 if installments_purchases_default > 0 else 0.0
                cash_adv_freq_default = 0.2 if cash_adv > 0 else 0.0
                cash_adv_trx_default = int(cash_adv / 500) if cash_adv > 0 else 0
                purchases_trx_default = int(purchases / 100) if purchases > 0 else 0
                prc_full_payment_default = 0.3 if payments > 0 else 0.0
                
                default_values = {
                    "BALANCE_FREQUENCY": balance_freq_default,
                    "ONEOFF_PURCHASES": oneoff_purchases_default,
                    "INSTALLMENTS_PURCHASES": installments_purchases_default,
                    "PURCHASES_FREQUENCY": purchases_freq_default,
                    "ONEOFF_PURCHASES_FREQUENCY": oneoff_freq_default,
                    "PURCHASES_INSTALLMENTS_FREQUENCY": installments_freq_default,
                    "CASH_ADVANCE_FREQUENCY": cash_adv_freq_default,
                    "CASH_ADVANCE_TRX": cash_adv_trx_default,
                    "PURCHASES_TRX": purchases_trx_default,
                    "PRC_FULL_PAYMENT": prc_full_payment_default
                }
                
                # Agregar columnas faltantes con valores por defecto
                for col in credit_feature_columns:
                    if col not in df_credit.columns:
                        df_credit[col] = default_values.get(col, 0.0)
                
                # Ordenar columnas como en el entrenamiento
                df_credit = df_credit[credit_feature_columns]
            else:
                df_credit = pd.DataFrame([credit_inputs])

            # Cargar modelo K-Means si no está en memoria
            kmeans = modelos.get("kmeans")
            if kmeans is None:
                try:
                    kmeans = joblib.load("modelos/modelo_kmeans.pkl")
                    modelos["kmeans"] = kmeans
                except Exception as e:
                    st.error(f"No se pudo cargar el modelo K-Means: {e}")
                    st.exception(e)

            if kmeans is not None:
                try:
                    # Intentar cargar el escalador usado en el entrenamiento de K-Means
                    credit_scaler = None
                    try:
                        credit_scaler = joblib.load("modelos/scaler_credit.pkl")
                    except Exception:
                        pass
                    
                    # Preparar datos para la predicción
                    X_credit = df_credit.values.astype(float)
                    
                    # Aplicar el mismo tipo de normalización que en el entrenamiento
                    if credit_scaler is not None:
                        try:
                            if hasattr(credit_scaler, 'n_features_in_') and credit_scaler.n_features_in_ == X_credit.shape[1]:
                                X_credit = credit_scaler.transform(X_credit)
                                st.info("Se aplicó normalización a los datos antes de la predicción.")
                            else:
                                # Si el escalador no coincide en columnas, se aplica una normalización básica
                                X_credit = (X_credit - X_credit.mean(axis=0)) / (X_credit.std(axis=0) + 1e-8)
                        except Exception:
                            X_credit = (X_credit - X_credit.mean(axis=0)) / (X_credit.std(axis=0) + 1e-8)
                    else:
                        # Si no hay escalador guardado, se aplica una normalización básica
                        X_credit = (X_credit - X_credit.mean(axis=0)) / (X_credit.std(axis=0) + 1e-8)
                    
                    # Mostrar detalles útiles para revisar resultados
                    try:
                        centroides = kmeans.cluster_centers_
                        distancias = []
                        for i, centroide in enumerate(centroides):
                            dist = np.linalg.norm(X_credit[0] - centroide)
                            distancias.append((i, dist))
                        distancias.sort(key=lambda x: x[1])

                        with st.expander("Ver detalles de los datos", expanded=False):
                            st.write("Valores ingresados en el formulario:")
                            for key, value in credit_inputs.items():
                                st.write(f"- {key}: {value}")
                            st.write(f"Total de columnas esperadas: {len(credit_feature_columns) if credit_feature_columns else 'N/A'}")
                            calculated_cols = [col for col in df_credit.columns if col not in credit_inputs]
                            st.write("Ejemplos de columnas calculadas automáticamente:")
                            for col in calculated_cols[:5]:
                                st.write(f"- {col}: {df_credit[col].iloc[0]:.2f}")
                            if len(calculated_cols) > 5:
                                st.write(f"... y {len(calculated_cols) - 5} columnas más")
                            st.write("Primeros valores del arreglo (después de normalización):")
                            st.write(X_credit[0][:10])
                            st.write("Distancias a los centroides de los clusters:")
                            for cluster_num, dist in distancias:
                                marcador = "Más cercano" if cluster_num == distancias[0][0] else ""
                                st.write(f"- Cluster {cluster_num}: {dist:.2f} {marcador}")
                    except Exception:
                        pass
                    
                    # Predicción de cluster
                    cluster_id = int(kmeans.predict(X_credit)[0])
                    st.subheader("Resultado de K-Means")
                    st.write(f"Número de cluster asignado: {cluster_id}")

                    # Descripción del perfil del cluster
                    cluster_info = cluster_profiles.get(str(cluster_id))
                    if cluster_info:
                        if isinstance(cluster_info, dict):
                            st.write(f"Descripción del perfil: {cluster_info.get('descripcion', 'N/A')}")
                        else:
                            st.write(f"Descripción del perfil: {cluster_info}")
                    else:
                        st.warning("No hay descripción disponible para este cluster.")
                        
                except Exception as e:
                    st.error(f"Error al predecir cluster: {str(e)}")
                    st.write(f"Forma de los datos: {df_credit.shape}")
                    st.write(f"Columnas: {list(df_credit.columns)}")
                    st.write(f"Valores: {df_credit.values}")
                    st.exception(e)
        except Exception as e:
            st.error(f"Error al procesar los datos de tarjeta de crédito: {str(e)}")
            st.exception(e)

# ------------------------------------------------------------
# Nota final en la interfaz
# ------------------------------------------------------------
st.info("Asegúrese de que las columnas del formulario estén alineadas con las del entrenamiento y que el escalador sea el mismo usado para entrenar.")
