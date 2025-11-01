import os
import warnings
import pickle
from queue import PriorityQueue

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.exceptions import InconsistentVersionWarning

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
import PIL.Image
import urllib.request

import streamlit as st
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from tensorflow.keras.models import load_model


# Path & rutas absolutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_modelo = os.path.join(BASE_DIR, "modelo", "red_neuronal.keras")
path_objetos = os.path.join(BASE_DIR, "modelo", "objetos.pkl")

# Cargar el modelo entrenado (y objetos auxiliares)
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
model = load_model(path_modelo)
scaler, label_encoder = pickle.load(open(path_objetos, "rb"))


# ----------------------------------------
# --- TABLERO & ALGORITMOS DE BUSQUEDA ---
# ----------------------------------------

def generar_tablero(n=5, dens_celdas_t3=0.2):
    tablero = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if np.random.rand() < dens_celdas_t3:
                tablero[i, j] = np.random.choice([2,3])
    tablero[0,0], tablero[n-1,n-1] = 1, 1
    return tablero

def reconstruir_camino(came_from, start, goal):
    camino = [goal]
    actual = goal
    while actual != start:
        actual = came_from.get(actual)
        if actual is None:
            return []
        camino.append(actual)
    camino.reverse()
    return camino

def heuristica(a, b): # Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def busqueda_generica(tablero, algoritmo="A*"):
    n = len(tablero)
    start, goal = (0, 0), (n-1, n-1)
    movimientos = [(0,1),(1,0),(-1,0),(0,-1)]
    
    open_list = PriorityQueue()
    open_list.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while not open_list.empty():
        _, actual = open_list.get()
        
        # si se llegó a la meta, devolver el camino completo
        if actual == goal:
            return reconstruir_camino(came_from, start, goal)
        
        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            if 0 <= nx < n and 0 <= ny < n:
                nuevo = (nx, ny)
                nuevo_costo = cost_so_far[actual] + tablero[nx, ny]
                
                # actualizar si se encontró un mejor costo
                if nuevo not in cost_so_far or nuevo_costo < cost_so_far[nuevo]:
                    cost_so_far[nuevo] = nuevo_costo
                    
                    if algoritmo == "UCS":
                        prioridad = nuevo_costo
                    elif algoritmo == "Greedy":
                        prioridad = heuristica(nuevo, goal)
                    elif algoritmo == "A*":
                        prioridad = nuevo_costo + heuristica(nuevo, goal)
                    else:
                        raise ValueError(f"Estrategia desconocida: {algoritmo}")
                    
                    # insertar el nodo en la cola con prioridad (heap queue)
                    open_list.put((prioridad, nuevo))
                    came_from[nuevo] = actual
    return []


# ----------------------------------------
# ---------------- MODELO ----------------
# ----------------------------------------

# Calcular las características estructurales y estadísticas de un tablero
def calcular_features(tablero):

    tamanio = tablero.shape[0]            # tamaño del tablero
    costo_prom = np.mean(tablero)         # costo de celda en promedio
    costo_moda = np.median(tablero)       # costo de celda más frecuente
    costo_var = np.var(tablero)           # varianza del costo por celda
    costo_min = np.min(tablero)           # costo mínimo
    costo_max = np.max(tablero)           # costo máximo
    dens_celda_t2 = np.mean(tablero == 2) # densidad de celdas con peso = 2
    dens_celda_t3 = np.mean(tablero == 3) # densidad de celdas con peso = 3
    dens_total = dens_celda_t2 + dens_celda_t3 # densidad pesada

    # entropía de costos
    valores, counts = np.unique(tablero, return_counts=True)
    entropia_costos = entropy(counts / counts.sum())

    # gradiente promedio
    difs = []
    for i in range(tamanio):
        for j in range(tamanio):
            for dx, dy in [(1,0), (0,1)]:  # solo derecha y abajo
                ni, nj = i + dx, j + dy
                if ni < tamanio and nj < tamanio:
                    difs.append(abs(tablero[i,j] - tablero[ni,nj]))
    gradiente_prom = np.mean(difs) if difs else 0

    # consolidar resultados
    return {
        "tamanio": tamanio,
        "costo_prom": costo_prom,
        "costo_moda": costo_moda,
        "costo_var": costo_var,
        "costo_min": costo_min,
        "costo_max": costo_max,
        "dens_celda_t2": dens_celda_t2,
        "dens_celda_t3": dens_celda_t3,
        "dens_total": dens_total,
        "entropia_costos": entropia_costos,
        "gradiente_prom": gradiente_prom
    }

# Predice el mejor algoritmo usando el modelo entrenado
def predecir_algoritmo(tablero, verbose=1):

    features = calcular_features(tablero)

    # extraer las features
    tamanio = features["tamanio"]
    costo_prom = features["costo_prom"]
    costo_moda = features["costo_moda"]
    costo_var = features["costo_var"]
    costo_min = features["costo_min"]
    costo_max = features["costo_max"]
    dens_celda_t2 = features["dens_celda_t2"]
    dens_celda_t3 = features["dens_celda_t3"]
    dens_total = features["dens_total"]
    entropia_costos = features["entropia_costos"]
    gradiente_prom = features["gradiente_prom"]

    # escalado de los features
    datos = [tamanio, costo_prom, costo_moda, costo_var, costo_min, costo_max, dens_celda_t2, dens_celda_t3, dens_total, entropia_costos, gradiente_prom]
    columnas = ['tamanio', 'costo_prom', 'costo_moda', 'costo_var', 'costo_min', 'costo_max', 'dens_celda_t2', 'dens_celda_t3', 'dens_total', 'entropia_costos', 'gradiente_prom']

    features_df = pd.DataFrame([datos], columns=columnas)
    features_scaled = scaler.transform(features_df)

    # hacer la prediccion
    predicciones = model.predict(features_scaled, verbose=verbose)
    pred_alg = np.argmax(predicciones, axis=1)[0]               # tomar el elemento com mayor probabilidad
    pred_alg = label_encoder.inverse_transform([pred_alg])[0]   # pasar codif. numérica a nombre del algoritmo

    return pred_alg, predicciones


# ----------------------------------------
# ----------- VISUALIZACIONES ------------
# ----------------------------------------

def mostrar_tablero(tablero, path=None):
    fig, ax = plt.subplots()
    ax.imshow(tablero, cmap='Blues', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    tamanio = tablero.shape[0]
    for i in range(tamanio):
        for j in range(tamanio):
            ax.text(j, i, int(tablero[i,j]), ha='center', va='center', color='black')

    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color='green', linewidth=2, alpha=0.6, marker='o', markersize=4, label='Mejor ruta')
        ax.legend(loc='upper right', prop={'size': 6})

    st.pyplot(fig)


@st.cache_data
def cargar_img_auto():
    img_url = "https://github.com/andres-garcia-alves/issd-dsia/raw/refs/heads/main/Proyecto-Final/miscelaneos/auto.png"

    with urllib.request.urlopen(img_url) as url:
        return np.array(PIL.Image.open(url))


def generar_tablero_animado(tablero, camino):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(tablero, cmap='Blues', origin='upper')
    ax.set_xticks([])
    ax.set_yticks([])

    # mostrar los costos en las celdas
    tamanio = tablero.shape[0]
    for i in range(tamanio):
        for j in range(tamanio):
            ax.text(j, i, int(tablero[i,j]), ha='center', va='center', color='black', fontsize=16)

    # dibujar camino y meta
    xs, ys = zip(*camino)
    ax.plot(ys, xs, color='green', linewidth=2, alpha=0.6, marker='o', markersize=4, label='')

    # cargar la imagen del auto
    img_data = cargar_img_auto()
    imagebox = OffsetImage(img_data, zoom=0.70)
    ab = AnnotationBbox(imagebox, camino[0][::-1], frameon=False)
    ax.add_artist(ab)

    # generar interpolaciones (para un movimiento fluido)
    posiciones_interp = []
    pasos_intermedios = 10
    for (x1, y1), (x2, y2) in zip(camino[:-1], camino[1:]):
        for t in np.linspace(0, 1, pasos_intermedios, endpoint=False):
            xi = x1 + (x2 - x1) * t
            yi = y1 + (y2 - y1) * t
            posiciones_interp.append((xi, yi))
    posiciones_interp.append(camino[-1])  # último punto exacto

    # callback invocado por matplotlib por cada frame
    def update(frame):
        # borrar el auto anterior
        for artist in ax.artists: artist.remove()

        # dibujar el auto en la nueva posición
        x, y = posiciones_interp[frame]
        ab_new = AnnotationBbox(imagebox, (y, x), frameon=False)
        ax.add_artist(ab_new)

        return ab_new,

    # crear y guardar el gif de la animación
    fps = 20
    anim = FuncAnimation(fig, update, frames=len(posiciones_interp), interval = 1000/fps, blit=False, repeat=False)

    archivo_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
    anim.save(archivo_gif, writer='pillow', fps=fps)
    plt.close(fig)

    return archivo_gif


# ----------------------------------------
# -------------- STREAMLIT ---------------
# ----------------------------------------

# inicializar keys en session_state
if 'tablero' not in st.session_state:   st.session_state['tablero'] = None
if 'camino' not in st.session_state:    st.session_state['camino'] = None
if 'alg_pred' not in st.session_state:  st.session_state['alg_pred'] = None
if 'gif_path' not in st.session_state:  st.session_state['gif_path'] = None

# header
st.title("🎯 Selector Inteligente de Algoritmos de Búsqueda")
st.markdown("Demo basada en IA entrenada con TensorFlow + algoritmos de búsqueda (BFS, A*, UCS, Greedy).")
st.write("")
st.write("")

# parámetros para el tablero
tamanio = st.slider("Tamaño del tablero", 4, 8, 5)
dens_celdas_t3 = st.slider("Densidad de celdas de costo alto (máx. 60%)", 0.0, 0.6, 0.2)
st.write("")
st.write("")

generar_btn = st.button("🧱 Generar tablero aleatorio")

if generar_btn:
    st.session_state['tablero'] = generar_tablero(tamanio, dens_celdas_t3=dens_celdas_t3)
    st.session_state['alg_pred'], _ = predecir_algoritmo(st.session_state['tablero'], verbose=0)
    st.session_state['camino'] = busqueda_generica(st.session_state['tablero'], algoritmo=st.session_state['alg_pred'])
    st.session_state['gif_path'] = None

# si hay un tablero en session_state, mostrarlo
if st.session_state['tablero'] is not None:
    tablero = st.session_state['tablero']
    alg_pred = st.session_state['alg_pred']
    camino = st.session_state['camino']

    st.write("🔹 Tablero generado:")
    mostrar_tablero(tablero)
    st.write("")
    st.write("")

    if camino:
        st.write("🗺️ La mejor ruta:")
        mostrar_tablero(tablero, camino)
        st.write(f"🧠 IA - algoritmo en uso: **{alg_pred}**")
        st.write("")
        st.write("")

        generar_animacion = st.button("🎞️ Generar animación", key="anim_button")
        if generar_animacion:
            # si el gif aún no existe, generarlo y guardarlo en la sesión
            with st.spinner("Generando animación, esto puede tardar un rato ..."):
                path_tmp = generar_tablero_animado(tablero, camino)
                st.session_state['gif_path'] = path_tmp

        # si ya existe un gif generado, mostrarlo
        if st.session_state['gif_path']:
            st.image(st.session_state['gif_path'], caption="Animación del recorrido", width='stretch')
            # st.write("✅ Animación generada con éxito.")

    else:
        st.warning("⚠️ No se encontró un camino válido para este tablero.")
