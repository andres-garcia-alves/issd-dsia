import os
import pickle
from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# path & rutas absolutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_modelo = os.path.join(BASE_DIR, "modelo", "red_neuronal.keras")
path_objetos = os.path.join(BASE_DIR, "modelo", "objetos.pkl")

# Cargar el modelo entrenado (y objetos auxiliares)
model = load_model(path_modelo)
scaler, label_encoder = pickle.load(open(path_objetos, "rb"))


# Funciones del proyecto
def generar_tablero(n=5, p_cost3=0.2):
    tablero = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if np.random.rand() < p_cost3:
                tablero[i, j] = np.random.choice([2,3])
    tablero[0,0], tablero[n-1,n-1] = 1, 1
    return tablero


def mostrar_tablero(tablero, path=None):
    fig, ax = plt.subplots()
    ax.imshow(tablero, cmap="summer", interpolation="nearest") # cmap="coolwarm"

    n = tablero.shape[0]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(tablero[i,j]), ha="center", va="center", color="black")

    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color="blue", linewidth=3, marker="o", markersize=5, label="Mejor ruta") # color="lime"
        ax.legend(loc="upper right")

    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)


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


def buscar_camino(tablero, modo="A*"):
    n = len(tablero)
    start, goal = (0, 0), (n - 1, n - 1)
    movimientos = [(0,1),(1,0),(-1,0),(0,-1)]
    
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    
    open_list = PriorityQueue()
    open_list.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while not open_list.empty():
        _, actual = open_list.get()
        
        if actual == goal:
            return reconstruir_camino(came_from, start, goal)
        
        for dx, dy in movimientos:
            nx, ny = actual[0] + dx, actual[1] + dy
            if 0 <= nx < n and 0 <= ny < n:
                nuevo = (nx, ny)
                nuevo_costo = cost_so_far[actual] + tablero[nx, ny]
                
                if nuevo not in cost_so_far or nuevo_costo < cost_so_far[nuevo]:
                    cost_so_far[nuevo] = nuevo_costo
                    
                    if modo == "A*":
                        prioridad = nuevo_costo + heuristic(nuevo, goal)
                    elif modo == "UCS":
                        prioridad = nuevo_costo
                    elif modo == "Greedy":
                        prioridad = heuristic(nuevo, goal)
                    else:
                        prioridad = nuevo_costo  # fallback
                    
                    open_list.put((prioridad, nuevo))
                    came_from[nuevo] = actual
    return []


# Interfaz
st.title("🎯 Selector Inteligente de Algoritmos de Búsqueda")
st.markdown("Mini-demo basada en IA entrenada con TensorFlow + heurísticas de búsqueda (BFS, A*, UCS, Greedy).")

# Parámetros del tablero
tamano = st.slider("Tamaño del tablero", 4, 8, 5)
densidad_cost3 = st.slider("Densidad de celdas con costo alto (máx. 60%)", 0.0, 0.6, 0.2)
generar_btn = st.button("Generar tablero aleatorio")

if generar_btn:
    tablero = generar_tablero(tamano, p_cost3=densidad_cost3)
    st.write("🔹 Tablero generado:")
    mostrar_tablero(tablero)

    # Calcular features del tablero
    costo_prom = np.mean(tablero)
    var_costo = np.var(tablero)
    densidad_cost3_real = np.mean(tablero == 3)

    X_input = np.array([[tamano, costo_prom, var_costo, densidad_cost3_real]])
    X_scaled = scaler.transform(X_input)
    
    pred = np.argmax(model.predict(X_scaled), axis=1)
    alg_predicho = label_encoder.inverse_transform(pred)[0]
    
    st.success(f"🧠 Algoritmo recomendado (IA): **{alg_predicho}**")

    # Calcular y mostrar ruta óptima del algoritmo sugerido
    path = buscar_camino(tablero, modo=alg_predicho)
    if path:
        st.write("🗺️ Mejor ruta encontrada:")
        mostrar_tablero(tablero, path)
    else:
        st.warning("⚠️ No se encontró camino válido en este tablero.")
