
# CONSIGNA:
# Realiza un programa en Python que busque palabras en un texto de varias páginas de manera eficiente.
# El programa debe permitir al usuario ingresar una o varias palabras y el programa debe devolver
# en qué páginas se encuentran y el número de ocurrencias.
# El programa debe ser capaz de manejar un archivo de texto.


# NOTAS:
# - Un archivo de texto no soporta el concepto de "páginas" como en un documento de Word o PDF.
#   Para simular "páginas", utilizo dobles saltos de línea como separador.

# - El archivo de texto debe estar en el mismo directorio que este script, o proporcionar la ruta completa.
#   Si no se suministra una ruta, se usará "Don Quijote de la Mancha.txt" como archivo predeterminado.

# - La estructura interna del índice (simulando nodos) se implementa de la siguiente manera:
#     "palabra" -> { "nro-página": cant-de-ocurrencias, "nro-página": cant-de-ocurrencias, ... }
#
#   Ejemplo:
#     "Esto" -> { 1: 1, 5: 2 }          # en pág 1 aparece 1 vez, en pág 5 aparece 2 veces
#     "Es"   -> { 1: 8, 2: 3, 5: 1 }    # en pág 1 aparece 8 veces, en pág 2 aparece 3 veces, en pág 5 aparece 1 vez
#     "Un"   -> { 4: 1, 2: 2, 4: 1 }    # en pág 4 aparece 1 veces, en pág 2 aparece 2 veces, en pág 4 aparece 1 vez
#     ...

import math
import time

def welcome():
  print("\n--- Buscador de Palabras en Texto")
  print("--- ISSD-DSIA - Clase 04\n")


# Leer el archivo y retornar su texto
def read_file():
  filepath = input("Ingrese la ruta al archivo de texto ('Enter' = default): ").strip()
  if not filepath:
    #filepath = "Resources\\Desarrollo.txt"
    filepath = "Resources\\Don Quijote de la Mancha.txt"

  with open(filepath, 'r', encoding='utf-8') as file:
    return file.read()


# Dividir el texto en "páginas" (asumiendo que cada "página" esté separada por 2 saltos de línea)
def parse_file(raw_text):
  print("Pre-procesando el archivo, por favor espere ...")
  words_dict = {}

  pages = raw_text.lower().split('\n\n')
  for idx, page in enumerate(pages):
    for word in page.split():
      append_word(words_dict, idx, word)

  return words_dict


# Agregar una palabra al índice
def append_word(index, page_number, word):
  if word not in index:               index[word] = {}
  if page_number not in index[word]:  index[word][page_number] = 0
  index[word][page_number] += 1


# Mostrar el índice completo (modo debug)
def show_words_dict(words_dict):
  print("\nIndice de palabras (modo debug):")
  print("<Palabra>    -> <Página>:<Ocurrencias>")
  print("---------------------------------------")

  for word in sorted(words_dict.keys()):
    print(f'{ word.ljust(12) } -> { words_dict[word] }')


# Separar el input de usuario
def get_user_input():
  print("\nIngrese las palabras a buscar")
  print("(separadas por coma, 'Enter' = finalizar):")
  user_input = input("") # \nIngrese las palabras a buscar (Enter para finalizar): 
  if not user_input:
    return None
  
  words = [word.strip().lower() for word in user_input.split(',')]
  return words


# Buscar palabras y mostrar sus ocurrencias por página
# Notas:
#   Esta función utiliza búsqueda directa en el diccionario: 'entry = words_dict[word]'
#   (usé un diccionario para evitar implementar la estructura de grafos a mano)
#   Obvio que no sirve para ilustrar la navegación en nodos de estructuras tipo árbol o grafo
def find_words_dict(words_dict, words):
  node_count = -1   # nodos visitados
  timespan = 0      # tiempo de búsqueda

  start = time.perf_counter()

  for word in words:
    print(f"\n- Palabra buscada: '{ word }'")

    if word in words_dict:
      entry = words_dict[word]
      for page, times in entry.items():
        print(f"  Página #{ page+1 }, aparece { times } veces.")
    else:
      print(f"  No se encontró en el texto.")

  timespan = time.perf_counter() - start
  timespan *= 1000  # convertir a ms

  return node_count, timespan


# Buscar palabras y mostrar sus ocurrencias por página (Método BFS)
# Notas:
#   Esta función simula una búsqueda en anchura (BFS).
#   Toma al diccionario como si fuera un árbol de nivel 1 y va recorriendo TODOS los "nodos" (entries) disponibles
def find_words_with_bfs(words_dict, words):
  visited_count = 0    # nodos visitados
  timespan = 0      # tiempo de búsqueda
  print(f"\nBuscando con BFS ...")

  start = time.perf_counter()

  for word in words:
    print(f"\n- Palabra buscada: '{ word }'")
    found = False

    for key, values in words_dict.items():
      visited_count += 1

      if key == word:
        for page, times in values.items():
          print(f"  Página #{ page+1 }, aparece { times } veces.")

        found = True
        break         # detener la búsqueda

    if found == False:
      print(f"  No se encontró en el texto.")

  timespan = time.perf_counter() - start
  timespan *= 1000    # convertir a ms

  return visited_count, timespan


# Buscar palabras y mostrar sus ocurrencias por página (Método con Heurística)
# Notas:
#   Esta función simula una búsqueda con Heurística.
#   Toma al diccionario como si fuera un árbol de nivel 1 y va eligiendo nodos a dedo según corresponda
#   Para la Heurística implementé un símil quick-sort (ordenamiento rápido)
def find_words_with_heuristics(words_dict, words):
  visited_count = 0    # nodos visitados
  timespan = 0      # tiempo de búsqueda
  next_node_idx = 0      # índice del nodo a visitar
  print(f"\nBuscando con Heurística ...")

  # Inicialmente elegí un diccionario como estructura de datos ...
  # En los diccionarios se accede exclusivamente por clave: dict['algo'], y no por índice: dict[0]
  # El problema es que mi heurística va a retornar el índice (número) de nodo a visitar así que aquí convierto el dict a lista de tuplas
  # Dejo la conversión fuera de la métrica de tiempo, para que sea una comparación justa contra el método BFS
  words_list = sorted(list(words_dict.items()))
  words_list_len = len(words_list)

  start = time.perf_counter()

  for word in words:
    print(f"\n- Palabra buscada: '{ word }'")

    unvisited_len = len(words_list)                 # cant de nodos pendientes de visitar
    next_node_idx = math.floor(unvisited_len / 2)   # nodo incial a la mitad de la lista

    while next_node_idx != -1:
      visited_count += 1
      unvisited_len = math.ceil(unvisited_len / 2)

      node = words_list[next_node_idx]               # el nodo actual
      next_node_idx = heuristics(node[0], word, next_node_idx, unvisited_len, words_list_len)

      if next_node_idx == -1: # encontrado
        for page, times in node[1].items():
          print(f"  Página #{ page+1 }, aparece { times } veces.")
        break

      if unvisited_len <= 1: # no existe
        print(f"  No se encontró en el texto.")
        break

  timespan = time.perf_counter() - start
  timespan *= 1000    # convertir a ms

  return visited_count, timespan


# Función de Heurística, retorna el siguiente nodo a visitar
# Retorna -1 si se encontró el nodo buscado
def heuristics(current_word, target_word, node_idx, unvisited_count, words_list_len):
  next_idx = 0
  step = math.ceil(unvisited_count / 2)

  if current_word == target_word:
    return -1   # encontrado
  
  if current_word > target_word:
    next_idx = node_idx - step
    return next_idx if next_idx >= 0 else 0
  else:
    next_idx = node_idx + step
    return next_idx if next_idx <= words_list_len-1 else words_list_len-1


# Mostar métricas de búsqueda
def show_metrics(bfs_count, bfs_timespan, heur_count, heur_timespan):
  print("\nMétricas de la búsqueda:")
  print(f"- BFS:        nodos visitados { bfs_count }, Tiempo: { round(bfs_timespan, 2) } ms")
  print(f"- Heurística: nodos visitados { heur_count }, Tiempo: { round(heur_timespan, 2) } ms")


# Bloque principal
def main():

  welcome()

  raw_text = read_file()
  words_dict = parse_file(raw_text)
  # show_words_dict(words_dict)    # para debug

  words = get_user_input()
  while words:

    #dict_count, dict_timespan = find_words_dict(words_dict, words)
    bfs_count, bfs_timespan = find_words_with_bfs(words_dict, words)
    heur_count, heur_timespan = find_words_with_heuristics(words_dict, words)

    show_metrics(bfs_count, bfs_timespan, heur_count, heur_timespan)
    words = get_user_input()

main()
