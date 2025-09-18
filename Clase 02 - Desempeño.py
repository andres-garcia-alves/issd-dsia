
# CONSIGNA:
# Realiza un programa en Python que busque palabras en un texto de varias páginas de manera eficiente.
# El programa debe permitir al usuario ingresar una o varias palabras y el programa debe devolver en qué páginas se encuentran y el número de ocurrencias. El programa debe ser capaz de manejar un archivo de texto.


# NOTAS:
# Un archivo de texto no soporta el concepto de "páginas" como en un documento de Word o PDF.
# Para simular "páginas", utilizo dobles saltos de línea como separador.
# El archivo de texto debe estar en el mismo directorio que este script o proporcionar la ruta completa.
# Si no se suministra una ruta, se usará "Don Quijote de la Mancha.txt" como archivo predeterminado.


import re


# Leer el archivo y dividirlo en "páginas"
# Asumiendo que cada "página" esté separada por 2 saltos de línea
def leer_archivo(archivo):
  with open(archivo, 'r', encoding='utf-8') as file:
    texto = file.read()
    paginas = texto.split('\n\n')
  return paginas


# Buscar palabras y contar sus ocurrencias por página
def buscar_palabras(paginas, palabras):
  resultados = {}

  for i, pagina in enumerate(paginas):
    conteo = {}

    for palabra in palabras:
      # expresión regular para contar las ocurrencias de la palabra (sin distinguir entre mayúsculas/minúsculas)
      ocurrencias = len(re.findall(r'\b' + re.escape(palabra) + r'\b', pagina, re.IGNORECASE))
      if ocurrencias > 0:
        conteo[palabra] = ocurrencias

    if conteo:
      resultados[f'Página {i + 1}'] = conteo

  return resultados


# Mostrar los resultados
def mostrar_resultados(resultados):
  if resultados:
    for pagina, conteos in resultados.items():
      print(f'\n{pagina}:')
      for palabra, conteo in conteos.items():
        print(f'La palabra "{ palabra }" aparece { conteo } veces.')
  else:
    print("No se encontraron las palabras en el texto.")


# Bloque principal
def main():
  archivo = input("Ruta del archivo de texto: ").strip()
  if not archivo:
    archivo = "Resources\Don Quijote de la Mancha.txt"

  paginas = leer_archivo(archivo)

  palabras_a_buscar = input("Palabras a buscar (separadas por coma): ")
  palabras = [palabra.strip() for palabra in palabras_a_buscar.split(',')]

  resultados = buscar_palabras(paginas, palabras)
  # print("resultados", resultados)
  mostrar_resultados(resultados)

main()
