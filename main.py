# # Importamos las librerias de Python necesarias
import numpy as np
from skimage.io import imshow, imread
import cv2
import os

def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    """
    Esta función recibe una imagen y devuelve otra imagen recortada.

    Args:
      ruta_img (str): Ruta de la imagen original que se desea recortar.
      ruta_img_crop (str): Ruta donde se guardará la imagen recortada.
      x_inicial (int): Coordenada x inicial del área de recorte.
      x_final (int): Coordenada x final del área de recorte.
      y_inicial (int): Coordenada y inicial del área de recorte.
      y_final (int): Coordenada y final del área de recorte.

    Return
      None
    """
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))


import matplotlib.pyplot as plt

# 1
linkImagen1 = "./imagenes/hongo.png"
linkImagen2 = "./imagenes/estrella.png"

imagen1 = imread(linkImagen1)
imagen2 = imread(linkImagen2)

plt.subplot(1, 2, 1)
plt.imshow(imagen1)
plt.subplot(1, 2, 2)
plt.imshow(imagen2)
plt.show()

#2
print(imagen1.shape)
print(imagen2.shape)

#3
size = min(imagen1.shape[0], imagen1.shape[1])  
x_inicial = (imagen1.shape[0] - size) // 2
x_final = x_inicial + size
y_inicial = (imagen1.shape[1] - size) // 2
y_final = y_inicial + size
ruta_img1_crop = "./imagenes_recortadas/hongo.png"
ruta_img2_crop = "./imagenes_recortadas/estrella.png"

recortar_imagen_v2(linkImagen1, ruta_img1_crop, x_inicial, x_final, y_inicial, y_final)
recortar_imagen_v2(linkImagen2, ruta_img2_crop, x_inicial, x_final, y_inicial, y_final)
img1_recortada = imread(ruta_img1_crop)
img2_recortada = imread(ruta_img2_crop)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_recortada)
plt.subplot(1, 2, 2)
plt.imshow(img2_recortada)
plt.show()

#4
print("Matriz de la imagen 1 recortada:")
print(img1_recortada)
print("Tamaño:", img1_recortada.shape)

#5
#1 indica que la primera dimension va a ser la segunda dimension anterior
#0 indica que la 2da dimension va a ser la primera anterior
#y 2 indica que la tercera dimension permanece igual (los colores)
img1_traspuesta = np.transpose(img1_recortada, (1,0,2))
img2_traspuesta = np.transpose(img2_recortada, (1,0,2))
print("Matriz de la imagen 1 traspuesta:")
print(img1_traspuesta)
print("Matriz de la imagen 2 traspuesta:")
print(img2_traspuesta)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_traspuesta)
plt.subplot(1, 2, 2)
plt.imshow(img2_traspuesta)
plt.show()

#6
def convertir_escala_grises(imagen_color):
    alto, ancho, _ = imagen_color.shape
    imagen_gris = np.zeros((alto, ancho), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            promedio_canal = np.mean(imagen_color[i, j])
            imagen_gris[i, j] = promedio_canal

    return imagen_gris

img1_gris = convertir_escala_grises(img1_recortada)
img2_gris = convertir_escala_grises(img2_recortada)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_gris, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img2_gris, cmap='gray')
plt.show()

#7
def calcular_inversa(matriz):
    if matriz.shape[0] != matriz.shape[1]:
      print("La matriz no es cuadrada")
      return None

    if np.linalg.det(matriz) == 0:
      print("El determinante de la matriz es cero")
      return None
        
    matriz_inversa = np.linalg.inv(matriz)
    return matriz_inversa
    
img1_inversa = calcular_inversa(img1_gris)
img2_inversa = calcular_inversa(img2_gris)
if img1_inversa is not None:
    print("Inversa de la matriz de la imagen 1:\n", img1_inversa)
if img2_inversa is not None:
    print("\nInversa de la matriz de la imagen 2:\n", img2_inversa)

#8
alpha1 = 1.5  # α > 1
alpha2 = 0.5  # 0 < α < 1
img1_escalar_mayor = np.clip(alpha1 * img1_gris, 0, 255).astype(np.uint8)
img1_escalar_menor = np.clip(alpha2 * img1_gris, 0, 255).astype(np.uint8)
plt.subplot(2, 3, 2)
plt.imshow(img1_escalar_mayor, cmap='gray')
plt.title("Caso 1: α > 1")
plt.subplot(2, 3, 3)
plt.imshow(img1_escalar_menor, cmap='gray')
plt.title("Caso 2: 0 < α < 1")
plt.show()

