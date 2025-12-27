**LAB PROJECT – HAND PASSWORD + FINGER PAINT**
PROYECTO FINAL DE VISIÓN POR ORDENADOR

⸻

DESCRIPCIÓN GENERAL

Este proyecto implementa un sistema de Visión por Ordenador en tiempo real que utiliza una cámara como entrada para:
	1.	Detectar una SECUENCIA DE PATRONES realizada con la mano (número de dedos).
	2.	Usar dicha secuencia como un SISTEMA DE SEGURIDAD.
	3.	Una vez validada la secuencia, permitir DIBUJAR EN DIRECTO SOBRE LA PANTALLA usando la punta del dedo índice.

El proyecto cumple con los requisitos del enunciado:
	•	Uso obligatorio de cámara
	•	Calibración offline
	•	Sistema de seguridad por patrones visuales
	•	Sistema adicional de aplicación libre (pintura en tiempo real)

Se incluyen dos implementaciones del sistema completo:
	•	Versión basada únicamente en OpenCV (visión clásica)
	•	Versión basada en MediaPipe + OpenCV (más robusta)

⸻

ESTRUCTURA DEL PROYECTO

Lab_Project/
│
├─ assets/
│  ├─ lab-project-dark.png
│  └─ lab-project-light.png
│
├─ Data/
│  ├─ Imagen_1.jpg
│  ├─ …
│  └─ Imagen_18.jpg
│
├─ output/
│  └─ (imágenes generadas durante la calibración)
│
├─ src/
│  ├─ calibracion.py
│  ├─ codigo_completo_CV2.py
│  ├─ codigo_completo.py
│  ├─ hand_finger_sequence.py
│  ├─ finger_paint.py
│  └─ test.py
│
├─ Lab_Project.pdf
└─ README.txt

⸻

REQUISITOS

HARDWARE
	•	Cámara obligatoria (webcam de macbook)

SOFTWARE
	•	Python 3.9 o superior
	•	Librerías necesarias:
	•	opencv-python
	•	numpy
	•	imageio
	•	mediapipe (solo para la versión MediaPipe)

Instalación de dependencias:

pip install opencv-python numpy imageio mediapipe

Nota:
La librería mediapipe NO es necesaria si se ejecuta únicamente el sistema basado en OpenCV clásico.

⸻

FLUJO DE EJECUCIÓN (IMPORTANTE)

EL ORDEN CORRECTO DE EJECUCIÓN ES:
	1.	PRIMERO: ejecutar la calibración
	2.	DESPUÉS: ejecutar cualquiera de los programas completos

⸻

PASO 1 – CALIBRACIÓN DE LA CÁMARA (OBLIGATORIO)

Script:
src/calibracion.py

Este script realiza la calibración de la cámara de forma OFFLINE usando imágenes de un tablero de ajedrez.

Qué hace:
	•	Carga las imágenes Imagen_1.jpg … Imagen_18.jpg desde la carpeta Data
	•	Detecta esquinas del tablero
	•	Calcula:
	•	Matriz intrínseca de la cámara (K)
	•	Coeficientes de distorsión
	•	Error RMS de reproyección
	•	Guarda los resultados en el archivo:

calib.npz

Ejecución:

python src/calibracion.py

El archivo calib.npz se utilizará automáticamente por el sistema OpenCV clásico.

⸻

PASO 2 – EJECUTAR EL SISTEMA COMPLETO

Una vez realizada la calibración, se puede ejecutar CUALQUIERA de los siguientes programas:

⸻

OPCIÓN A – SISTEMA COMPLETO SOLO CON OPENCV (VISIÓN CLÁSICA)

Script:
src/codigo_completo_CV2.py

Ejecución:

python src/codigo_completo_CV2.py

Características:
	•	Segmentación de piel en espacio de color YCrCb
	•	Ajuste de parámetros mediante trackbars
	•	Detección del contorno principal de la mano
	•	Conteo de dedos usando convex hull y convexity defects
	•	Sistema de seguridad por secuencia de patrones
	•	Pintura en tiempo real con la punta del dedo
	•	Cambio de color y borrado mediante gestos

Controles de teclado:
	•	q o ESC → salir
	•	c → limpiar canvas
	•	r → reset completo del sistema
	• +	/ - → aumentar / disminuir grosor del trazo

Gestos (una vez desbloqueado):
	•	1 dedo → pintar
	•	5 dedos (3 segundos) → borrar canvas
	•	4 dedos (3 segundos) → color azul
	•	3 dedos (3 segundos) → color rojo
	•	2 dedos (3 segundos) → color verde

Notas:
	•	La mano debe permanecer dentro de la ROI marcada en pantalla.
	•	Los sliders permiten ajustar la segmentación de piel según la iluminación.

⸻

OPCIÓN B – SISTEMA COMPLETO CON MEDIAPIPE + OPENCV

Script:
src/codigo_completo.py

Ejecución:

python src/codigo_completo.py

Funcionamiento:
	•	Modo SECUENCIA:
	•	El sistema espera la secuencia de dedos: 5 → 4 → 3 → 2 → 1 → 0
	•	Se muestra en pantalla el paso actual
	•	Modo PINTURA:
	•	Se dibuja con la punta del dedo índice
	•	El dibujo se realiza dentro de un área delimitada
	•	Se permite borrar y cambiar de color mediante gestos mantenidos

Controles de teclado:
	•	ESPACIO → salir
	•	R → reiniciar (volver al modo secuencia)
	•	+	/ - → cambiar grosor del pincel
	•	c → limpiar canvas

Gestos mantenidos (3 segundos):
	•	0 dedos → borrar canvas
	•	5 dedos → color rojo
	•	4 dedos → color azul
	•	3 dedos → color verde
	•	2 dedos → color amarillo

⸻

SCRIPTS AUXILIARES

test.py
	•	Prueba simple de cámara y resolución.
Ejecución:
python src/test.py

hand_finger_sequence.py
	•	Detector independiente de secuencia de dedos (MediaPipe).
Ejecución:
python src/hand_finger_sequence.py

finger_paint.py
	•	Programa independiente de pintura con el dedo (MediaPipe).
Ejecución:
python src/finger_paint.py

⸻

NOTAS IMPORTANTES
	•	La calibración debe realizarse antes de usar el sistema completo.
	•	Si calib.npz no existe, el sistema OpenCV clásico continuará sin corrección de distorsión.
	•	Para mejorar rendimiento se puede reducir la resolución de captura.
	•	Se recomienda iluminación uniforme y fondos simples.

⸻

AUTORES

Proyecto académico de Visión por Ordenador.
Alejandro De Haro González
Daniel Escobosa Martínez

⸻

LICENCIA

Uso académico / educativo.