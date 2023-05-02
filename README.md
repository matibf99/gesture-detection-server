# Gesture Detection Server

Este proyecto es un servidor creado con Flask que utiliza OpenCV y Mediapipe para recibir y procesar imágenes en tiempo real. La funcionalidad principal del servidor es la detección de gestos en imágenes.

## Requisitos

Antes de poder utilizar este servidor, necesitará tener instalado lo siguiente:
- Python 3.8
- Flask 2.0.1 o superior
- OpenCV 4.5.3 o superior
- Mediapipe 0.8.7 o superior
Además, se recomienda el uso de una cámara web para probar el servidor.

## Uso

Para utilizar este servidor, siga los siguientes pasos:

1. Clone el repositorio:

```bash
 git clone https://github.com/matibf99/gesture-detection-server
```

2. Instale las dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecute el servidor:

```bash
python main.py
```

4. Abra un navegador web y vaya a la dirección http://localhost:5000.

5. Permita el acceso a la cámara web si se le solicita.

6. Pruebe la funcionalidad del servidor enviando imágenes y viendo el resultado de la detección de gestos en tiempo real.

## Funcionamiento

El servidor utiliza Flask para recibir las imágenes enviadas por el cliente. Luego, utiliza OpenCV para procesar las imágenes y Mediapipe para detectar los gestos en las mismas. El resultado de la detección de gestos se devuelve al cliente y se muestra en tiempo real en el navegador web.