# Gesture Detection Server

Este proyecto es un servidor creado con Flask que utiliza OpenCV y Mediapipe para recibir y procesar imágenes en tiempo real. La funcionalidad principal del servidor es la detección de gestos en imágenes.

## Requisitos

Antes de poder utilizar este servidor, necesitará tener instalado lo siguiente:
- Python 3.8
- Flask 2.3.
- OpenCV 4.7.0.72
- Mediapipe 0.9.3.0

Además, se recomienda el uso de una cámara web para probar el servidor.

## Configurar la fuente de imágenes para el proyecto

Para usar este proyecto, es necesario configurar la variable 'IMG_SOURCE' en el archivo .env. Hay dos opciones para hacerlo:

- Si desea usar una webcam, escriba 'camera:///' seguido del índice de la cámara. Por ejemplo, 'camera:///0' para usar la primera cámara disponible.
- Si desea usar una URL para obtener imágenes, escriba 'url:///' seguido de la URL de la imagen. Por ejemplo, 'url:///http://example.com/image' para usar la imagen en la URL especificada.

Por ejemplo, si se quiere utilizar como fuente de imágenes la primera cámara disponible:

```bash
# .env
IMG_SOURCE=camera:///0
```

Si desea utilizar una URL en su lugar, puede configurar la variable de la siguiente manera:

```bash
# .env
IMG_SOURCE=url:///http://example.com/image
```

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

4. Configurar las variables de entorno como se explica en el apartado *'Configurar la fuente de imágenes para el proyecto'*.

3. Ejecute el servidor:

```bash
python main.py
```

4. Abra un navegador web y vaya a la dirección http://localhost:5000.

5. Permita el acceso a la cámara web si se le solicita.

6. Pruebe la funcionalidad del servidor enviando imágenes y viendo el resultado de la detección de gestos en tiempo real.

## Funcionamiento

El servidor utiliza Flask para recibir las imágenes enviadas por el cliente. Luego, utiliza OpenCV para procesar las imágenes y Mediapipe para detectar los gestos en las mismas. El resultado de la detección de gestos se devuelve al cliente y se muestra en tiempo real en el navegador web.