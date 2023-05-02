import threading
import cv2

from flask import Flask, render_template, Response

from image_processor import ImageProcessor


app = Flask(__name__)
image_processor = ImageProcessor()


def gen_frames():  # generate frame by frame from camera
    while True:
        last_frame = image_processor.last_frame

        if last_frame is None:
            print("last_frame is None")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    print("video_feed")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    print("index")
    return render_template("index.html")


if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()
    threading.Thread(target=lambda: image_processor.run()).start()
