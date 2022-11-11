import model as m
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(m.main_func(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)