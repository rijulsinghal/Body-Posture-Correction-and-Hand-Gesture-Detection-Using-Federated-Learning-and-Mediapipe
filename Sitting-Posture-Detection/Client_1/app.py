import model as m
from flask import Flask, render_template, Response , request

app = Flask(__name__)
mode = None
@app.route('/video_feed' , methods = ["GET","POST"])
def video_feed():
    if request.method == "POST":
        try:
            mode = request.form.get("mode")
            print(mode)
            return Response(m.main_func(mode),mimetype='multipart/x-mixed-replace; boundary=frame')
        except:
            return Response(m.main_func(None),mimetype='multipart/x-mixed-replace; boundary=frame')

    return Response(m.main_func(None),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True , port=8080)