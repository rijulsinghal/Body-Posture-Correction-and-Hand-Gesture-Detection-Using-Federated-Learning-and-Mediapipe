from flask import Flask, render_template, Response , request
import Ardha_Pincha_Mayurasana
import Bitilasana
import Chaturanga_Dandasana
import Dandasana
import Hanumanasana
import Krounchasana
import Matsyasana
import Paripurna_Navasana
import Parivrtta_Trikonasana
import Purvottanasana
import Utkata_Konasana
import Vrikshasana
import Virabhadrasana_2
import os

app = Flask(__name__)
yoga_posture = None
picfolder = os.path.join('static','yoga-posture') 
app.config['UPLOAD_FOLDER'] = picfolder

@app.route('/' , methods = ['POST' , 'GET'])
def index():
    pic =  os.path.join(app.config['UPLOAD_FOLDER'] ,'default.png')
    if request.method == "POST":
        global yoga_posture
        yoga_posture = request.form.get("yoga-posture")
        pic = os.path.join(app.config['UPLOAD_FOLDER'] , yoga_posture + '.png')
    return render_template('index.html' , user_image = pic)    

@app.route('/video_feed' , methods = ["GET","POST"])
def video_feed():
   if request.method == "POST":
        print(yoga_posture)
        if(yoga_posture == "Ardha_Pincha_Mayurasana"):
            return Response(Ardha_Pincha_Mayurasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Bitilasana"):
            return Response(Bitilasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Chaturanga_Dandasana"):
            return Response(Chaturanga_Dandasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Dandasana"):
            return Response(Dandasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Utkata_Konasana"):
            return Response(Utkata_Konasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Hanumanasana"):
            return Response(Hanumanasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Krounchasana"):
            return Response(Krounchasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Matsyasana"):
            return Response(Matsyasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Paripurna_Navasana"):
            return Response(Paripurna_Navasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Parivrtta_Trikonasana"):
            return Response(Parivrtta_Trikonasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Purvottanasana"):
            return Response(Purvottanasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Vrikshasana"):
            return Response(Vrikshasana.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
        elif(yoga_posture == "Virabhadrasana-2"):
            return Response(Virabhadrasana_2.main(),mimetype='multipart/x-mixed-replace; boundary=frame')
            

if __name__ == '__main__':
    app.run(debug=True)