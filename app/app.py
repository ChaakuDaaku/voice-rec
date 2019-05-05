import flask as fa
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import librosa
from werkzeug import secure_filename
import os

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['wav'])
app = fa.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global model, graph
model = load_model('./model/voice_model.h5')
graph = tf.get_default_graph()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return fa.render_template('homepage.html')

@app.route('/error')
def error():
    return fa.render_template('errorPage.html')

@app.route('/result',methods=['GET', 'POST'])
def get_upload():
    if fa.request.method == 'POST':
        file = fa.request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            y, sr = librosa.load('uploads/'+ filename, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            if ps.shape != (128, 128):
                errorMsg = "Please enter file with duration greater than 3 secs"
                return fa.render_template('homepage.html', errorMessage = errorMsg)
            else:    
                ps = np.array([ps.reshape( (128, 128, 1) )])
                with graph.as_default():
                    prediction = model.predict_classes(ps)
                    deg = model.predict(ps)
                    deg = deg[0][prediction]
                    deg = float(deg) * 100
                    if prediction == 1:
                        return fa.render_template('result.html', result = 'linda', degree = deg)
                    else:
                        return fa.render_template('result.html', result = 'random', degree = deg)
    
if __name__ == '__main__':
    app.run(debug=True)
