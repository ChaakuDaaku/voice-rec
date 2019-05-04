import flask as fl
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import librosa
from werkzeug import secure_filename
import os

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['wav'])
app = fl.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global model, graph
model = load_model('./model/voice_model.h5')
graph = tf.get_default_graph()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return fl.render_template('homepage.html')

@app.route('/result',methods=['GET', 'POST'])
def get_upload():
    if fl.request.method == 'POST':
        file = fl.request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            y, sr = librosa.load('uploads/'+ filename, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            ps = np.array([ps.reshape( (128, 128, 1) )])
            # y = keras.utils.to_categorical(y, 4)
            # y= y/y
            with graph.as_default():
                prediction = model.predict_classes(ps)
                # score = model.evaluate(ps, np.array(y))
                if prediction == 1:
                    return fl.render_template('result.html', result = 'linda')
                else:
                    return fl.render_template('result.html', result = 'random')
    
if __name__ == '__main__':
    app.run(debug=True)
