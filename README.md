# Voice Recognition using Keras
Voice Recognition application using Keras libraries and deployed using Flask.
I have used [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset and [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

### Getting Started
Let's first get all the dependencies loaded.
```
pip install -r requirements.txt
```

Extract all the datasets, and add the 'wavs' folder from LJ Speech as 'fold11' amongst other folders in extracted UrbanSound8K folder

Now run main.ipynb cell by cell if you want to retrain your model. Or else you can use my saved model under the 'model' folder.

To start the application, type this in command line
```
python app/app.py
```


## TOOD
* ~Add Percent match feature~
