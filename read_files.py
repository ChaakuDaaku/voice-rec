import os
from pandas import DataFrame

data_path = os.path.dirname('./audio/linda/')
data = sorted(os.listdir(data_path))[0:11790]
label = ["linda"]*len(data)

dataset = {'audio': data, 'class': label}

df = DataFrame(dataset, columns= ['audio', 'class'])

df.to_csv('./lmeta.csv')