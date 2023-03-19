from flask import (Flask, request)
import tensorflow as tf
from simplifier_new import serverParser
from tensorflow_addons.layers import CRF
import numpy as np
import os

os.environ["CLASSPATH"] = "C:/Users/karun/Documents/Code/fyp-data-scraper/stanford-corenlp-full-2018-02-27/*"
app = Flask("app")
trainlookup_layer = tf.keras.layers.StringLookup(max_tokens=20000)
trainlookup_layer.set_vocabulary('./words.txt')
PORT = 5000
HOST = "127.0.0.1"
DEBUG = True
m1 = tf.keras.models.load_model('./m1.h5', custom_objects={"CRF": CRF})
m2 = tf.keras.models.load_model('./m2.h5', custom_objects={"CRF": CRF})
fm = tf.keras.models.load_model('./fm.h5')

def processs(x, pred):
  res = []
  for i in range(len(x)):
    b = pred[i]
    tot = (sum(b) / len(b)) * 100
    label = 1
    if tot <= 50:
      label = 0
    res.append(label)
  return res

@app.route("/predict", methods=["POST"])
def index():
    if request.method == "POST":
        print(request.form)
        sentence = request.form["sentence"]
        simpleSentence = serverParser(sentence)
        inp1 = trainlookup_layer(tf.strings.lower(sentence.split()))
        inp2 = trainlookup_layer(tf.strings.lower(simpleSentence.split()))
        op1 = m1.predict(np.array([inp1.numpy(), ]))
        op1 = processs(np.array([inp1.numpy(), ]), op1)
        op2 = m2.predict(np.array([inp2.numpy(), ]))
        op2 = processs(np.array([inp2.numpy(), ]), op2)
        res = fm.predict([[op1, op2]])
        return {"confidence": float(res[0][0].item())}, 200
    
if __name__ == "__main__":
    app.run(threaded=True, host=HOST, port=PORT, debug=DEBUG)