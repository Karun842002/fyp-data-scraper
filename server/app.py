import random
from flask import (Flask, request, render_template)
from simplifier_new import serverParser


app = Flask("app", static_folder="build/static", template_folder="build")
PORT = 5000
HOST = "127.0.0.1"
DEBUG = True

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/predict", methods=["GET"])
def index():
    if request.method == "GET":
        sentence = request.args["sentence"]
        simpleSentence = serverParser(sentence)
        return {"confidence": random.randrange(0,100,1)}, 200
    
if __name__ == "__main__":
    app.run(threaded=True, host=HOST, port=PORT, debug=DEBUG)