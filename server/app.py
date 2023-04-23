from flask import (Flask, request, render_template)
from model import Model
app = Flask("app", static_folder="build/static", template_folder="build")
model = Model(app.static_folder)

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/predict", methods=["GET"])
def index():
    if request.method == "GET":
        res = model.predictV2(request.args['sentence'])
        return {"confidence": res}, 200
    
if __name__ == "__main__":
    app.run()