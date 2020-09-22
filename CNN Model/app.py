import os
import json
import pandas as pd

from variables import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dog_sim import DogSimDetector
import logging
logging.getLogger('tensorflow').disabled = True

from util import *
from flask import Flask
from flask import jsonify
from flask import request
'''
        python -W ignore app.py
'''

app = Flask(__name__)

model = DogSimDetector()
model.model_conversion()
model.run()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    img_id = int(message['img_id'])
    model.predict_neighbour(img_id)

    # response = {
    #         'input_id': recommended_ids
    # }
    # return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=host, port= port, threaded=False)