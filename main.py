from flask import Flask, request, Response
from flask_cors import CORS
import jsonpickle
import numpy as np
import cv2
from predict import *

app = Flask(__name__)
CORS(app)

@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #Do AI predict.py
    prediction = predict(img)

    # print(img)



    # build a response dict to send back to client
    response = {'message': prediction}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True)

