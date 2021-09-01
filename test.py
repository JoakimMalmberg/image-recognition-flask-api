from __future__ import print_function
from flask.wrappers import Response
import requests
import json
import cv2
import jsonpickle


# File for testing API

BASE = "http://127.0.0.1:5000/"
test_url = BASE + 'api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread("images/dogtest.jpg")
_, img_encoded = cv2.imencode('.jpg', img)


res = requests.post(test_url, data=img_encoded.tostring(), headers=headers)


print(res.json())