import requests

url = 'http://localhost:5000/predict_api'

r = requests.post(url, 'VEry nice product')

print(r)