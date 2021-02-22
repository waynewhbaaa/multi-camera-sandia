import os
from flask import Flask, request, jsonify
from cam_client import Client

app = Flask(__name__)
client = None
data_dir = ''

@app.route("/init")
def initialize_server():
    id = request.args['id']
    dataset_dir = request.args['dataset_dir']
    print("Server initialize for cam: ", id)
    global client
    client=Client(id=id, dataset_dir=dataset_dir)
    return "OK"

@app.route("/bbox")
def send_bbox_info():
    id = request.args['start_id']
    global client
    results = client.first_phase(id)

    return jsonify(results)


@app.route("/video")
def send_video():
    bitrate = request.args['bitrate']
    global client
    results = client.second_phase(bitrate)

    return jsonify("OK")
