import os
from flask import Flask, request, jsonify, send_from_directory
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
    start_id = request.args['start_id']
    global client
    result = client.second_phase(bitrate, start_id)

    if(result == "OK"):
        try:
            return send_from_directory('temp-cropped', filename="temp.mp4", as_attachment=True)
        except FileNotFoundError:
            abort(404)

    return jsonify("Abort")
