import argparse
import os
import json

import cv2
import numpy
import pytz
import logging
import platform

import yaml
from flask_cors import CORS
from flask import jsonify, request
from flask_helper import route, FlaskApp
from typing import List, Dict
from pathlib import Path
from base_model import BaseModel
from inference import Predictor

is_win = platform.system().lower().startswith("win")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
COUNTER = 0
TZ = pytz.timezone('Asia/Shanghai')
global_name = __name__
logging.getLogger('werkzeug').setLevel(logging.FATAL)
base_dir = os.path.dirname(os.path.abspath(__file__))


def load_models(cfg) -> Dict[str, BaseModel]:
    res = {}
    for name, model_cfg in cfg.items():
        cls = model_cfg['class']
        kwargs = model_cfg['kwargs']
        cls = eval(cls)
        m = cls(**kwargs)
        res[name] = m
    return res


class ModelServer(FlaskApp):

    def __init__(self, model_config, name=global_name):
        super().__init__(name)
        CORS(self.app)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        self.app.jinja_env.auto_reload = True
        model_config = yaml.load(open(model_config, 'rb'), yaml.FullLoader)
        self.models = load_models(model_config)

    @route('/find_models', methods=['GET'])
    def config(self):
        res = []
        for name, m in self.models.items():
            info = {
                "name": name,
                "labels": m.get_labels()
            }
            res.append(info)
        return jsonify(res)

    @route("/predict_imgs", methods=["POST"])
    def predict_imgs(self):
        print(request.form)
        data = json.loads(request.form.get("models"))
        imgs = []
        models = []
        for d in data:
            print(d)
            name = d['name']
            labels = d['labels']
            models.append([labels, self.models[name]])
        annotation = []
        res = {
            "rect": {
                "annotation": annotation,
            }
        }
        for key, f in request.files.items():
            print(key, f, f.name, f.filename)
            content = f.stream.read()
            img = cv2.imdecode(numpy.frombuffer(content, "uint8"), cv2.IMREAD_COLOR)
            for names, m in models:
                labels, boxes, _ = m.predict(img)
                for label, box in zip(labels, boxes):
                    if label not in names:
                        continue
                    annotation.append([label, box.tolist()])
            imgs.append(img)
        return jsonify(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='192.168.124.15', help='host')
    parser.add_argument('--port', type=str, default='5678', help='port')
    opt = parser.parse_args()
    server = ModelServer("model_server.yaml", global_name)
    server.start(opt.host, opt.port, debug=False)
