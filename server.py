# -*- encoding: utf-8 -*-
from flask import Flask, request, Response
from gevent.pywsgi import WSGIServer
from gevent import monkey
import json
import logging
import sys
import requests
from utils import *
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
from keras.layers import Lambda
from keras.utils import multi_gpu_model
import numpy as np
import os
import keras.backend.tensorflow_backend as KTF

#进行配置，每个GPU使用60%上限现存
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

monkey.patch_all()
app = Flask(__name__)

dim = 512
maxlen = 64
maxRec = 10
dict_path = '/search/odin/guobk/data/model/chinese_L-12_H-768_A-12/vocab.txt' 
path_model0 = '/search/odin/guobk/data/simcse/'
path_model = os.path.join(path_model0,'model/model_002.h5')

encoder = keras.models.load_model(path_model,compile = False)
tokenizer = get_tokenizer(dict_path)

def convert_to_ids(inputStr):
    token_ids = [tokenizer.encode(inputStr, maxlen=maxlen)[0]]
    a_token_ids = sequence_padding(token_ids)
    return a_token_ids
def embedding(encoder,token_ids,mode='a'):
    vecs = encoder.predict([token_ids,
                            np.zeros_like(token_ids)],
                            verbose=True)
    if mode=='a':
        vecs = vecs[:,:dim]
    else:
        vecs = vecs[:,dim:]
    vecs = l2_normalize(vecs)
    return vecs
@app.route('/search', methods=['POST'])
def test1():
    r = request.json
    query = r["input"]
    try:
        a_token_ids = convert_to_ids(query)
        vec = embedding(encoder,a_token_ids,mode='a')[0]
        message = 'success'
    except Exception as e:
        # app.logger.error("error:",str(e))
        message = 'Error:'+str(e)
    response = {'message':message,'input':query}
    app.logger.error('SEARCH_output\t' + json.dumps(response))
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    port = int(sys.argv[1])
    server = WSGIServer(("0.0.0.0", port), app)
    print("Server started")
    server.serve_forever()


#