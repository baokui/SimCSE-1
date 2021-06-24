from utils import *
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
from keras.layers import Lambda
import random
from keras.utils import multi_gpu_model
import json
import numpy as np
import os
from modules import test
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
models = sys.argv[1].split(',')
path_models = sys.argv[2].split(',')
path_docs = sys.argv[3]
path_queries = sys.argv[4]
path_target = sys.argv[5]
maxQ = int(sys.argv[6])
dim = 512
maxlen = 64
maxRec = 10
dict_path = '/search/odin/guobk/data/model/chinese_L-12_H-768_A-12/vocab.txt' 
path_model0 = '/search/odin/guobk/data/simcse/'

path_models = [os.path.join(path_model0,f) for f in path_models]
encoders = [keras.models.load_model(path_models[i],compile = False) for i in range(len(path_models))]
# 建立分词器
tokenizer = get_tokenizer(dict_path)
with open(path_docs,'r') as f:
    Docs = json.load(f)
with open(path_queries,'r') as f:
    Queries = json.load(f)[:maxQ]
D = [Docs[i]['content'] for i in range(len(Docs))]
Q = [Queries[i]['input'].replace('*','') for i in range(len(Queries))]
Q1 = Q + ['我们']*(len(D)-len(Q))
data = [(Q1[i],D[i],0) for i in range(len(D))]
a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
for i in range(len(encoders)):
    print(i)
    a_vecs = embedding(encoders[i],a_token_ids[:len(Q)],'a')
    b_vecs = embedding(encoders[i],b_token_ids,'b')
    a_vecs = a_vecs[:len(Q)]
    s = a_vecs.dot(np.transpose(b_vecs))
    idx = np.argsort(-s,axis=-1)
    for j in range(len(idx)):
        score = [s[j][ii] for ii in idx[j][:maxRec]]
        contents = [D[ii] for ii in idx[j][:maxRec]]
        Queries[j]['rec_'+models[i]] = [contents[k]+'\t%0.4f'%score[k] for k in range(len(score))]
with open(path_target,'w') as f:
    json.dump(Queries,f,ensure_ascii=False,indent=4)
