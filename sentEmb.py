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
dim = 512
maxlen = 64
maxRec = 10
dict_path = '/search/odin/guobk/data/model/chinese_L-12_H-768_A-12/vocab.txt' 
path_model0 = '/search/odin/guobk/data/simcse/'

models = ['bert-cls-base']
path_models = ['model/model_002.h5']

path_models = [os.path.join(path_model0,f) for f in path_models]

encoders = [keras.models.load_model(path_models[i],compile = False) for i in range(len(path_models))]
# 建立分词器
tokenizer = get_tokenizer(dict_path)

with open('/search/odin/guobk/data/simcse/20210621/train.txt','r') as f:
    S = f.read().strip().split('\n')
S = [s.split('\t') for s in S]
D = list(set([s[1] for s in S]))
Q = list(set([s[0] for s in S]))
D1 = D + ['我们']*(len(Q)-len(D))
data = [(Q[i],D1[i],0) for i in range(len(D1))]
a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)

IdxLayer = [23,39,55,71]

layers = [Model(inputs=encoders[0].input, outputs=encoders[0].layers[idx].output) for idx in IdxLayer]

V_b = []
for i in range(len(layers)):
    v = layers[i].predict([b_token_ids[:len(D)],np.zeros_like(b_token_ids[:len(D)])],verbose=True)
    V_b.append(v)

i = 0
a_vecs = embedding(encoders[i],a_token_ids,'a')
b_vecs = embedding(encoders[i],b_token_ids[:len(D)],'b')

# V_D = {D[i]:[np.float(t) for t in list(b_vecs[i])] for i in range(len(D))}
# V_Q = {Q[i]:a_vecs[i] for i in range(len(Q))}
# with open('/search/odin/guobk/data/simcse/20210621/V_D.json','w') as f:
#     json.dump(V_D,f,ensure_ascii=False)
np.save('/search/odin/guobk/data/simcse/20210621/Docs.npy',b_vecs)
with open('/search/odin/guobk/data/simcse/20210621/Docs.txt','w') as f:
    f.write('\n'.join(D))

np.save('/search/odin/guobk/data/simcse/20210621/Queries.npy',a_vecs)
with open('/search/odin/guobk/data/simcse/20210621/Queries.txt','w') as f:
    f.write('\n'.join(Q))