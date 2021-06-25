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
from modules import getDocs
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
def convert_to_ids_b(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    b_token_ids = []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d, maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return b_token_ids
dim = 512
maxlen = 64
maxRec = 10
dict_path = '/search/odin/guobk/data/model/chinese_L-12_H-768_A-12/vocab.txt' 
path_model0 = '/search/odin/guobk/data/simcse/'
path_model = 'model_simple/model_final.h5'
path_model = os.path.join(path_model0,path_model)
encoder = keras.models.load_model(path_model,compile = False)
# 建立分词器
tokenizer = get_tokenizer(dict_path)

Docs = getDocs()
D = [Docs[k] for k in Docs]
Ids = [k for k in Docs]
b_token_ids = convert_to_ids_b(D, tokenizer, maxlen)

b_vecs = embedding(encoder,b_token_ids,'b')
Xtrn = []
for i in range(len(Ids)):
    x = Ids[i]+'\t'+'\t'.join(["%0.8f"%t for t in b_vecs[i]])
    Xtrn.append(x)
import random
Xtst = random.sample(Xtrn,1000)
with open('/search/odin/guobk/data/bertSent/faiss_search/bert_simcse/Docs.txt','w') as f:
    f.write('\n'.join(Xtrn))
with open('/search/odin/guobk/data/bertSent/faiss_search/bert_simcse/Queries.txt','w') as f:
    f.write('\n'.join(Xtst))
with open('/search/odin/guobk/data/bertSent/faiss_search/bert_simcse/Docs.json','w') as f:
    json.dump(Docs,f,ensure_ascii=False,indent=4)