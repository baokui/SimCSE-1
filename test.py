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
path_models = ['model/model_002.h5','model_BERT_cls/model_final.h5',\
    'model_RoBERTa_cls/model_final.h5','model_BERT_first-last-avg/model_004.h5',\
        'model_RoBERTa_last-avg/model_004.h5','model_BERT_last-avg/model_final.h5']
path_models = [os.path.join(path_model0,f) for f in path_models]
models = ['bert-cls-base','bert-cls','roberta-cls','bert-first-last-avg','roberta-last-avg','bert-last-avg']
encoders = [keras.models.load_model(path_models[i],compile = False) for i in range(len(path_models))]
# 建立分词器
tokenizer = get_tokenizer(dict_path)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json','r') as f:
    Docs = json.load(f)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries.json','r') as f:
    Queries = json.load(f)
D = [Docs[i]['content'] for i in range(len(Docs))]
Q = [Queries[i]['content'] for i in range(len(Queries))]
Q1 = Q + ['我们']*(len(D)-len(Q))
data = [(Q1[i],D[i],0) for i in range(len(D))]
a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)

if 1:
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
    with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-compare-ab.json','w') as f:
        json.dump(Queries,f,ensure_ascii=False,indent=4)


def result_review():
    import json
    with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-compare-ab.json','r') as f:
        D = json.load(f)
    c0 = '*对生活充满希望'
    c1 = '*生活已太累。心好疲惫'
    D1 = []
    flag = False
    for d in D:
        if d['content'] == c0:
            D1.append(d)
            flag = True
            continue
        if d['content'] == c1:
            D1.append(d)
            flag = False
            continue
        if flag:
            D1.append(d)
    D2 = []
    for i in range(len(D1)):
        keys = D1[i].keys()
        c = 0
        for k in keys:
            if 'rec_' in k:
                c += len([t for t in D1[i][k] if t[0]=='*'])
        if c>0:
            D2.append(D1[i])

    R = {}
    for i in range(len(D2)):
        keys = D2[i].keys()
        for k in keys:
            if 'rec_' in k:
                c = len([t for t in D2[i][k] if t[0]=='*'])
                if k in R:
                    R[k]+=c
                else:
                    R[k]=c