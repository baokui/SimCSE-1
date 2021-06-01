#! -*- coding: utf-8 -*-
# SimCSE 中文测试
from utils import *
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
from keras.layers import Lambda
import random
devideLayer = Lambda(lambda inputs: inputs / 2)
jieba.initialize()
# 基本参数
dim = 512
model_type, pooling, task_name, dropout_rate = 'BERT cls allscene 0.3'.split(' ')
import json
import numpy as np
maxRecall = 20
minSim = 0.0
modeltag = 'simcse_sup_ab'
path_model = '/search/odin/guobk/data/simcse/model/model_005.h5'
path_target="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/result-simcse_sup_ab.json"
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json','r') as f:
    Docs = json.load(f)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries.json','r') as f:
    Queries = json.load(f)
D = [Docs[i]['content'] for i in range(len(Docs))]
Q = [Queries[i]['content'] for i in range(len(Queries))]
Q1 = Q + ['我们']*(len(D)-len(Q))
data = [(Q1[i],D[i],0) for i in range(len(D))]
if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64

# bert配置
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12',
    'WoBERT': 'chinese_wobert_plus_L-12_H-768_A-12',
    'NEZHA': 'nezha_base_wwm',
    'RoFormer': 'chinese_roformer_L-12_H-768_A-12',
    'BERT-large': 'uer/mixed_corpus_bert_large_model',
    'RoBERTa-large': 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16',
    'NEZHA-large': 'nezha_large_wwm',
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

dict_path = '/search/odin/guobk/data/model/%s/vocab.txt' % model_name

# 建立分词器
if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立模型
encoder = keras.models.load_model(path_model,compile = False)
# test data
token_ids = []
a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
for i in range(len(a_token_ids)):
    token_ids.append([a_token_ids[i],b_token_ids[i]])

a_token_ids = [t[0] for t in token_ids[:len(Q)]]
b_token_ids = [t[1] for t in token_ids]
a_vecs = encoder.predict([a_token_ids,
                        np.zeros_like(a_token_ids)],
                        verbose=True)[:,:dim]
b_vecs = encoder.predict([b_token_ids,
                        np.zeros_like(b_token_ids)],
                        verbose=True)[:,dim:]
# 标准化，相似度，相关系数
# labels = labels[:100]
a_vecs = l2_normalize(a_vecs)
b_vecs = l2_normalize(b_vecs)

Vq = a_vecs
Vd = b_vecs
Sd = D
R = []
for i in range(1000):
    s = Vd.dot(Vq[i])
    idx = np.argsort(-s)
    rec = [Sd[j]+'\t%0.4f'%s[j] for j in idx[:maxRecall] if s[j]>=minSim]
    d = Queries[i]
    d['rec_'+modeltag] = rec
    R.append(d)
    if i%100==0:
        print(i,len(Vq))
R = R[:1000]
with open(path_target,'w') as f:
    json.dump(R,f,ensure_ascii=False,indent=4)

def result_merge():
    import os
    import json
    files = os.listdir('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/')
    files = [os.path.join('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/',f) for f in files if 'result' == f[:6]]
    files = ['/search/odin/guobk/data/bert_semantic/finetuneData_new_test/result-20210522.json','/search/odin/guobk/data/bert_semantic/finetuneData_new_test/result-simcse_sup_ab.json']
    with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries.json','r') as f:
        Queries = json.load(f)
    F = []
    for file in files:
        with open(file,'r') as f:
            F.append(json.load(f))
    for i in range(1000):
        for j in range(len(files)):
            ks = F[j][i].keys()
            k = [t for t in ks if 'rec_' in t and 'origin' not in t][0]
            Queries[i][k] = F[j][i][k]
    with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-comp.json','w') as f:
        json.dump(Queries[:1000],f,ensure_ascii=False,indent=4)
# base          :1,3,0,3,4,4,5,3,5,4
# bert双塔       :1,0,0,4,3,4,5,4,5,4
# Simcse-unsup  :1,0,0,1,2,1,0,0,4
# simcse-sup    :1,0,0,0,0,0,3,0,0,1
# simcse-sup-ab :1,4,3,5,5,5,5,3,5,3
    