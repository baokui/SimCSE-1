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
def write_excel(path_target,data,sheetname='Sheet1'):
    import xlwt
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet(sheetname)
    # 写入excel
    # 参数对应 行, 列, 值
    rows,cols = len(data),len(data[0])
    for i in range(rows):
        for j in range(cols):
            worksheet.write(i, j, label=str(data[i][j]))
    # 保存
    workbook.save(path_target)
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

models = ['bert-cls-base','bert-cls','roberta-cls','bert-first-last-avg','roberta-last-avg','bert-last-avg']
path_models = ['model/model_002.h5','model_BERT_cls/model_final.h5',\
    'model_RoBERTa_cls/model_final.h5','model_BERT_first-last-avg/model_004.h5',\
        'model_RoBERTa_last-avg/model_004.h5','model_BERT_last-avg/model_final.h5']

models = ['bert-cls-base','bert-cls-simple']
path_models = ['model/model_002.h5','model_simple/model_final.h5']

path_models = [os.path.join(path_model0,f) for f in path_models]

encoders = [keras.models.load_model(path_models[i],compile = False) for i in range(len(path_models))]
# 建立分词器
tokenizer = get_tokenizer(dict_path)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json','r') as f:
    Docs = json.load(f)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries2.json','r') as f:
    Queries = json.load(f)
D = [Docs[i]['content'] for i in range(len(Docs))]
Q = [Queries[i]['content'].replace('*','') for i in range(len(Queries))]
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
    with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-compare-ab2-simple.json','w') as f:
        json.dump(Queries,f,ensure_ascii=False,indent=4)
# keys = ['id','content','clicks','rec_origin','rec_'+models[0],'rec_'+models[1]]
# Q = []
# for d in Queries:
#     Q.append({k:d[k] for k in keys})
# with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-compare-ab2-simple.json','w') as f:
#     json.dump(Q,f,ensure_ascii=False,indent=4)

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

    keys = ['rec_origin', 'rec_bert-cls-base', 'rec_bert-cls', 'rec_roberta-cls', 'rec_bert-first-last-avg', 'rec_roberta-last-avg', 'rec_bert-last-avg']
    R = [['inputText','clickContents','rec_base','label_base']]
    for k in keys[1:]:
        R[0].extend([k,k.replace('rec','label')])
    for i in range(len(D2)):
        n = max([len(D2[i][k]) for k in keys])
        r = [['']*len(R[0]) for _ in range(n)]
        r[0][0] = D2[i]['content'] if D2[i]['content'][0]!='*' else D2[i]['content'][1:]
        for k in range(len(D2[i]['clicks'])):
            r[k][1] = D2[i]['clicks'][k]
        for j in range(len(keys)):
            for k in range(len(D2[i][keys[j]])):
                r[k][2+int(2*j)] = D2[i][keys[j]][k] if D2[i][keys[j]][k][0]!='*' else D2[i][keys[j]][k][1:]
                r[k][2+int(2*j)+1] = '' if D2[i][keys[j]][k][0]!='*' else 1
        R.extend(r)
    
    write_excel('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-compare-ab.xls',R)
    with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries2.json','w') as f:
        json.dump(D2,f,ensure_ascii=False,indent=4)