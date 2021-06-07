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
from keras.utils import multi_gpu_model
from modules import data_generator,test
import os
dim = 512
def sampleGenerator(token_ids,maxNb_pos=100,maxNb_neg=1000):
    a_token_ids = [t[0] for t in token_ids[:maxNb_pos]]
    b_token_ids = [t[1] for t in token_ids[:maxNb_pos]]
    labels = [1]*len(a_token_ids)
    b_token_ids_neg_cand = random.sample([t[1] for t in token_ids],maxNb_neg)
    a_token_ids_neg = []
    b_token_ids_neg = []
    for i in range(len(b_token_ids_neg_cand)):
        j = 0
        while sum(b_token_ids[j])==sum(b_token_ids_neg_cand[i]):
            j+=1
            if j==len(b_token_ids)-1:
                j = 0
        a_token_ids_neg.append(a_token_ids[j])
        b_token_ids_neg.append(b_token_ids_neg_cand[i])
        labels.append(0)
    a_token_ids.extend(a_token_ids_neg)
    b_token_ids.extend(b_token_ids_neg)
    return a_token_ids,b_token_ids,labels
def demo_test(encoder,a_token_ids,b_token_ids,labels,dim=512,mode='test'):
    #sim_trn, cor_train = test(train_token_ids)
    a,b,sim_tst, cor_test = test(encoder,a_token_ids,b_token_ids,labels,dim)
    print('corrcoef of %s is %0.4f'%(mode,cor_test))

# 基本参数
data_path = '/search/odin/guobk/data/chn/senteval_cn/'
devideLayer = Lambda(lambda inputs: inputs / 2)
jieba.initialize()
dim = 512
task_name = 'allscene'
maxlen = 64

# 加载数据集
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['valid', 'test']
}
dict_path = '/search/odin/guobk/data/model/chinese_L-12_H-768_A-12/vocab.txt' 
# 建立分词器
tokenizer = get_tokenizer(dict_path)
# test data
test_token_ids = []
valid_token_ids = []
for name, data in datasets.items():
    if 'test' in name:
        a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
        for i in range(len(a_token_ids)):
            test_token_ids.append([a_token_ids[i],b_token_ids[i]])
    elif 'valid' in name:
        a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
        for i in range(len(a_token_ids)):
            valid_token_ids.append([a_token_ids[i],b_token_ids[i]])

a_token_ids_tst,b_token_ids_tst,labels_tst = sampleGenerator(test_token_ids,1000,2000)
a_token_ids_val,b_token_ids_val,labels_val = sampleGenerator(valid_token_ids,1000,2000)
path_model0 = '/search/odin/guobk/data/simcse/model_new/init/model_init.h5'
path_model1 = '/search/odin/guobk/data/simcse/model_new/model_final.h5'
path_model2 = '/search/odin/guobk/data/simcse/model_roberta/model_final.h5'
encoder0 = keras.models.load_model(path_model0,compile = False)
encoder1 = keras.models.load_model(path_model1,compile = False)
encoder2 = keras.models.load_model(path_model2,compile = False)

print('init model')
demo_test(encoder0,a_token_ids_tst,b_token_ids_tst,labels_tst,mode='test')
demo_test(encoder0,a_token_ids_val,b_token_ids_val,labels_val,mode='test')
print('bert')
demo_test(encoder1,a_token_ids_tst,b_token_ids_tst,labels_tst,mode='test')
demo_test(encoder1,a_token_ids_val,b_token_ids_val,labels_val,mode='test')
print('roberta')
demo_test(encoder2,a_token_ids_tst,b_token_ids_tst,labels_tst,mode='test')
demo_test(encoder2,a_token_ids_val,b_token_ids_val,labels_val,mode='test')

