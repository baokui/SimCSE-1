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
from modules import data_generator,simcse_loss,test

devideLayer = Lambda(lambda inputs: inputs / 2)

jieba.initialize()

# 基本参数
# model_type, pooling, task_name, dropout_rate = sys.argv[1:]

dim = 512
model_type, pooling, task_name, dropout_rate = 'BERT cls allscene 0.3'.split(' ')
assert model_type in [
    'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
    'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
]
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B','allscene']
dropout_rate = float(dropout_rate)
maxlen = 64
# 加载数据集
data_path = '/search/odin/guobk/data/chn/senteval_cn/'
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train', 'valid', 'test']
}
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
config_path = '/search/odin/guobk/data/model/%s/bert_config.json' % model_name
checkpoint_path = '/search/odin/guobk/data/model/%s/bert_model.ckpt' % model_name
dict_path = '/search/odin/guobk/data/model/%s/vocab.txt' % model_name
# 建立分词器
tokenizer = get_tokenizer(dict_path)
# 建立模型
encoder = get_encoder_ab(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate,
        dim=dim
    )

# 语料id化
# train data
train_token_ids = []
for name, data in datasets.items():
    if 'train' in name:
        a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
        for i in range(len(a_token_ids)):
            train_token_ids.append([a_token_ids[i],b_token_ids[i]])
# train_token_ids = np.array(train_token_ids)
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

# SimCSE训练
# test ...............
encoder.save('/search/odin/guobk/data/simcse/model/model_init.h5')
demo_test1()

# train ................
import os
gpus = 8
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
save_dir = "/search/odin/guobk/data/simcse/model/"
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=False, period=1)
train_generator = data_generator(train_token_ids, 64*gpus)

parallel_encoder = multi_gpu_model(encoder, gpus=gpus)
parallel_encoder.compile(loss=simcse_loss,
                       optimizer=Adam(1e-5))

parallel_encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=5,callbacks=[checkpointer]
)
encoder.save('/search/odin/guobk/data/simcse/model/model_final.h5')
demo_test1()

# encoder = keras.models.load_model('/search/odin/guobk/data/simcse/model/model_trained.h5',compile = False)

def demo_test():
    sim_trn, cor_train = test(train_token_ids)
    sim_tst, cor_test = test(test_token_ids)
    sim_val, cor_valid = test(valid_token_ids)
    print('corrcoef of train, test, valid is %0.4f, %0.4f, %0.4f'%(cor_train,cor_test,cor_valid))

'''
import csv
import random
path_source = '/search/odin/guobk/data/chn/senteval_cn/allscene/train1.csv'
S = []
with open(path_source, 'r') as f:
    reader = csv.reader(f)
    print(type(reader))
    for row in reader:
        S.append(row)
S = S[1:]
S1 = [[s[0].replace('\t',''),s[1].replace('\t','')] for s in S]
S1 = ['\t'.join(t) for t in S1]
S1 = list(set(S1))
random.shuffle(S1)
Xvalid = S1[:100000]
Xtest = S1[100000:200000]
Xtrn = S1[200000:]
Xtrn = [t+'\t1' for t in Xtrn]
Xtest = [t+'\t1' for t in Xtest]
Xvalid = [t+'\t1' for t in Xvalid]
task_name = 'allscene'
data_path = '/search/odin/guobk/data/chn/senteval_cn/'
X = [Xtrn,Xvalid,Xtest]
f0 = ['train', 'valid', 'test']
for i in range(3):
    f = f0[i]
    filename = '%s%s/%s.%s.data' % (data_path, task_name, task_name, f)
    with open(filename,'w',encoding='utf-8') as f:
        f.write('\n'.join(X[i]))

# path_source = '/search/odin/guobk/data/chn/senteval_cn/allscene/train1.csv'
# S = []
# with open(path_source, 'r') as f:
# 	data = csv.reader((line.replace('\0','') for line in f), delimiter=",")
# 	for row in data:
# 		S.append(row)
# headers = S[0]
# rows = S[1:]
# with open(path_source,'w')as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(headers)
#     f_csv.writerows(rows)

'''