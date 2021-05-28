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
model_type, pooling, task_name, dropout_rate = sys.argv[1:]

dim = 512
# model_type, pooling, task_name, dropout_rate = 'BERT cls allscene 0.3'.split(' ')
assert model_type in [
    'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
    'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
]
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B','allscene']
dropout_rate = float(dropout_rate)

if task_name == 'PAWSX':
    maxlen = 128
else:
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
if model_type == 'NEZHA':
    checkpoint_path = '/search/odin/guobk/data/model/%s/model.ckpt-691689' % model_name
elif model_type == 'NEZHA-large':
    checkpoint_path = '/search/odin/guobk/data/model/%s/model.ckpt-346400' % model_name
else:
    checkpoint_path = '/search/odin/guobk/data/model/%s/bert_model.ckpt' % model_name
dict_path = '/search/odin/guobk/data/model/%s/vocab.txt' % model_name

# 建立分词器
if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立模型
if model_type == 'RoFormer':
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='roformer',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
elif 'NEZHA' in model_type:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='nezha',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
else:
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

class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids[0])
            batch_token_ids.append(token_ids[1])
            #batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


# def simcse_loss(y_true, y_pred):
#     """用于SimCSE训练的loss
#     """
#     # 构造标签
#     idxs = K.arange(0, K.shape(y_pred[0])[0]/2)
#     idxs_1 = idxs[None, :]
#     idxs_2 = idxs[:, None]
#     y_true = K.equal(idxs_1, idxs_2)
#     y_true = K.cast(y_true, K.floatx())
#     # 计算相似度
#     outputA, outputB = y_pred
#     outputA = outputA[::2] #取偶数行，即取A句的featureA
#     outputB = outputB[1::2] #取奇数行，即取B句的featureB
#     outputA = K.l2_normalize(outputA, axis=1)
#     outputB = K.l2_normalize(outputB, axis=1)
#     similarities = K.dot(outputA, K.transpose(outputB))
#     similarities = similarities * 20
#     loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
#     return K.mean(loss)

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0]/2)
    idxs_1 = idxs[None, :]
    idxs_2 = idxs[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    outputA = Lambda(lambda x: x[:,:dim])(y_pred)
    outputB = Lambda(lambda x: x[:,dim:])(y_pred)
    outputA = outputA[::2] #取偶数行，即取A句的featureA
    outputB = outputB[1::2] #取奇数行，即取B句的featureB
    outputA = K.l2_normalize(outputA, axis=1)
    outputB = K.l2_normalize(outputB, axis=1)
    similarities = K.dot(outputA, K.transpose(outputB))
    similarities = similarities * 2
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)
# SimCSE训练


# test ...............
encoder.save('/search/odin/guobk/data/simcse/model/model_init.h5')
demo_test1()

# train ................
import os
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
save_dir = "/search/odin/guobk/data/simcse/model/"
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=False, period=1)
train_generator = data_generator(train_token_ids, 64)
encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=5,callbacks=[checkpointer]
)
encoder.save('/search/odin/guobk/data/simcse/model/model_final.h5')
demo_test1()

# encoder = keras.models.load_model('/search/odin/guobk/data/simcse/model/model_trained.h5',compile = False)

def demo_test1():
    sim_trn, cor_train = test1(train_token_ids)
    sim_tst, cor_test = test1(test_token_ids)
    sim_val, cor_valid = test1(valid_token_ids)
    print('corrcoef of train, test, valid is %0.4f, %0.4f, %0.4f'%(cor_train,cor_test,cor_valid))
def test1(token_ids,maxNb_pos=100,maxNb_neg=1000):
    a_token_ids = [t[0] for t in token_ids[:maxNb_pos]]
    b_token_ids = [t[1] for t in token_ids[:maxNb_pos]]
    labels = [1]*len(a_token_ids)
    b_token_ids_neg_cand = random.sample([t[1] for t in train_token_ids],maxNb_neg)
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
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    #print('corrcoef:%0.4f'%corrcoef)
    return sims, corrcoef

from sklearn import preprocessing
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1

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