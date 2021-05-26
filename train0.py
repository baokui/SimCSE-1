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
    a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    if 'train' in name:
        for i in range(len(a_token_ids)):
            train_token_ids.append(a_token_ids[i])
            train_token_ids.append(b_token_ids[i])
train_token_ids = np.array(train_token_ids)
# test data
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
for name, data in datasets.items():
    if 'train' in name:
        continue
    a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    a_token_ids = list(a_token_ids)
    b_token_ids = list(b_token_ids)
    n = len(a_token_ids)
    for j in range(n):
        neg = random.sample(b_token_ids, 5)
        neg = [t for t in neg if sum(t)!=sum(b_token_ids[j])]
        a_token_ids.extend([a_token_ids[j]]*len(neg))
        b_token_ids.extend(neg)
        labels.extend([0]*len(neg))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)



# if task_name != 'PAWSX':
#     np.random.shuffle(train_token_ids)
#     train_token_ids = train_token_ids[:10000]


class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
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
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)
# SimCSE训练
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))

# test
test()

train_generator = data_generator(train_token_ids, 64)
encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=1
)

def test():
    # 语料向量化
    all_vecs = []
    for i in range(len(all_names)):
        if 'train' in all_names[i]:
            continue
        a_token_ids, b_token_ids = all_token_ids[i]
        # a_token_ids, b_token_ids = a_token_ids[:100], b_token_ids[:100]
        a_vecs = encoder.predict([a_token_ids,
                                np.zeros_like(a_token_ids)],
                                verbose=True)[:,:dim]
        b_vecs = encoder.predict([b_token_ids,
                                np.zeros_like(b_token_ids)],
                                verbose=True)[:,dim:]
        all_vecs.append((a_vecs, b_vecs))
    # 标准化，相似度，相关系数
    all_corrcoefs = []
    for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
        # labels = labels[:100]
        a_vecs = l2_normalize(a_vecs)
        b_vecs = l2_normalize(b_vecs)
        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = compute_corrcoef(labels, sims)
        all_corrcoefs.append(corrcoef)

    all_corrcoefs.extend([
        np.average(all_corrcoefs),
        np.average(all_corrcoefs, weights=all_weights)
    ])
    for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
        print('%s: %s' % (name, corrcoef))


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