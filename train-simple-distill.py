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
import os

gpus = int(sys.argv[1])
modelInit = sys.argv[2]=='1'
# 基本参数
model_type, pooling, task_name, dropout_rate, epochs,batch_size,path_train,config_path,save_dir,nb_epochs = sys.argv[3:]
nb_epochs = int(nb_epochs)
alpha = 0.8
beta = 1 - alpha
data_path = '/search/odin/guobk/data/chn/senteval_cn/'
# save_dir = "/search/odin/guobk/data/simcse/model_{}_{}-{}-batch{}/".format(model_type,pooling,epochs,batch_size)
path_model_init = '/search/odin/guobk/data/simcse/model_new/init/model_002.h5'

devideLayer = Lambda(lambda inputs: inputs / 2)

jieba.initialize()

dim = 512
# model_type, pooling, task_name, dropout_rate = 'BERT cls allscene 0.3'.split(' ')
assert model_type in [
    'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
    'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
]
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B','allscene']

dropout_rate = float(dropout_rate)
batch_size = int(batch_size)
maxlen = 64

V_Q = np.load('/search/odin/guobk/data/simcse/20210621/Queries.npy')
V_D = np.load('/search/odin/guobk/data/simcse/20210621/Docs.npy')
with open('/search/odin/guobk/data/simcse/20210621/Queries.txt','r') as f:
    Q = f.read().strip().split('\n')
with open('/search/odin/guobk/data/simcse/20210621/Docs.txt','r') as f:
    D = f.read().strip().split('\n')
V_D = {D[i].strip():V_D[i] for i in range(len(D))}
V_Q = {Q[i].strip():V_Q[i] for i in range(len(Q))}


def convert_to_ids_ab(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels, emb_a, emb_b = [], [], [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
        emb_a.append(V_Q[d[0]])
        emb_b.append(V_D[d[1]])
    token_ids = sequence_padding(a_token_ids+b_token_ids)
    n = int(len(token_ids)/2)
    a_token_ids,b_token_ids = token_ids[:n],token_ids[n:]
    return a_token_ids, b_token_ids, labels, emb_a, emb_b
class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        batch_emb = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids[0])
            batch_token_ids.append(token_ids[1])
            batch_emb.append(token_ids[2])
            batch_emb.append(token_ids[3])
            #batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                batch_emb = np.array(batch_emb)
                yield [batch_token_ids, batch_segment_ids], batch_emb
                batch_token_ids = []
                batch_emb = []
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
# config_path = '/search/odin/guobk/data/model/%s/bert_config.json' % model_name
checkpoint_path = '/search/odin/guobk/data/model/%s/bert_model.ckpt' % model_name
dict_path = '/search/odin/guobk/data/model/%s/vocab.txt' % model_name
# 建立分词器
tokenizer = get_tokenizer(dict_path)
# 建立模型
encoder0 = get_encoder_ab(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate,
        dim=dim
    )
if len(encoder0)==2:
    encoder,bert = encoder0
else:
    encoder = encoder0
# 语料id化
# train data
if 'npy' in path_train:
    train_token_ids = np.load(path_train)
else:
    trainData = load_data(path_train)
    for i in range(len(trainData)):
        trainData[i] = [t.strip() for t in trainData[i][:-1]]+[trainData[i][-1]]
    train_token_ids = []
    a_token_ids, b_token_ids, labels, emb_a, emb_b = convert_to_ids_ab(trainData, tokenizer, maxlen)
    for i in range(len(a_token_ids)):
        train_token_ids.append([a_token_ids[i],b_token_ids[i],emb_a[i],emb_b[i]])
    
    train_token_ids_txt = []
    for i in range(len(a_token_ids)):
        train_token_ids_txt.append([a_token_ids[i],b_token_ids[i]])
    train_token_ids_txt = np.array(train_token_ids_txt)
    np.save(path_train.replace('txt','npy'),train_token_ids_txt)
    emb = (emb_a,emb_b)
    np.save(path_train.replace('.txt','-vec.npy'),emb)


# train_token_ids = []
# for name, data in datasets.items():
#     if 'train' in name:
#         a_token_ids, b_token_ids, labels = convert_to_ids_ab(data, tokenizer, maxlen)
#         for i in range(len(a_token_ids)):
#             train_token_ids.append([a_token_ids[i],b_token_ids[i]])
# train_token_ids = np.array(train_token_ids)
# np.save(train_data,train_token_ids)
def simcse_loss(y_true0, y_pred):
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
    y_true_A = y_true0[::2]
    y_true_B = y_true0[1::2]
    mse_a = K.mean(K.square(y_true_A-outputA),axis=-1)
    mse_b = K.mean(K.square(y_true_B-outputB),axis=-1)
    mse_loss = K.mean(mse_a) + K.mean(mse_b)
    cse_loss = K.mean(loss)
    total_loss = alpha * cse_loss + beta * mse_loss
    return total_loss
# SimCSE训练
# test ...............
# encoder.save('/search/odin/guobk/data/simcse/model/model_init.h5')
# demo_test()

# train ................
encoder.summary()
if modelInit:
    encoder = keras.models.load_model(path_model_init,compile = False)
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=False, period=1)
train_generator = data_generator(train_token_ids, batch_size*gpus)

parallel_encoder = multi_gpu_model(encoder, gpus=gpus)
parallel_encoder.compile(loss=simcse_loss,
                       optimizer=Adam(1e-5))
encoder.save(os.path.join(save_dir,'model_init.h5'))
parallel_encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=nb_epochs,callbacks=[checkpointer]
)
encoder.save(os.path.join(save_dir,'model_final.h5'))

