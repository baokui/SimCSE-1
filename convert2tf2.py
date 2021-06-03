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
model_type, pooling, task_name, dropout_rate = 'BERT cls allscene 0.3'.split(' ')
dropout_rate = float(dropout_rate)
dim = 512
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