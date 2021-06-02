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
encoder = keras.models.load_model('/search/odin/guobk/data/simcse/model/model_005.h5',compile = False)