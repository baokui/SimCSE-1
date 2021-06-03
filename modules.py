from utils import *
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
from keras.layers import Lambda
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

def test(encoder,a_token_ids,b_token_ids,labels):
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
    return a_vecs,b_vecs,sims, corrcoef

from sklearn import preprocessing
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1
