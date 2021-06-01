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
def test(token_ids,maxNb_pos=100,maxNb_neg=1000):
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
