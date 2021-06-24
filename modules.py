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

def test(encoder,a_token_ids,b_token_ids,labels,dim):
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

def getfiles(rootpath):
    datas = []
    def eachFile(filepath):
        if os.path.isfile(filepath):  # 如果是文件
            if 'part-' in filepath:
                datas.append(filepath)
                return
        fileNames = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
        for file in fileNames:
            newDir = filepath + '/' + file  # 将文件命加入到当前文件路径后面
            # print(newDir)
            # if os.path.isdir(newDir): # 如果是文件夹
            if os.path.isfile(newDir):  # 如果是文件
                if 'part-' in newDir:
                    datas.append(newDir)
            else:
                eachFile(newDir)  # 如果不是文件，递归这个文件夹的路径
    eachFile(rootpath)
    return datas
def readData1(files,adtypes,minlen=5):
    S = []
    for file in files:
        with open(file,'r') as f:
            s = f.read().strip().split('\n')
            s = [t.split('\t') for t in s]
            s = [t for t in s if t[2]!='-1' and len(t[1].strip())>=minlen and t[4][:4] in adtypes]
            S.extend(s)
        print(file,len(S))
    #S = [t.split('\t') for t in S]
    return S
def getDocs():
    import pymysql
    conn = pymysql.connect(
        host='mt.tugele.rds.sogou',
        user='tugele_new',
        password='tUgele2017OOT',
        charset='utf8',
        port=3306,
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    # ****python, 必须有一个游标对象， 用来给数据库发送sql语句， 并执行的.
    # 2. 创建游标对象，
    cur = conn.cursor()
    # 4). **************************数据库查询*****************************
    # sqli = 'SELECT * FROM tugele.ns_flx_wisdom_words_new'
    sqli = 'SELECT a.id,a.content,a.isDeleted FROM (tugele.ns_flx_wisdom_words_new a) where a.status=1 and a.isDeleted=0'
    cur.execute('SET NAMES utf8mb4')
    cur.execute("SET CHARACTER SET utf8mb4")
    cur.execute("SET character_set_connection=utf8mb4")
    result = cur.execute(sqli)  # 默认不返回查询结果集， 返回数据记录数。
    info = cur.fetchall()  # 3). 获取所有的查询结果
    # print(info)
    # print(len(info))
    # 4. 关闭游标
    cur.close()
    # 5. 关闭连接
    conn.close()
    S = {str(info[i][0]):info[i][1] for i in range(len(info))}
    return S