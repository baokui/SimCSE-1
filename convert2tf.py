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
#encoder.save_weights('/search/odin/guobk/data/simcse/model/model_005-weights.h5')
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dim = 512
path_export_model,version = '/search/odin/guobk/data/simcse/model_simple/pbmodel/','1'
data_path = '/search/odin/guobk/data/chn/senteval_cn/'
task_name = 'allscene'
train_data = '%s%s/%s.%s.data.npy' % (data_path, task_name, task_name, 'train')
train_token_ids = np.load(train_data)
x0,x1 = train_token_ids[0]
a_token_ids = [x0]
b_token_ids = [x1]
######################################################################################
# 转成pb文件
'''
python keras_to_tensorflow.py \
    --input_model="/search/odin/guobk/data/simcse/model_simple/model_final.h5" \
    --output_model="/search/odin/guobk/data/simcse/model_simple/model_final.pb"
'''
######################################################################################
# 生成pb文件
# 程序开始时声明
tf.reset_default_graph()
sess = tf.Session()
# 读取得到的pb文件加载模型
with gfile.FastGFile("/search/odin/guobk/data/simcse/model_simple/model_final.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # 把图加到session中
    tf.import_graph_def(graph_def, name='')
    # 获取当前计算图
graph = tf.get_default_graph()
tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
# 从图中获输出那一层
#pred = graph.get_tensor_by_name("dense_73/Tanh:0")
pred = graph.get_tensor_by_name("dense_25/Tanh:0")
inputToken = graph.get_tensor_by_name("Input-Token:0")
inputSegment = graph.get_tensor_by_name("Input-Segment:0")
inputs = [inputToken,inputSegment]
# 保存为pb文件
sess.run(tf.global_variables_initializer())
builder = tf.saved_model.builder.SavedModelBuilder(path_export_model + "/" + version)
# y = pred
y,_ = tf.split(pred,2, axis=-1)
#y = graph.get_tensor_by_name("Transformer-6-FeedForward-Norm/add_1:0")
signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'feat_index': inputToken,'feat_index1':inputSegment},
                                                                     outputs={'scores': y})
builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature,tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
builder.save()
print(path_export_model + "/" + version)
# with graph.as_default():
#     res_a = sess.run(pred, feed_dict={inputToken: a_token_ids,inputSegment:np.zeros_like(a_token_ids)})
#     res_b = sess.run(pred, feed_dict={inputToken: b_token_ids,inputSegment:np.zeros_like(b_token_ids)})
# # 执行得到结果
# res_a = res_a[:,:dim]
# res_b = res_b[:,dim:]
######################################################################################
# 原始embed
encoder = keras.models.load_model('/search/odin/guobk/data/simcse/model_simple/model_final.h5',compile = False)
a_vecs = encoder.predict([a_token_ids,
                            np.zeros_like(a_token_ids)],
                            verbose=True)[:,:dim]
b_vecs = encoder.predict([b_token_ids,
                        np.zeros_like(b_token_ids)],
                        verbose=True)[:,dim:]
a_vecs = a_vecs[0]
b_vecs = b_vecs[0]
######################################################################################
# docker 部署tf-serving
'''
source=/search/odin/guobk/data/simcse/model_simple/pbmodel/
model=simcseSearch
target=/models/$model
ps -ef | grep 8501|grep -v grep | awk '{print "kill -9 "$2}'|sh
sudo docker run -p 8501:8501 --mount type=bind,source=$source,target=$target -e MODEL_NAME=$model -t tensorflow/serving >> ./log/tfserving-cpu-$model.log 2>&1 &
curl http://localhost:8501/v1/models/$model/versions/0
curl http://localhost:8501/v1/models/$model #查看模型所有版本服务状态
curl http://localhost:8501/v1/models/$model/metadata #查看服务信息，输入大小等

'''

######################################################################################
# 验证生成环境结果
import requests
url = 'http://10.160.25.112:8501/v1/models/simcseSearch:predict'
# url = 'http://tensorflow-bert-semantic.thanos.sogou/v1/models/bert_semantic_simcse:predict'
# url = 'http://yunbiaoqing-tensorflow.thanos-lab.sogou/v1/models/bert_semantic_simcse:predict'
url = 'http://yunbiaoqing-tensorflow.thanos-lab.sogou/v1/models/bert_semantic_simcse:predict'
a_tokens = [int(t) for t in x0]
b_tokens = [int(t) for t in x1]
feed_dict = {'instances': [{'feat_index': a_tokens, 'feat_index1': [0]*len(x0)}]}
r = requests.post(url,json=feed_dict)
a_vecs1 = r.json()['predictions'][0]
feed_dict = {'instances': [{'feat_index': b_tokens, 'feat_index1': [0]*len(x1)}]}
r = requests.post(url,json=feed_dict)
b_vecs1 = r.json()['predictions'][0]

mse_a = np.mean([(a_vecs[i]-a_vecs1[i])**2 for i in range(len(a_vecs))])
mse_b = np.mean([(b_vecs[i]-b_vecs1[i])**2 for i in range(len(b_vecs))])

