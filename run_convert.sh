python keras_to_tensorflow.py \
    --input_model="/search/odin/guobk/data/simcse/model/model_005.h5" \
    --output_model="/search/odin/guobk/data/simcse/model/pbmodel/0/model_005.pb"


source=/search/odin/guobk/data/simcse/model/pbmodel/
model=simcseSearch
target=/models/$model
ps -ef | grep 8501|grep -v grep | awk '{print "kill -9 "$2}'|sh
sudo docker run -p 8501:8501 --mount type=bind,source=$source,target=$target -e MODEL_NAME=$model -t tensorflow/serving >> ./log/tfserving-cpu-$model.log 2>&1 &

curl http://localhost:8501/v1/models/$model/versions/0
curl http://localhost:8501/v1/models/$model #查看模型所有版本服务状态
curl http://localhost:8501/v1/models/$model/metadata #查看服务信息，输入大小等

url=http://localhost:8501/v1/models/$model:predict
#url=http://yunbiaoqing-tensorflow.thanos-lab.sogou/v1/models/vpa_prose_intention:predict

data="{\"instances\": [{ \"feat_index1\" :[0,0,0,0,0,0,0,0], \"feat_index\":[0,1,2,3,4,5,6,7]}]}"
curl -H "Content-Type: application/json" -d $data -X POST $url