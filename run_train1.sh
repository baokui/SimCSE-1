gpus=4
model=BERT
pooler=last-avg
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $pooler allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4,5,6,7
gpus=4
model=BERT
pooler=first-last-avg
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $pooler allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
model=RoBERTa
pooler=last-avg
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $pooler allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
model=RoBERTa
pooler=first-last-avg
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $pooler allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
model=BERT
pooler=pooler
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $pooler allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
model=BERT
pooler=cls
epochs=epoch3
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler-${epochs}/
mkdir $savepath
nohup python -u train2.py $gpus 0 $model $pooler allscene 0.3 $epochs >> log/train1-$model-$pooler-$epochs.log 2>&1 &


export CUDA_VISIBLE_DEVICES=4,5,6,7
gpus=4
model=SimBERT
pooler=cls
epochs=epoch3
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler-${epochs}/
mkdir $savepath
nohup python -u train2.py $gpus 0 $model $pooler allscene 0.3 $epochs >> log/train1-$model-$pooler-$epochs.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
model=BERT
pooler=cls
epochs=epoch3
batch_size=128
path_train=/search/odin/guobk/data/simcse/20210621/train.npy
config_path=/search/odin/guobk/data/simcse/model_simple/bert_config.json
save_dir=/search/odin/guobk/data/simcse/model_simple
nb_epochs=5
nohup python -u train2.py $gpus 0 $model $pooler allscene 0.3  $epochs $batch_size $path_train $config_path $save_dir $nb_epochs >> log/train1-simple.log 2>&1 &