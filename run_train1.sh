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
epochs=epoch5
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler-${epochs}/
mkdir $savepath
nohup python -u train2.py $gpus 0 $model $pooler allscene 0.3 $epochs >> log/train1-$model-$pooler-$epochs.log 2>&1 &
