gpus=4
model=BERT
pooler=last-avg
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
model=BERT
pooler=first-last-avg
savepath=/search/odin/guobk/data/simcse/model_${model}_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $pooler allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &