gpus=4
model=BERT
pooler=cls
savepath=/search/odin/guobk/data/simcse/model_$model_$pooler/
mkdir $savepath
nohup python -u train1.py $gpus 0 $model $cls allscene 0.3 >> log/train1-$model-$pooler.log 2>&1 &
