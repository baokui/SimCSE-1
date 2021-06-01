export CUDA_VISIBLE_DEVICES=1
python eval.py BERT cls LCQMC 0.3 >> log/test-LCQMC.log 2>&1 &