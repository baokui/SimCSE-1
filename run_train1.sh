nohup python -u train1.py 4 0 BERT cls allscene 0.3 >> log/train1-bert.log 2>&1 &

nohup python -u train1.py 8 0 RoBERTa cls allscene 0.3 >> log/train1-RoBERTa.log 2>&1 &