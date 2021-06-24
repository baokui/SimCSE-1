models=bert_cls_base,bert_cls_simple,bert_cls_simple_distill
path_models=model/model_002.h5,model_simple/model_final.h5,model_simple_distill/model_final.h5
path_docs=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json
path_queries=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-test-0623.json
path_target=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/test-0623.json
maxQ=200
python -u test_search.py $models $path_models $path_docs $path_queries $path_target $maxQ >> log/simple-distill.log 2>&1 &