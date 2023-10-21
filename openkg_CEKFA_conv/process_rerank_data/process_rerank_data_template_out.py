import json
from copy import deepcopy

# 为了确保能完全对比，只修改文本中的模板，其余都一样（候选的排序、分数）
# 本文件只修改了 输出的前缀


PROMPT_OUTPUT = {
    "t1": "The correct ranking of the candidate answers would be:\n{option_label_str}\n",
    "t2": "Ranking:\n{option_label_str}\n",
    "t3": "\n{option_label_str}\n",
}
K = 10

template_out_old = "The correct ranking of the candidate answers would be:\n"
template_out_new = "Ranking:\n"     # TODO 
# template_out_new = "\n"     # TODO 

data_dir = 'results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/'
frerank_data_train_old = data_dir + "ReVerb20K_rerank_data_train_rs_t1_Top10.json"
frerank_data_valid_old = data_dir + "ReVerb20K_rerank_data_valid_rs_t1_Top10.json"
frerank_data_test_old = data_dir + "ReVerb20K_rerank_data_test_rs_t1_Top10.json"
frerank_data_train_new = data_dir + "ReVerb20K_rerank_data_train_rs_t2_Top10.json"      # TODO ReVerb45K t3
frerank_data_valid_new = data_dir + "ReVerb20K_rerank_data_valid_rs_t2_Top10.json"      # TODO
frerank_data_test_new = data_dir + "ReVerb20K_rerank_data_test_rs_t2_Top10.json"      # TODO

rerank_data_train_old = json.load(open(frerank_data_train_old, 'r'))
rerank_data_valid_old = json.load(open(frerank_data_valid_old, 'r'))
rerank_data_test_old = json.load(open(frerank_data_test_old, 'r'))

rerank_data_train_new = []
for data in rerank_data_train_old:
    if len(data["option_scores"]) < K:
        raise Exception
    data_new = deepcopy(data)
    data_new["output"] = data["output"].replace(template_out_old, template_out_new)
    rerank_data_train_new.append(data_new)

rerank_data_train_new_str = json.dumps(rerank_data_train_new, indent=4)
with open(frerank_data_train_new, 'w') as fw:
    fw.write(rerank_data_train_new_str)

rerank_data_valid_new = []
for data in rerank_data_valid_old:
    if len(data["option_scores"]) < K:
        raise Exception
    data_new = data
    data_new["output"] = data["output"].replace(template_out_old, template_out_new)
    rerank_data_valid_new.append(data_new)
rerank_data_valid_new_str = json.dumps(rerank_data_valid_new, indent=4)
with open(frerank_data_valid_new, 'w') as fw:
    fw.write(rerank_data_valid_new_str)


rerank_data_test_new = []
for data in rerank_data_test_old:
    if len(data["option_scores"]) < K:
        raise Exception
    data_new = data
    data_new["output"] = data["output"].replace(template_out_old, template_out_new)
    rerank_data_test_new.append(data_new)
rerank_data_test_new_str = json.dumps(rerank_data_test_new, indent=4)
with open(frerank_data_test_new, 'w') as fw:
    fw.write(rerank_data_test_new_str)