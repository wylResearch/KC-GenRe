import json

datadir = "/data/liqiwang/openkg/qlora-v2/results/"
##### 20k
# f_old = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_train_rs_t2_Top30.json"
# f_new = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_train_rs_t2_Top30_clean.json"

# f_old = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_valid_rs_t2_Top30.json"
# f_new = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_valid_rs_t2_Top30_clean.json"

# f_old = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_Top30.json"
# f_new = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_Top30_clean.json"

##### 45k
f_old = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_train_rs_t2_Top30.json"
f_new = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_train_rs_t2_Top30_clean.json"

f_old = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_valid_rs_t2_Top30.json"
f_new = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_valid_rs_t2_Top30_clean.json"

f_old = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_Top30.json"
f_new = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_Top30_clean.json"



rerank_data_old = json.load(open(f_old, 'r'))

rerank_data_new = []
for i, data in enumerate(rerank_data_old):
    data.pop("cand_info", None)
    rerank_data_new.append(data)

rerank_data_new_str = json.dumps(rerank_data_new, indent=4)
print("write file:", f_new)
with open(f_new, 'w') as fw:
    fw.write(rerank_data_new_str)
