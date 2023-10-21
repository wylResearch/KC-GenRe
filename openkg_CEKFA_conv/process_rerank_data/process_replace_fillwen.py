import json
import pdb
# 读取test文件，将fillwen的格式去掉，保留其kf和cand


datadir = "/data/liqiwang/openkg/qlora-v2/results/"

# f_clean = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_Top30_clean.json"
# f_fillwen_supp = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_fillwen_cand_a7_Top3_Top30_kf3.json"
# f_clean_supp = datadir + "ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_cand_a7_Top3_Top30_kf3.json"

f_clean = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_Top30_clean.json"
f_fillwen_supp = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_fillwen_cand_a7_Top3_Top30_kf3.json"
f_clean_supp = datadir + "ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_cand_a7_Top3_Top30_kf3.json"


rerank_data_test_clean = json.load(open(f_clean, 'r'))
rerank_data_test_fillwen_supp = json.load(open(f_fillwen_supp, 'r'))

rerank_data_new = []
for i in range(len(rerank_data_test_clean)):
    data_clean = rerank_data_test_clean[i]
    data_fillwen_supp = rerank_data_test_fillwen_supp[i]

    input_clean = data_clean["input"].split("### Question:")
    input_fillwen_supp = data_fillwen_supp["input"].split("### Question:")

    input_clean_supp = input_fillwen_supp[0] + "### Question:" + input_clean[-1]
    data_clean["input"] = input_clean_supp
    rerank_data_new.append(data_clean)

rerank_data_new_str = json.dumps(rerank_data_new, indent=4)
print("write file:", f_clean_supp)
with open(f_clean_supp, 'w') as fw:
    fw.write(rerank_data_new_str)




