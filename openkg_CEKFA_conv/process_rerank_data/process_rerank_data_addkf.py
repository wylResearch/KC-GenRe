import json
from copy import deepcopy
import pdb 
import os

from process_rerank_data import read_knownfacts

# 为了确保能完全对比，只添加已知事实，其余都一样（候选的排序、分数）


################## TODO To be changed ##################
dataset = "ReVerb20K"     # ReVerb20K  ReVerb45K
KNOWNFACTS_NUM = 3
SPLITER = "### Question:"     # support 加在 question前面
PROMPT_old = "Combine what you know"
PROMPT_new = "Combine what you know and the following knowledge"

file_prefix_old = "t2_fillwen"
# file_prefix_new = "t2_fillwen_kf3"

##################   To be changed   ##################
if dataset == "ReVerb20K":
    data_dir = 'results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/' 
    knownfacts_dir = './files_supporting/ReVerb20K/canonical_triples/ambv2/'
else:
    # data_dir = 'results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data/' 
    data_dir = 'results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/' 
    knownfacts_dir = './files_supporting/ReVerb45K/canonical_triples/ambv2/'

# frerank_data_train_old = data_dir + dataset + "_rerank_data_train_rs_"+file_prefix_old+"_Top10.json"                # file_prefix_old # TODO t2
# frerank_data_valid_old = data_dir + dataset + "_rerank_data_valid_rs_"+file_prefix_old+"_Top10.json"                # t2
# frerank_data_test_old = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_old+"_Top10.json"                  # t1                  
# frerank_data_train_new_kf = data_dir + dataset + "_rerank_data_train_rs_"+file_prefix_new+"_Top10.json"      
# frerank_data_valid_new_kf = data_dir + dataset + "_rerank_data_valid_rs_"+file_prefix_new+"_Top10.json"      
# frerank_data_test_new_kf = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_new+"_Top10.json"     
# frerank_data_test_kf_new = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_new+"_Top10_kf3.json"     
frerank_data_train_old = data_dir + dataset + "_rerank_data_train_rs_"+file_prefix_old+"_Top20.json"                # file_prefix_old # TODO t2
frerank_data_valid_old = data_dir + dataset + "_rerank_data_valid_rs_"+file_prefix_old+"_Top20.json"                # t2
frerank_data_test_old = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_old+"_Top20.json"                  # t1                  
frerank_data_train_new_kf = data_dir + dataset + "_rerank_data_train_rs_"+file_prefix_old+"_Top20_kf"+ str(KNOWNFACTS_NUM) + ".json"      
frerank_data_valid_new_kf = data_dir + dataset + "_rerank_data_valid_rs_"+file_prefix_old+"_Top20_kf"+ str(KNOWNFACTS_NUM) + ".json"     
frerank_data_test_new_kf = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_old+"_Top20_kf"+ str(KNOWNFACTS_NUM) + ".json"   

knownfacts_path_test = os.path.join(knownfacts_dir, 'retrieval_v2_test_Top'+str(KNOWNFACTS_NUM) +'.txt')
knownfacts_path_train = os.path.join(knownfacts_dir, 'retrieval_v2_train_Top'+str(KNOWNFACTS_NUM) +'.txt')
knownfacts_path_valid = os.path.join(knownfacts_dir, 'retrieval_v2_valid_Top'+str(KNOWNFACTS_NUM) +'.txt')
hr2kfs_train = read_knownfacts(knownfacts_path_train)
hr2kfs_valid = read_knownfacts(knownfacts_path_valid)
hr2kfs_test = read_knownfacts(knownfacts_path_test)


def process_input_output_addkf(frerank_data_old, frerank_data_new, hr2kfs):
    rerank_data_old = json.load(open(frerank_data_old, 'r'))
    
    rerank_data_new = []
    for i, data in enumerate(rerank_data_old):
        # if "option_scores" in data and len(data["option_scores"]) < K:
        #     raise Exception
        
        # pdb.set_trace()
        input_old = data["input"]
        instruction, qa = input_old.split(SPLITER)  # instruction里可能包含 Supporting information

        head_id, rel_id, tail_id = data['trip_id']
        if hr2kfs is not None:
            kf = hr2kfs[(head_id, rel_id)] if (head_id, rel_id) in hr2kfs else None
        input_new = "{}### Supporting information: {}\n\n{}{}".format(instruction, kf, SPLITER, qa)
        input_new = input_new.replace(PROMPT_old, PROMPT_new)
        data_new = data
        data_new["input"] = input_new
                         
        # pdb.set_trace()

        rerank_data_new.append(data_new)
    rerank_data_new_str = json.dumps(rerank_data_new, indent=4)
    
    print("write file:", frerank_data_new)
    with open(frerank_data_new, 'w') as fw:
        fw.write(rerank_data_new_str)
    return 



# process_input_output_addkf(frerank_data_train_old, frerank_data_train_new_kf, hr2kfs_train)
# process_input_output_addkf(frerank_data_valid_old, frerank_data_valid_new_kf, hr2kfs_valid)
process_input_output_addkf(frerank_data_test_old, frerank_data_test_new_kf, hr2kfs_test)

