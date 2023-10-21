import json
from copy import deepcopy
import pdb 


# 为了确保能完全对比，只修改文本中模板，其余都一样（候选的排序、分数）
# 本文件可以修改了 输入的前缀输入中问问题的方式 和 输出的前缀
# TODO 注意可能含有support info
PROMPT_INSTRUCTION = {
    "p1" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. Combine what you know, output a ranking of these candidate answers.\n\n",
    "p2" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. Output a ranking of these candidate answers.\n\n",
    "p3" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. You need to consider the suitability of each candidate answer as a solution to the question, and output the ranking of these candidates from high to low according to their suitability.\n\n",
    "p4" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. Combine what you know, consider the suitability of each candidate answer as a solution to the question, and output the ranking of these candidates from high to low according to their suitability.\n\n",
    "p5" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. The questions and candidate answers have been combined into candidate corresponding statements. Combine what you know, output a ranking of these candidate answers.\n\n",
    "p6" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers.",
    "p7" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. The questions and candidate answers have been combined into candidate corresponding statements.", # p6 p7 用于添加  The questions and candidate answers have been combined into candidate corresponding statements. （因为测试集也需要修改prompt，而kf3的部分含有known facts 的描述）
    "p8" : "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. The questions and candidate answers have been combined into candidate corresponding statements. Knowledge related to some candidates will be provided that may be useful for ranking.",  # 每个候选提供有效信息
}       # 训练集 默认没有 known facts 的 prompt
PROMPT_QUESTION = {
    'q1':{
        'tail': "{head} {relation} what?",
        'head': "what {relation} {tail}?",
        },
    'q2':{
        'tail': "What tail entities are missing from the statement '{head} {relation} _________' and can be inferred from the head entity '{head}' and the relation '{relation}'?",
        'head': "What head entities are missing from the statement '_________ {relation} {tail}' and can be inferred from the tail entity '{tail}' and the relation '{relation}'?",
        },
    'q3':{
        'tail': "{head} {relation} ____?",
        'head': "____ {relation} {tail}?",
        },
}

PROMPT_ANSWER = {
    'a1': ("", ),
    'a2': ("### Candidate Answers: ", ),
    'a3': ("Candidate Answers: ", ),
    'a4': ("", "use quotes"),                                 # 选项使用 '' 标记起来
    'a5': ("### Candidate Answers: ", "use quotes"),          # 选项使用 '' 标记起来
    'a6': ("", "fill in ques"),                               # 对应p5 p7,将每个答案和问题进行组合
    'a7': ("", "fill in ques and add cand_triples", "### Candidate supporting knowledge: "),          # 在a6的基础上添加每个答案的支持信息, 合并后放在问题前面
    'a8': ("", "fill in ques and add cand_triples", "supporting knowledge: ", "(No supporting knowledge was found)"),          # 在a6的基础上添加每个答案的支持信息，放在每个候选答案的位置
    'a9': ("", "fill in ques and add cand_triples", "supporting knowledge: ", ""),          # 在a6的基础上添加每个答案的支持信息，放在每个候选答案的位置
    
}   # TODO 实现可能不一样，需要修改

PROMPT_OUTPUT = {
    "t1": "The correct ranking of the candidate answers would be:\n",
    "t2": "Ranking:\n",
    "t3": "\n",
}

################## TODO To be changed ##################
dataset = "ReVerb45K"     # ReVerb20K  ReVerb45K
K = 10
CAND_INFO_TOPK = 3          # 每个候选最多使用的三元组数目
CAND_INFO_THRESHOLD=0.80    # 相似度阈值
KNOWNFACTS_NUM = 3


PROMPT_INSTRUCTION_old = "p6"       # TODO
PROMPT_INSTRUCTION_new = "p8"

PROMPT_ANSWER_old = "a1"
PROMPT_ANSWER_new = "a7"

PROMPT_QUESTION_old = "q1"
PROMPT_QUESTION_new = "q1"

PROMPT_OUTPUT_old = "t2"
PROMPT_OUTPUT_new = "t2"

file_prefix_old = "t2"
file_prefix_new = "t2_fillwen_cand_a7_Top"+str(CAND_INFO_TOPK)


##################   To be changed   ##################



if dataset == "ReVerb20K":
    data_dir = 'results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/' 
    fcandtriples = f"results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_candi/ReVerb20K_rerank_data_%s_rs_t2_Top50_candtriples.json"
else:
    # data_dir = 'results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data/' 
    data_dir = 'results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/' 
    fcandtriples = f"results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_candi/ReVerb45K_rerank_data_%s_rs_t2_Top50_candtriples.json"

frerank_data_train_old = data_dir + dataset + "_rerank_data_train_rs_"+file_prefix_old+"_Top30.json"                # file_prefix_old # TODO t2
frerank_data_valid_old = data_dir + dataset + "_rerank_data_valid_rs_"+file_prefix_old+"_Top30.json"                # t2
frerank_data_test_old = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_old+"_Top30.json"                  # t1                  
frerank_data_test_kf_old = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_old+"_Top30_kf"+str(KNOWNFACTS_NUM)+".json"           # t1
frerank_data_train_new = data_dir + dataset + "_rerank_data_train_rs_"+file_prefix_new+"_Top30.json"      
frerank_data_valid_new = data_dir + dataset + "_rerank_data_valid_rs_"+file_prefix_new+"_Top30.json"      
frerank_data_test_new = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_new+"_Top30.json"     
frerank_data_test_kf_new = data_dir + dataset + "_rerank_data_test_rs_"+file_prefix_new+"_Top30_kf"+str(KNOWNFACTS_NUM)+".json"   

fcandtriples_train = fcandtriples % "train"
fcandtriples_valid = fcandtriples % "valid"
fcandtriples_test = fcandtriples % "test"


REPLACE_PROMPT_INSTRUCTION = True if PROMPT_INSTRUCTION_old != PROMPT_INSTRUCTION_new else False
REPLACE_PROMPT_QUESTION = True if PROMPT_QUESTION_old != PROMPT_QUESTION_new else False
REPLACE_ANSWER_QUESTION = True if PROMPT_ANSWER_old != PROMPT_ANSWER_new else False
REPLACE_PROMPT_OUTPUT = True if PROMPT_OUTPUT_old != PROMPT_OUTPUT_new else False


def process_input_output_fillin_add_cand_facts(frerank_data_old, frerank_data_new, fcandtriples):
    print("read file:", frerank_data_old)
    rerank_data_old = json.load(open(frerank_data_old, 'r'))
    candtriples = json.load(open(fcandtriples, 'r'))
    candtriples_dict = {tuple(item["trip_id"]):item for item in candtriples}
    
    rerank_data_new = []
    for i, data in enumerate(rerank_data_old):
        # if "option_scores" in data and len(data["option_scores"]) < K:
        #     raise Exception
        
        # pdb.set_trace()
        input_old = data["input"]
        instruction, qa = input_old.split("### Question: ")  # instruction里可能包含 Supporting information
        question, ans = qa.split("\n\n", 1)

        candtriples_i = candtriples_dict[tuple(data["trip_id"])]["cand_info"] 

        data_new = data

        # pdb.set_trace()
        
        if REPLACE_PROMPT_INSTRUCTION:      
            instruction_new = instruction.replace(PROMPT_INSTRUCTION[PROMPT_INSTRUCTION_old], PROMPT_INSTRUCTION[PROMPT_INSTRUCTION_new])
        else:
            instruction_new = instruction
        
        trip_str = data["trip_str"]
        if trip_str[1].startswith("inverse of "):                                 # test 数据不一定是一正一反，只是一部分需要重排的
            tail, inv_rel, head = trip_str
            rel = inv_rel.replace("inverse of ", "")
        else:
            head, rel, tail = trip_str

        if REPLACE_PROMPT_QUESTION:
            if trip_str[1].startswith("inverse of "):
                question_new = PROMPT_QUESTION[PROMPT_QUESTION_new]['head'].format_map({"tail": tail, "relation": rel})
            else: 
                question_new = PROMPT_QUESTION[PROMPT_QUESTION_new]['tail'].format_map({"head": head, "relation": rel})     
        else:
            question_new = question
        
        if REPLACE_ANSWER_QUESTION:
            if PROMPT_ANSWER_old in ["a1", "a4", "a6"]:
                if PROMPT_ANSWER_new in ["a2", "a3", "a5"]:
                    ans_new = "{}{}".format(PROMPT_ANSWER[PROMPT_ANSWER_new][0], ans)
                else:
                    ans_new = ans
            elif PROMPT_ANSWER_old in ["a2", "a3", "a5"]:
                ans_new = ans.replace(PROMPT_ANSWER[PROMPT_ANSWER_old][0], PROMPT_ANSWER[PROMPT_ANSWER_new][0])
            else:
                raise NotImplementedError
            
            # pdb.set_trace()

            if PROMPT_ANSWER_new in ["a4", "a5"]:
                ans_, resp = ans_new.split("\n\n")
                ans_ = ans_.split("\n")
                ans_ = ["{}.'{}'".format(ans_i.split(". ")[0], ans_i.split(". ")[1])  for ans_i in ans_]
                ans_new = "{}\n\n{}".format("\n".join(ans_), resp)
            elif PROMPT_ANSWER_new in ["a6"]:
                ans_, resp = ans_new.split("\n\n")
                ans_ = ans_.split("\n")
                if trip_str[1].startswith("inverse of "):
                    ans_ = ["{}. {} {} {}?".format(ans_i.split(". ")[0],ans_i.split(". ")[1],rel,tail)  for ans_i in ans_]
                else:
                    ans_ = ["{}. {} {} {}?".format(ans_i.split(". ")[0], head,rel,ans_i.split(". ")[1])  for ans_i in ans_]
                ans_new = "{}\n\n{}".format("\n".join(ans_), resp)
            elif PROMPT_ANSWER_new in ["a7"]:
                ans_, resp = ans_new.split("\n\n")
                ans_ = ans_.split("\n")
                cand_info_all = []
                for ans_i in ans_:
                    cand_info_i = candtriples_i[ans_i.split(". ", 1)[1]]["triples"]
                    cand_info_i_use = []
                    for info in cand_info_i[:CAND_INFO_TOPK]:
                        if info["score"] >= CAND_INFO_THRESHOLD:
                            cand_info_i_use.append(info['triple_str'])
                        else:
                            break
                    if len(cand_info_i_use) > 0:
                        cand_info_all.append("; ".join(cand_info_i_use))
                
                if len(cand_info_all) > 0:
                    cand_info_all = "; ".join(cand_info_all)
                else:
                    cand_info_all = ""

                instruction_new = "{}{}{}".format(instruction_new, PROMPT_ANSWER[PROMPT_ANSWER_new][2], cand_info_all)

                if trip_str[1].startswith("inverse of "):
                    ans_ = ["{}. {} {} {}?".format(ans_i.split(". ")[0], ans_i.split(". ")[1],rel,tail)  for ans_i in ans_]
                else:
                    ans_ = ["{}. {} {} {}?".format(ans_i.split(". ")[0], head,rel,ans_i.split(". ")[1])  for ans_i in ans_]
                ans_new = "{}\n\n{}".format("\n".join(ans_), resp)
            elif PROMPT_ANSWER_new in ["a8", "a9"]:
                ans_, resp = ans_new.split("\n\n")
                ans_ = ans_.split("\n")
                ans_, resp = ans_new.split("\n\n")
                ans_ = ans_.split("\n")
                cand_info_all = []
                # pdb.set_trace()

                for ans_i in ans_:
                    cand_info_i = candtriples_i[ans_i.split(". ")[1]]["triples"]
                    cand_info_i_use = []
                    for info in cand_info_i[:CAND_INFO_TOPK]:
                        if info["score"] >= CAND_INFO_THRESHOLD:
                            cand_info_i_use.append(info['triple_str'])
                        else:
                            break
                    if len(cand_info_i_use) > 0:
                        cand_info_all.append("({}{})".format(PROMPT_ANSWER[PROMPT_ANSWER_new][2], "; ".join(cand_info_i_use)))
                    else:
                        cand_info_all.append(PROMPT_ANSWER[PROMPT_ANSWER_new][3])
                # pdb.set_trace()

                if trip_str[1].startswith("inverse of "):
                    ans_ = ["{}. {} {} {}? {}".format(ans_i.split(". ")[0],ans_i.split(". ")[1],rel,tail,
                                                           cand_info_all[j])  for j, ans_i in enumerate(ans_)]
                else:
                    ans_ = ["{}. {} {} {}? {}".format(ans_i.split(". ")[0], head,rel,ans_i.split(". ")[1],
                                                           cand_info_all[j])  for j, ans_i in enumerate(ans_)]
                ans_new = "{}\n\n{}".format("\n".join(ans_), resp)
            # pdb.set_trace()
        else:
            ans_new = ans
    
        input_new = "{}### Question: {}\n\n{}".format(instruction_new, question_new, ans_new)
        data_new["input"] = input_new
                         
        if REPLACE_PROMPT_OUTPUT and "output" in data:
            output = data["output"]
            output_new = output.replace(PROMPT_OUTPUT[PROMPT_OUTPUT_old], PROMPT_OUTPUT[PROMPT_OUTPUT_new])
            data_new["output"] = output_new
        
        # pdb.set_trace()
        if "cand_info" in data_new:
            del data_new['cand_info'] 

        rerank_data_new.append(data_new)
    rerank_data_new_str = json.dumps(rerank_data_new, indent=4)
    
    print("write file:", frerank_data_new)
    with open(frerank_data_new, 'w') as fw:
        fw.write(rerank_data_new_str)
    return 



# process_input_output_fillin_add_cand_facts(frerank_data_train_old, frerank_data_train_new)
# process_input_output_fillin_add_cand_facts(frerank_data_valid_old, frerank_data_valid_new)
process_input_output_fillin_add_cand_facts(frerank_data_test_old, frerank_data_test_new, fcandtriples_test)
process_input_output_fillin_add_cand_facts(frerank_data_test_kf_old, frerank_data_test_kf_new, fcandtriples_test)

