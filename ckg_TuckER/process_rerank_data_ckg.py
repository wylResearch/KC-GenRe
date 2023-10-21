import os
import json
import argparse
import string
import numpy as np
import random
from collections import OrderedDict, defaultdict
from retrieve_supp_triples_ckg import read_ent2defi, get_train_triples_str, read_preds
import pdb



PROMPT_INPUT_FILL = (
    "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. "
    "The questions and candidate answers have been combined into candidate corresponding statements. "
    "Combine what you know, output a ranking of these candidate answers.\n\n"
    "### Question: {question}\n\n{option_ans}\n\n### Response:"
)           # 
PROMPT_INPUT_FILL_DEFI1 = (
    "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. "
    "The questions and candidate answers have been combined into candidate corresponding statements. "
    "Combine what you know, output a ranking of these candidate answers.\n\n"
    "### Question: {question}\n\n{option_ans}\n\n### Response:"
)          
PROMPT_INPUT_FILL_DEFI2 = (
    "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. "
    "The questions and candidate answers have been combined into candidate corresponding statements. "
    "Knowledge related to some candidates will be provided that may be useful for ranking. "
    "Combine what you know and the following knowledge, output a ranking of these candidate answers.\n\n"
    "### Candidate definitions: {cand_definitions}\n\n"
    "### Question: {question}\n\n{option_ans}\n\n### Response:"
)          



 
 

PROMPT_OUTPUT = {
    "t1": "The correct ranking of the candidate answers would be:\n{option_label_str}\n",
    "t2": "Ranking:\n{option_label_str}\n",
    "t3": "\n{option_label_str}\n",
}
OPTIONS_LETTER = string.ascii_uppercase + string.ascii_lowercase
print(OPTIONS_LETTER)


def procsss_rerank_data_test_KFs(args, preds_data_test, K=10):
    rerank_data = []
    rerank_data_candinfo = []
    rerank_data_defi1 = []
    rerank_data_defi2 = []
    options = OPTIONS_LETTER[:K]
    for data in preds_data_test:
        if data['rank'] <= K:   
            head_id, rel_id, tail_id = data['trip_id']
            head, rel, tail = data['trip_str']
            head_name = args.ent2name[head]
            head_defi = args.ent2defi[head]
            tail_name = args.ent2name[tail]
            tail_defi = args.ent2defi[tail]

            if rel.endswith("_reverse"):
                rel = rel.split("_reverse")[0]
                template = args.rel2template[rel]
                question = template.replace("[Y]", head_name)  
                question = question.replace("[X]", "_____")  
                question = question[:-2] + " ?"  
                reverse = True
            else:
                template = args.rel2template[rel]
                question = template.replace("[X]", head_name) 
                question = question.replace("[Y]", "_____")  
                question = question[:-2] + " ?"  
                reverse = False
            question_defi = "{} ({}: {})".format(question, head_name, head_defi)

            option_ans = []
            option_ans_defi1 = []
            option_ans_defi2 = []
            cand_info = []
            for option, cand_item in zip(options, data['cands']):
                cand_id, cand, cand_score = cand_item
                cand_name =  args.ent2name[cand]
                cand_defi =  args.ent2defi[cand]
                cand_fillwen = question.replace("_____", cand_name)
                cand_fillwen_defi = question.replace("_____", cand_name+ " ("+cand_defi+")")
                option_ans.append(option + ". " + cand_fillwen)       
                option_ans_defi1.append(option + ". " + cand_fillwen_defi)       
                option_ans_defi2.append("{}: {}".format(cand_name, cand_defi))
                cand_info.append((cand, cand_id))
            option_ans_str = "\n".join(option_ans)
            option_ans_str_defi1 = "\n".join(option_ans_defi1)
            option_ans_str_defi2 = "; ".join(option_ans_defi2)
            
            data_input = PROMPT_INPUT_FILL.format_map({"question": question, "option_ans": option_ans_str})
            data_input_defi1 = PROMPT_INPUT_FILL_DEFI1.format_map({"question": question_defi, "option_ans": option_ans_str_defi1})
            data_input_defi2 = PROMPT_INPUT_FILL_DEFI2.format_map({"question": question_defi, "option_ans": option_ans_str, "cand_definitions": option_ans_str_defi2})

            data_ = {
                "input":data_input,
                "trip_id":data["trip_id"],
                "trip_str":data["trip_str"],
                "question":question,
            }
            data_candinfo = {
                "input":data_input,
                "trip_id":data["trip_id"],
                "trip_str":data["trip_str"],
                "question":question,
                "cand_info":data['cands'],
            }
            data_defi1 = {
                "input":data_input_defi1,
                "trip_id":data["trip_id"],
                "trip_str":data["trip_str"],
                "question":question,
            }
            data_defi2 = {
                "input":data_input_defi2,
                "trip_id":data["trip_id"],
                "trip_str":data["trip_str"],
                "question":question,
            }
            rerank_data.append(data_)
            rerank_data_defi1.append(data_defi1)
            rerank_data_defi2.append(data_defi2)
            rerank_data_candinfo.append(data_candinfo)
    return rerank_data, rerank_data_candinfo, rerank_data_defi1, rerank_data_defi2

    
def procsss_rerank_data_train_KFs_rankscore(args, preds_data_train, K=10, out_template='t1', filter_no_candi=False):
    # 使用测试数据的目标实体所在的cluster的所有t作为正确的候选
    # 排除掉候选中其他正确cluster对应的实体，剩下的实体作为候选
    missing_num_options_e0 = 0
    missing_num_options_ltK = 0

    rerank_data = []
    rerank_data_candinfo = []
    rerank_data_defi1 = []
    rerank_data_defi2 = []
    options = OPTIONS_LETTER[:K]
    for data in preds_data_train:
        head_id, rel_id, tail_id = data['trip_id']
        head, rel, tail = data['trip_str']
        head_name = args.ent2name[head]
        head_defi = args.ent2defi[head]
        tail_name = args.ent2name[tail]
        tail_defi = args.ent2defi[tail]

        if rel.endswith("_reverse"):
            rel = rel.split("_reverse")[0]
            template = args.rel2template[rel]
            question = template.replace("[Y]", head_name)  
            question = question.replace("[X]", "_____")  
            question = question[:-2] + " ?"  
            reverse = True
        else:
            template = args.rel2template[rel]
            question = template.replace("[X]", head_name) 
            question = question.replace("[Y]", "_____")  
            question = question[:-2] + " ?"  
            reverse = False
        question_defi = "{} ({}: {})".format(question, head_name, head_defi)
        
        # 训练集，只选当前三元组对应的尾实体作为正确的
        option_ans = [[0, tail, cand_item[2], tail_id, tail_name, tail_defi] for cand_item in data['cands'] if cand_item[1] == tail]
        if len(option_ans) == 0:    # topK 里没有目标实体
            option_ans = [[0, tail, 100000, tail_id, tail_name, tail_defi]]   # 先随机赋值一个很大的分数
        elif len(option_ans) > 1:
            raise Exception
        idx = 1 
        
        for cand_item in data['cands']:
            cand_id, cand, cand_score = cand_item
            if cand_id != tail_id:
                cand_name =  args.ent2name[cand]
                cand_defi =  args.ent2defi[cand]
                option_ans.append([idx, cand, cand_score, cand_id, cand_name, cand_defi])
                idx += 1
            if idx == K:
                break
        
        if len(option_ans) == 1: # 有可能训练集存的候选都是要filter的,所以没有目标?
            missing_num_options_e0 += 1
            if filter_no_candi:
                continue
            else:
                option_ans[0][2] = 1    # 这个值无所谓，因为在计算的时候因为只有这一个选项
        elif len(option_ans) < K:
            missing_num_options_ltK += 1
        
        # pdb.set_trace()

        # 如果目标答案的分数不是最好的，随机改为最好的用于排序分数的训练
        if len(option_ans) > 1 and option_ans[0][2] < option_ans[1][2]:     # 第一阶段的分数是越大越好 !!! 
            high = option_ans[1][2] + (option_ans[1][2] - option_ans[0][2])
            option_ans[0][2] = random.uniform(option_ans[1][2], high)         # 不从0开始随机取分数了,怕和后面的分数分布差距太大! 而且从0开始不对（候选分数是负的时算出来的目标分数不是最小的）
        elif len(option_ans) > 1 and option_ans[0][2] == option_ans[1][2]:     # 第一阶段的分数是越小越好
            if len(option_ans) > 2:
                high = option_ans[1][2] + abs(option_ans[2][2] - option_ans[1][2])
                option_ans[0][2] = random.uniform(option_ans[1][2], high)
            else:
                raise Exception

        # 打乱列表顺序
        option_ans_random = list(option_ans)
        random.shuffle(option_ans_random)

        option_ans_input = []
        option_ans_input_defi1 = []
        option_ans_input_defi2 = []
        option_scores = []
        for option, ans in zip(options, option_ans_random):
            ans_fillwen = question.replace("_____", ans[4])
            ans_fillwen_defi = question.replace("_____", ans[4]+" ("+ans[5]+")")
            option_ans_input.append(option + ". " + ans_fillwen)
            option_ans_input_defi1.append(option + ". " + ans_fillwen_defi)
            option_ans_input_defi2.append("{}: {}".format(ans[4], ans[5]))
            option_scores.append(ans[2])
        option_ans_input_str = "\n".join(option_ans_input)
        option_ans_input_str_defi1 = "\n".join(option_ans_input_defi1)
        option_ans_input_str_defi2 = "; ".join(option_ans_input_defi2)

        indexes = [option_ans_random.index(item) for item in option_ans]
        
        option_label = [options[idx_random] + ". " for i, idx_random in enumerate(indexes)]
        option_label_str = "\n".join(option_label)
        
        data_input = PROMPT_INPUT_FILL.format_map({"question": question, "option_ans": option_ans_input_str})
        data_input_defi1 = PROMPT_INPUT_FILL_DEFI1.format_map({"question": question_defi, "option_ans": option_ans_input_str_defi1})
        data_input_defi2 = PROMPT_INPUT_FILL_DEFI2.format_map({"question": question_defi, "option_ans": option_ans_input_str, "cand_definitions": option_ans_input_str_defi2})

        option_label_str = PROMPT_OUTPUT[out_template].format_map({"option_label_str":option_label_str})
        # pdb.set_trace()
        cand_info = [(item[1], item[3]) for item in option_ans]
        data_ = {
            "input":data_input,
            "output":option_label_str,
            "trip_id":data["trip_id"],
            "trip_str":data["trip_str"],
            "option_scores":option_scores, # 按照 选项A-
        }
        
        data_candinfo = {
            "input":data_input,
            "trip_id":data["trip_id"],
            "trip_str":data["trip_str"],
            "question":question,
            "cand_info":data['cands'],
            "option_ans_random":option_ans_random,
        }

        data_defi1 = {
            "input":data_input_defi1,
            "output":option_label_str,
            "trip_id":data["trip_id"],
            "trip_str":data["trip_str"],
            "option_scores":option_scores, # 按照 选项A-
        }

        data_defi2 = {
            "input":data_input_defi2,
            "output":option_label_str,
            "trip_id":data["trip_id"],
            "trip_str":data["trip_str"],
            "option_scores":option_scores, # 按照 选项A-
        }
        rerank_data.append(data_)
        rerank_data_defi1.append(data_defi1)
        rerank_data_defi2.append(data_defi2)
        rerank_data_candinfo.append(data_candinfo)

    print(missing_num_options_e0)
    print(missing_num_options_ltK)
    return rerank_data, rerank_data_candinfo, rerank_data_defi1, rerank_data_defi2




def write_data(rerank_data, outfile):
    rerank_data_str = json.dumps(rerank_data, indent=4)
    with open(outfile, 'w') as fw:
        fw.write(rerank_data_str)
    return 


def process_main(args):
    # 只输出 无kf和无cand的版本
    preds_data_train = read_preds(args.preds_path_json_train)
    preds_data_valid = read_preds(args.preds_path_json_valid)
    preds_data_test = read_preds(args.preds_path_json_test)
    
    rerank_data_train, rerank_data_train_candinfo, rerank_data_train_defi1, rerank_data_train_defi2 = procsss_rerank_data_train_KFs_rankscore(args, preds_data_train, args.rerank_Top_K, args.out_template, args.filter_no_candi)
    print("writing file:", args.outfile_train)
    write_data(rerank_data_train, args.outfile_train)
    print("writing file:", args.outfile_train_defi1)
    write_data(rerank_data_train_defi1, args.outfile_train_defi1)
    print("writing file:", args.outfile_train_defi2)
    write_data(rerank_data_train_defi2, args.outfile_train_defi2)
    print("writing file:", args.outfile_train_candinfo)
    write_data(rerank_data_train_candinfo, args.outfile_train_candinfo)

    rerank_data_valid, rerank_data_valid_candinfo, rerank_data_valid_defi1, rerank_data_valid_defi2 = procsss_rerank_data_train_KFs_rankscore(args, preds_data_valid, args.rerank_Top_K, args.out_template, args.filter_no_candi)
    print("writing file:", args.outfile_valid)
    write_data(rerank_data_valid, args.outfile_valid)
    print("writing file:", args.outfile_valid_defi1)
    write_data(rerank_data_valid_defi1, args.outfile_valid_defi1)
    print("writing file:", args.outfile_valid_defi2)
    write_data(rerank_data_valid_defi2, args.outfile_valid_defi2)
    print("writing file:", args.outfile_valid_candinfo)
    write_data(rerank_data_valid_candinfo, args.outfile_valid_candinfo)

    rerank_data_test, rerank_data_test_candinfo, rerank_data_test_defi1, rerank_data_test_defi2 = procsss_rerank_data_test_KFs(args, preds_data_test, args.rerank_Top_K)
    print("writing file:", args.outfile_test)
    write_data(rerank_data_test, args.outfile_test)
    print("writing file:", args.outfile_test_defi1)
    write_data(rerank_data_test_defi1, args.outfile_test_defi1)
    print("writing file:", args.outfile_test_defi2)
    write_data(rerank_data_test_defi2, args.outfile_test_defi2)
    print("writing file:", args.outfile_test_candinfo)
    write_data(rerank_data_test_candinfo, args.outfile_test_candinfo)
    return 




    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process_rerank_data for KG')
    parser.add_argument('--data_path', dest='data_path', default='./dataset/', help='directory path of KG datasets')        
    parser.add_argument('--dataset', dest='dataset', default='Wiki27K', help='Dataset Choice')
    parser.add_argument('--saved_model_name_rank', dest='saved_model_name_rank', default='', help='')       
    parser.add_argument('--rerank_Top_K', dest='rerank_Top_K', default=10, type=int, help='rerank top K predicted candidates.') 
    parser.add_argument('--out_template', dest='out_template', default='t2', help='')  
    
    parser.add_argument('--filter_no_candi',   dest='filter_no_candi',   default=False,  action='store_true', help='')

    args = parser.parse_args()

    topk = 50
    args.ranks_path_test = os.path.join(args.saved_model_name_rank, 'ranks_test.npy')
    args.preds_path_json_train = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_train.json')     
    args.preds_path_json_valid = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_valid.json')
    args.preds_path_json_test = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_test_retrieved.json')
    # set data files
    args.data_path = os.path.join(args.data_path, args.dataset) 
    args.data_files = {
        'ent2id_path'       : args.saved_model_name_rank + '/ent2id.json',
        'rel2id_path'       : args.saved_model_name_rank + '/rel2id.json',
        'train_trip_path'   : args.data_path + '/train.txt',
        'test_trip_path'    : args.data_path + '/test.txt',
        'ent2defi_path'   : args.data_path + '/entity2definition.txt',
        'ent2name_path'   : args.data_path + '/entity2label.txt',
        'rel2template_path'   : args.data_path + '/relation2template.json',
    }
    
    args.ent2id = json.load(open(args.data_files["ent2id_path"], 'r')) 
    args.rel2id = json.load(open(args.data_files["rel2id_path"], 'r')) 
    args.rel2template = json.load(open(args.data_files["rel2template_path"], 'r')) 
    args.ent2defi = read_ent2defi(args.data_files["ent2defi_path"])
    args.ent2name = read_ent2defi(args.data_files["ent2name_path"])

    args.train_triples_str, args.train_triples_str_defi, args.ent2triplestr, args.train_triples_hr_str, args.train_triples_hr_str2triple_id = get_train_triples_str(args.data_files["train_trip_path"], args)

    filter_flag = '_filter' if args.filter_no_candi else ""
    fprefix = "rs_%s_Top%d%s.json" % (args.out_template, args.rerank_Top_K, filter_flag)
    fprefix_defi1 = "rs_%s_Top%d_defi1%s.json" % (args.out_template, args.rerank_Top_K, filter_flag)
    fprefix_defi2 = "rs_%s_Top%d_defi2%s.json" % (args.out_template, args.rerank_Top_K, filter_flag)
    fprefix_candinfo = "rs_%s_Top%d_candinfo%s.json" % (args.out_template, args.rerank_Top_K, filter_flag)
    
    args.outdir = os.path.join(args.saved_model_name_rank, "rerank_data_new")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    args.outfile_train = os.path.join(args.outdir, args.dataset + '_rerank_data_train_' + fprefix) 
    args.outfile_train_defi1 = os.path.join(args.outdir, args.dataset + '_rerank_data_train_' + fprefix_defi1) 
    args.outfile_train_defi2 = os.path.join(args.outdir, args.dataset + '_rerank_data_train_' + fprefix_defi2) 
    args.outfile_train_candinfo = os.path.join(args.outdir, args.dataset + '_rerank_data_train_' + fprefix_candinfo) 
    
    args.outfile_valid = os.path.join(args.outdir, args.dataset + '_rerank_data_valid_' + fprefix) 
    args.outfile_valid_defi1 = os.path.join(args.outdir, args.dataset + '_rerank_data_valid_' + fprefix_defi1) 
    args.outfile_valid_defi2 = os.path.join(args.outdir, args.dataset + '_rerank_data_valid_' + fprefix_defi2) 
    args.outfile_valid_candinfo = os.path.join(args.outdir, args.dataset + '_rerank_data_valid_' + fprefix_candinfo) 
    
    args.outfile_test = os.path.join(args.outdir, args.dataset + '_rerank_data_test_' + fprefix) 
    args.outfile_test_defi1 = os.path.join(args.outdir, args.dataset + '_rerank_data_test_' + fprefix_defi1) 
    args.outfile_test_defi2 = os.path.join(args.outdir, args.dataset + '_rerank_data_test_' + fprefix_defi2) 
    args.outfile_test_candinfo = os.path.join(args.outdir, args.dataset + '_rerank_data_test_' + fprefix_candinfo) 

    process_main(args)
    print("over")

 