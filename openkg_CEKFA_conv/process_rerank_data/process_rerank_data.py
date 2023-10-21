import os
import json
import argparse
import string
import numpy as np
import random
import pdb

# 加载第一阶段的模型, 预测出每个训练样本和测试样本，存储该阶段的排序
# 如果 rank 不在前10，则直接用于最后计算重排序后的各项指标
# 如果 rank 在前10，则应该额外留下该部分数据前10个候选(需filter,不包括训练集里的作为候选)用于第二阶段的重排序，
#   对于构造大模型的训练数据，需要选择 rank1 这种数据(可能数据的数量会很少，但是质量得高啊，唉)，

# 直接用基础模型不加训练的输出推理的结果

# 添加 rank 阶段各个选项的分数, 注意 该阶段的分数是越小排序越高

PROMPT_INPUT_WITHOUT_KF = (
    "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. "
    "Combine what you know, output a ranking of these candidate answers.\n\n"
    "### Question: {question}\n\n{option_ans}\n\n### Response:"
)           # 不包括解释
PROMPT_INPUT_WITH_KF = (
    "Below is an instruction that describes a task, paired with a question and corresponding candidate answers. "
    "Combine what you know and the following knowledge, output a ranking of these candidate answers.\n\n"
    "### Supporting information: {known_facts}\n\n" 
    "### Question: {question}\n\n{option_ans}\n\n### Response:"
)           # 不包括解释

PROMPT_OUTPUT = {
    "t1": "The correct ranking of the candidate answers would be:\n{option_label_str}\n",
    "t2": "Ranking:\n{option_label_str}\n",
    "t3": "\n{option_label_str}\n",
}
OPTIONS_LETTER = string.ascii_uppercase + string.ascii_lowercase
print(OPTIONS_LETTER)

def procsss_rerank_data_test_KFs(preds_data_test, hr2kfs, K=10):
    rerank_data = []
    options = OPTIONS_LETTER[:K]
    for data in preds_data_test:
        if data['rank'] <= K:   # 仅重排序这部分结果
            head_id, rel_id, tail_id = data['trip_id']
            head, rel, tail = data['trip_str']
            if rel.startswith("inverse of "):
                question = " ".join(["what", rel.split("inverse of ")[-1], head, "?"])
            else:
                question = " ".join([head, rel, "what ?"]) 

            option_ans = []
            cand_info = []
            for option, cand in zip(options, data['cands']):
                cand_id, cand_str, cand_score = cand
                option_ans.append(option + ". " + cand_str)
                cand_info.append((cand_str, cand_id))
            option_ans = "\n".join(option_ans)
            
            if hr2kfs is not None:
                kf = hr2kfs[(head_id, rel_id)] if (head_id, rel_id) in hr2kfs else None
                if kf is not None:
                    data_input = PROMPT_INPUT_WITH_KF.format_map({"known_facts":kf, "question": question, "option_ans": option_ans})
                else:
                    raise Exception
            else:
                data_input = PROMPT_INPUT_WITHOUT_KF.format_map({"question": question, "option_ans": option_ans})

            data_ = {
                "input":data_input,
                "trip_id":data["trip_id"],
                "trip_str":data["trip_str"],
                "cand_info":cand_info,
            }
            rerank_data.append(data_)
    return rerank_data

    
def procsss_rerank_data_train_KFs_rankscore(preds_data_train, hr2kfs, K=10, out_template='t1', filter_no_candi=False):
    # 使用测试数据的目标实体所在的cluster的所有t作为正确的候选
    # 排除掉候选中其他正确cluster对应的实体，剩下的实体作为候选
    missing_num_options_e0 = 0
    missing_num_options_ltK = 0

    rerank_data = []
    options = OPTIONS_LETTER[:K]
    for data in preds_data_train:
        head_id, rel_id, tail_id = data['trip_id']
        head, rel, tail = data['trip_str']
        if rel.startswith("inverse of "):
            question = " ".join(["what", rel.split("inverse of ")[-1], head, "?"])
        else:
            question = " ".join([head, rel, "what ?"]) 
        
        # 训练集, 只有三元组对应的尾实体作为正确的，不能用同一个clust的实体(因为要么使用并不准确的cesi_clust, 要么不能用gold_clust)
        option_ans = [[0, tail, cand[2], tail_id] for cand in data['cands'] if cand[1] == tail]
        if len(option_ans) == 0:
            option_ans = [[0, tail, 100000, tail_id]]   # 先随机赋值一个很大的分数
        elif len(option_ans) > 1:
            raise Exception
        idx = 1 
         
        target_ents_id = [item[0] for item in data['target_ents']]
        target_filter_entids = data['target_filter_entids']
        filter_set = set(target_ents_id) | set(target_filter_entids)
        for cand in data['cands']:
            cand_id, cand_str, cand_score = cand
            if cand_id not in filter_set:
                option_ans.append([idx, cand_str, cand_score, cand_id])
                idx += 1
            if idx == K:
                break
        
        if len(option_ans) == 1:# 有可能训练集存的候选都是要filter的,所以没有目标?
            missing_num_options_e0 += 1
            if filter_no_candi:
                continue
            else:
                option_ans[0][2] = 1    # 这个值无所谓，因为在计算的时候因为只有这一个选项
        elif len(option_ans) < K:
            missing_num_options_ltK += 1
        
        # pdb.set_trace()

        # 如果目标答案的分数不是最好的，随机改为最好的用于排序分数的训练
        if len(option_ans) > 1 and option_ans[0][2] > option_ans[1][2]:     # 第一阶段的分数是越小越好
            low = option_ans[1][2] - (option_ans[0][2] - option_ans[1][2])
            option_ans[0][2] = random.uniform(low, option_ans[1][2])         # 不从0开始随机取分数了,怕和后面的分数分布差距太大! 而且从0开始不对（候选分数是负的时算出来的目标分数不是最小的）
        elif len(option_ans) > 1 and option_ans[0][2] == option_ans[1][2]:     # 第一阶段的分数是越小越好
            if len(option_ans) > 2:
                low = option_ans[1][2] - abs(option_ans[2][2] - option_ans[1][2])
                option_ans[0][2] = random.uniform(low, option_ans[1][2])
            else:
                raise Exception

        # 打乱列表顺序
        option_ans_random = list(option_ans)
        random.shuffle(option_ans_random)

        option_ans_input = []
        option_scores = []
        for option, ans in zip(options, option_ans_random):
            option_ans_input.append(option + ". " + ans[1])
            option_scores.append(ans[2])
        option_ans_input_str = "\n".join(option_ans_input)

        indexes = [option_ans_random.index(item) for item in option_ans]
        
        option_label = [options[idx_random] + ". " for i, idx_random in enumerate(indexes)]
        option_label_str = "\n".join(option_label)
        
        if hr2kfs is not None:
            kf = hr2kfs[(head_id, rel_id)] if (head_id, rel_id) in hr2kfs else None
            if kf is not None:
                data_input = PROMPT_INPUT_WITH_KF.format_map({"known_facts":kf, "question": question, "option_ans": option_ans_input_str})
            else:
                raise Exception
        else:
            data_input = PROMPT_INPUT_WITHOUT_KF.format_map({"question": question, "option_ans": option_ans_input_str})
        

        option_label_str = PROMPT_OUTPUT[out_template].format_map({"option_label_str":option_label_str})
        # pdb.set_trace()
        cand_info = [(item[1], item[3]) for item in option_ans]
        data_ = {
            "input":data_input,
            "output":option_label_str,
            "trip_id":data["trip_id"],
            "trip_str":data["trip_str"],
            "option_scores":option_scores, # 按照 选项A-
            "cand_info":cand_info,
        }
        rerank_data.append(data_)
    print(missing_num_options_e0)
    print(missing_num_options_ltK)
    return rerank_data


def read_knownfacts(filepath):
    hr2kfs = {}
    for line in open(filepath,'r'):
        line = line.strip().split("\t")
        hr2kfs[(int(line[0]), int(line[1]))] = line[2]
    return hr2kfs

def read_preds(filepath):
    preds_data = []
    max_cand_score = []
    for line in open(filepath,'r'):
        line = line.strip()
        line = json.loads(line) 
        preds_data.append(line)
        max_cand_score.extend([cand[2] for cand in line["cands"]])
    print("max_cand_score:", max(max_cand_score))
    return preds_data

def write_data(rerank_data, outfile):
    rerank_data_str = json.dumps(rerank_data, indent=4)
    with open(outfile, 'w') as fw:
        fw.write(rerank_data_str)
    return 


def process_main(args):
    hr2kfs_test = read_knownfacts(args.knownfacts_path_test)
    
    # test 文件是同时输出 无kf 和 有kf 的版本
    preds_data_test = read_preds(args.preds_path_json_test)
    print("writing file:", args.outfile_test)
    rerank_data_test = procsss_rerank_data_test_KFs(preds_data_test, None, args.rerank_Top_K)
    write_data(rerank_data_test, args.outfile_test)
    print("writing file:", args.outfile_test_kf)
    rerank_data_test_kf = procsss_rerank_data_test_KFs(preds_data_test, hr2kfs_test, args.rerank_Top_K)
    write_data(rerank_data_test_kf, args.outfile_test_kf)
    
    # train/valid 文件是 只输出 无kf 或 有kf 二选一版本
    preds_data_train = read_preds(args.preds_path_json_train)
    preds_data_valid = read_preds(args.preds_path_json_valid)
    if args.add_train_knownfacts:
        hr2kfs_train = read_knownfacts(args.knownfacts_path_train)
        hr2kfs_valid = read_knownfacts(args.knownfacts_path_valid)

        print("writing file:", args.outfile_train_kf)
        rerank_data_train_kf = procsss_rerank_data_train_KFs_rankscore(preds_data_train, hr2kfs_train, args.rerank_Top_K, args.out_template, args.filter_no_candi)
        write_data(rerank_data_train_kf, args.outfile_train_kf)

        print("writing file:", args.outfile_valid_kf)
        rerank_data_valid_kf = procsss_rerank_data_train_KFs_rankscore(preds_data_valid, hr2kfs_valid, args.rerank_Top_K, args.out_template, args.filter_no_candi)
        write_data(rerank_data_valid_kf, args.outfile_valid_kf)
    else:
        print("writing file:", args.outfile_train)
        rerank_data_train = procsss_rerank_data_train_KFs_rankscore(preds_data_train, None, args.rerank_Top_K, args.out_template, args.filter_no_candi)
        write_data(rerank_data_train, args.outfile_train)

        print("writing file:", args.outfile_valid)
        rerank_data_valid = procsss_rerank_data_train_KFs_rankscore(preds_data_valid, None, args.rerank_Top_K, args.out_template, args.filter_no_candi)
        write_data(rerank_data_valid, args.outfile_valid)

    return 



    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process_rerank_data for OpenKG')
    parser.add_argument('--data_path', dest='data_path', default='./dataset/', help='directory path of KG datasets')        
    parser.add_argument('--dataset', dest='dataset', default='ReVerb20K', help='Dataset Choice')
    parser.add_argument('--saved_model_name_rank', dest='saved_model_name_rank', default='', help='')       
    parser.add_argument('--rerank_Top_K', dest='rerank_Top_K', default=10, type=int, help='rerank top K predicted candidates.') 
    parser.add_argument('--out_template', dest='out_template', default='t2', help='')  
    
    parser.add_argument('--knownfacts_dir', dest='knownfacts_dir', default='./files_supporting/ReVerb20K/canonical_triples/ambv2/', help='')  
    parser.add_argument('--knownfacts_num', dest='knownfacts_num', default=3, type=int, help='') 

    parser.add_argument('--add_train_knownfacts',   dest='add_train_knownfacts',   default=False,  action='store_true', help='')
    parser.add_argument('--filter_no_candi',   dest='filter_no_candi',   default=False,  action='store_true', help='')

    args = parser.parse_args()

    topk = 1000 if args.dataset == 'ReVerb20K' else 1000
    args.ranks_path_test = os.path.join(args.saved_model_name_rank, 'ranks_test.npy')
    args.preds_path_json_train = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_train.json')     
    args.preds_path_json_valid = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_valid.json')
    args.preds_path_json_test = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_test.json')
    args.data_path = os.path.join(args.data_path, args.dataset) 
    args.data_files = {
        'ent2id_path'       : args.data_path + '/ent2id.txt',
        'rel2id_path'       : args.data_path + '/rel2id.txt',
    }

    filter_flag = '_filter' if args.filter_no_candi else ""
    fprefix = "rs_%s_Top%d%s.json" % (args.out_template, args.rerank_Top_K, filter_flag)
    fprefix_kf = "rs_%s_Top%d_kf%d%s.json" % (args.out_template, args.rerank_Top_K, args.knownfacts_num, filter_flag)
    args.outdir = os.path.join(args.saved_model_name_rank, "rerank_data_new")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.knownfacts_path_test = os.path.join(args.knownfacts_dir, 'retrieval_v2_test_Top'+str(args.knownfacts_num) +'.txt')
    args.outfile_test = os.path.join(args.outdir, args.dataset + '_rerank_data_test_' + fprefix) 
    args.outfile_test_kf = os.path.join(args.outdir, args.dataset + '_rerank_data_test_' + fprefix_kf) 
    
    args.outfile_train = os.path.join(args.outdir, args.dataset + '_rerank_data_train_' + fprefix) 
    args.outfile_valid = os.path.join(args.outdir, args.dataset + '_rerank_data_valid_' + fprefix) 
    if args.add_train_knownfacts: 
        args.knownfacts_path_train = os.path.join(args.knownfacts_dir, 'retrieval_v2_train_Top'+str(args.knownfacts_num) +'.txt')
        args.knownfacts_path_valid = os.path.join(args.knownfacts_dir, 'retrieval_v2_valid_Top'+str(args.knownfacts_num) +'.txt')
        args.outfile_train_kf = os.path.join(args.outdir, args.dataset + '_rerank_data_train_' + fprefix_kf) 
        args.outfile_valid_kf = os.path.join(args.outdir, args.dataset + '_rerank_data_valid_' + fprefix_kf) 
    
    process_main(args)
    print("over")

 