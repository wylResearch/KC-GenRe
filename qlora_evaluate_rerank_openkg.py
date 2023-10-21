import os
import re
import json
import argparse
import string
import numpy as np
import random
from collections import OrderedDict, defaultdict
from CEKFA_conv.read_data_utils import * 
import pdb




OPTIONS_LETTER = string.ascii_uppercase + string.ascii_lowercase


def process_gen_out_options(output):        
    pattern = r'([A-Z]|[a-z])\.'    

    matches = re.findall(pattern, output)

    options = [match.strip().split('.')[0] for match in matches]

    return options


def get_gen_out_rerank(test_data_rank_preds_path, test_data_rerank_inp_path, test_data_rerank_gen_path, K):
    """
    """
    test_data_preds_rank_all = OrderedDict()
    for line in open(test_data_rank_preds_path, 'r'):
        line = json.loads(line)
        test_data_preds_rank_all[tuple(line["trip_id"])] = line
    all_indices_set = set(range(len(line['cands'])))
    test_data_in = json.load(open(test_data_rerank_inp_path, 'r'))
    test_data_gen = []
    for line in open(test_data_rerank_gen_path, 'r'):
        test_data_gen.append(json.loads(line))

    letter_to_index = {letter: i for i, letter in enumerate(OPTIONS_LETTER)}

    test_data_preds_rerank = OrderedDict()
    missing_num_allop = 0
    missing_num_someop = 0
    test_data_gen_dict = {}
    for i in range(len(test_data_in)):
        input = test_data_in[i]["input"]
        test_triple = tuple(test_data_in[i]["trip_id"])
        test_data_preds = test_data_preds_rank_all[test_triple]

        cands = test_data_preds['cands'] 
        
        answers = test_data_gen[i]["prediction_with_input"].split("Response:")[-1]
        answers_option_list = process_gen_out_options(answers)
        answers_indices = [letter_to_index[letter] for letter in answers_option_list]
        # pdb.set_trace()

        seen = set()
        answers_indices_ = [x for x in answers_indices if x < len(cands) and x not in seen and not seen.add(x)]
        
        if len(answers_indices_) == 0:   
            test_data_preds_rerank[test_triple] = -1
            missing_num_allop += 1
            # print(test_data_gen[i]["prediction_with_input"])
            # print('---')
        else:
            if len(answers_indices_) < K:  
                missing_indices = all_indices_set - set(answers_indices_)
                for idx in missing_indices:
                    answers_indices_.append(idx)
                missing_num_someop += 1
                # print(test_data_gen[i]["prediction_with_input"])
                # print('---')
            answer_ents = [cands[idx] for idx in answers_indices_]
            answer_ents = [item[0] for item in answer_ents]
            test_data_preds_rerank[test_triple]=answer_ents
        test_data_gen_dict[test_triple]= test_data_gen[i]    
        
    print("missing_num (missing some options(no valid option)):", missing_num_allop)
    print("missing_num (missing some options):", missing_num_someop)

    return test_data_preds_rank_all, test_data_preds_rerank, test_data_gen_dict, test_data_gen

       
        
def get_rerank(scores_idx, clust, entid2clustid, filter_clustID, candidates, K=10):
    """
    """    
    rank = 1
    high_rank_clust = set()
    for i in range(len(scores_idx)):        # num_nodes
        entid = scores_idx[i]
        if entid not in candidates:
            continue
        if entid in clust:              
            break
        else:           
            if entid2clustid[entid] not in high_rank_clust and entid2clustid[entid] not in filter_clustID:
                rank+=1
                high_rank_clust.add(entid2clustid[entid])

    top_cands_ent = []
    for entid in scores_idx:
        if entid not in candidates:
            continue
        else:
            top_cands_ent.append(entid)
        if len(top_cands_ent) >= K:
            break
    return rank, top_cands_ent



def evaluate_perf_rerank(test_data_preds_rank_all, test_data_preds_rerank, true_clusts, entid2clustid_gold, label_filter, id2ent, K=10, raw_flag=False, test_data_gen_dict=None):
    """
    """
    Hits = [1,3,10,20,30,50]
    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(Hits)))  
    
    ranked_cands = []
    candidates = set(id2ent.keys())

    ranks_raw = []
    ranks_new = []
    write_case = []
    
    for j, test_data_preds in enumerate(test_data_preds_rank_all.values()):
        test_triple = test_data_preds["trip_id"]
        head, rel, tail = test_triple

        rank_j = test_data_preds['rank']
        cur_ranked_cands = [item[0] for item in test_data_preds['cands']]
        
        if not raw_flag:
            if test_data_preds['rank'] <= K: # rerank
                t_clust = set(true_clusts[tail])             
                _filter = []
                if (head,rel) in label_filter:              
                    _filter = label_filter[(head,rel)]      
                
                # import pdb
                # pdb.set_trace()
                sample_sorts = test_data_preds_rerank[tuple(test_triple)]
                if sample_sorts != -1:      
                    rank_j, cur_ranked_cands = get_rerank(sample_sorts, t_clust, entid2clustid_gold, _filter, candidates, K)
                    ranks_raw.append(test_data_preds['rank'])
                    ranks_new.append(rank_j)
                    
                    if ranks_new[-1] < ranks_raw[-1] and ranks_new[-1]==1:
                        # print(ranks_raw[-1], ranks_new[-1])
                        # print(test_data_gen_dict[tuple(test_triple)])
                        write_case.append({'test_triple':tuple(test_triple),'ranks_raw':ranks_raw[-1],'ranks_new':ranks_new[-1],'preds':test_data_gen_dict[tuple(test_triple)]})
                        # pdb.set_trace()

        hits_j = np.ones((len(Hits)))
        for i,r in enumerate(Hits):
            if rank_j>r:
                hits_j[i]=0
            else:
                break
        if j%2==1:
            H_Rank.append(rank_j)
            H_inv_Rank.append(1/rank_j)
            H_Hits += hits_j
        else:
            T_Rank.append(rank_j)
            T_inv_Rank.append(1/rank_j)
            T_Hits += hits_j
            
        ranked_cands.append(cur_ranked_cands)
    mean_rank_head = np.mean(np.array(H_Rank))
    mean_rank_tail = np.mean(np.array(T_Rank))
    mean_rank = 0.5*(mean_rank_head+mean_rank_tail) 
    mean_inv_rank_head = np.mean(np.array(H_inv_Rank))
    mean_inv_rank_tail = np.mean(np.array(T_inv_Rank))
    mean_inv_rank = 0.5*(mean_inv_rank_head+mean_inv_rank_tail)
    hits_at_head = {}
    hits_at_tail = {}
    hits_at = {}
    for i, hits in enumerate(Hits):        # [1,3,10,30,50]
        hits_at_head[hits] = H_Hits[i]/len(H_Rank)
        hits_at_tail[hits] = T_Hits[i]/len(T_Rank)
        hits_at[hits] = 0.5*(hits_at_head[hits]+hits_at_tail[hits])
    perf = {'mr': mean_rank,
            'mrr': mean_inv_rank,
            'hits@': hits_at,
            'head_mr': mean_rank_head,
            'head_mrr': mean_inv_rank_head,
            'head_hits@': hits_at_head,
            'tail_mr': mean_rank_tail,
            'tail_mrr': mean_inv_rank_tail,
            'tail_hits@': hits_at_tail,
            }
    if len(ranks_raw) > 0 and len(ranks_new)>0:
        print('rerank_valid_num', len(ranks_raw))
        print(sum(ranks_raw)/len(ranks_raw), sum(ranks_new)/len(ranks_new))

    with open('case_study.json', 'w') as fout:
        json.dump(write_case, fout, indent=4)
    return perf, {'tail':T_Rank, 'head':H_Rank}, ranked_cands, ranks_new






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process_rerank_data for OpenKG')
    parser.add_argument('--data_path', dest='data_path', default='./dataset/', help='directory path of KG datasets')        # TODO  ./
    parser.add_argument('--dataset', dest='dataset', default='ReVerb20K', help='Dataset Choice')
    parser.add_argument("--train_proportion", type=float, default=1.0)
    parser.add_argument('--reverse',     dest='reverse',     default=True,  action='store_true', help='whether to add inverse relation edges')
    parser.add_argument('--test_data_rank_preds_path', dest='test_data_rank_preds_path', default='./dataset/ReVerb20K_preds_TopK_test.json', help='data path of test data')        
    parser.add_argument('--test_data_rerank_inp_path', dest='test_data_rerank_inp_path', default='./dataset/ReVerb20K_rerank_data_test_v1.json', help='data path of test data')        
    parser.add_argument('--test_data_rerank_gen_path', dest='test_data_rerank_gen_path', default='./output/ReVerb20K-7b-codev1-datav1/predictions_sample_p09.jsonl', help='data path of the generated output for test data')        
    parser.add_argument('--rerank_Top_K', dest='rerank_Top_K', default=10, type=int, help='rerank top K predicted candidates.') 
    parser.add_argument('--raw_flag',     dest='raw_flag',     default=False,  action='store_true', help='')


    args = parser.parse_args()
    args.data_path = os.path.join(args.data_path, args.dataset) 
    train_file = '/train_trip_%.1f.txt'%args.train_proportion if args.train_proportion < 1 else '/train_trip.txt'

    # set data files
    args.data_files = {
        'ent2id_path'       : args.data_path + '/ent2id.txt',
        'rel2id_path'       : args.data_path + '/rel2id.txt',
        'train_trip_path'   : args.data_path + train_file,
        'test_trip_path'    : args.data_path + '/test_trip.txt',
        'valid_trip_path'   : args.data_path + '/valid_trip.txt',
        'gold_npclust_path' : args.data_path + '/gold_npclust.txt',
    }

    ent2id, id2ent = get_noun_phrases(args.data_files["ent2id_path"])
    rel2id, id2rel, inverse2rel_map = get_relation_phrases(args.data_files["rel2id_path"], args.reverse)
    true_clusts, entid2clustid_gold, unique_clusts_gold , num_groups_gold_np = get_clusters(args.data_files["gold_npclust_path"])
    train_trips, rel2id, label_graph, train_trips_without_rev, label_graph_other, label_filter = get_train_triples(
                                                                                        args.data_files["train_trip_path"],
                                                                                        entid2clustid_gold,rel2id,
                                                                                        id2rel, args.reverse)

    test_data_preds_rank_all, test_data_preds_rerank, test_data_gen_dict, _ = get_gen_out_rerank(args.test_data_rank_preds_path, args.test_data_rerank_inp_path, args.test_data_rerank_gen_path, args.rerank_Top_K)
    test_rerank_perf, test_rerank_ranks, test_rerank_cands, _ = evaluate_perf_rerank(test_data_preds_rank_all, test_data_preds_rerank, true_clusts, 
                                                                                  entid2clustid_gold, label_filter, id2ent, args.rerank_Top_K, args.raw_flag, test_data_gen_dict)

    out_path = os.path.join(os.path.dirname(args.test_data_rerank_gen_path), "rerank_perf.json")
    out_path = args.test_data_rerank_gen_path.split(".jsonl")[0] + "_rerank_perf.json"
    with open(out_path, 'w') as fout:
        json.dump(test_rerank_perf, fout)
    
    print("raw_flag%s\tMRR:%6f\tMR:%6f\thits@1:%6f\thits@3:%6f\thits@10:%6f\thits@20:%6f\thits@30:%6f\thits@50:%6f\n\n" % (args.raw_flag, test_rerank_perf["mrr"], test_rerank_perf["mr"], 
                                                                                                  test_rerank_perf["hits@"][1], test_rerank_perf["hits@"][3], 
                                                                                                  test_rerank_perf["hits@"][10], test_rerank_perf["hits@"][20], 
                                                                                                  test_rerank_perf["hits@"][30],test_rerank_perf["hits@"][50]))
    file_path = 'results_' + args.dataset +'.txt'
    with open(file_path, mode='a', encoding='utf-8') as file_obj:
        file_obj.write("%s\t\traw_flag%s\tMRR:%6f\tMR:%6f\thits@1:%6f\thits@3:%6f\thits@10:%6f\thits@20:%6f\thits@30:%6f\thits@50:%6f\n\n" % (args.test_data_rerank_gen_path, args.raw_flag, 
                                                                                                                   test_rerank_perf["mrr"], 
                                                                                                                   test_rerank_perf["mr"], 
                                                                                                                   test_rerank_perf["hits@"][1], 
                                                                                                                   test_rerank_perf["hits@"][3], 
                                                                                                                   test_rerank_perf["hits@"][10], 
                                                                                                                   test_rerank_perf["hits@"][20], 
                                                                                                                   test_rerank_perf["hits@"][30], 
                                                                                                                   test_rerank_perf["hits@"][50]))


