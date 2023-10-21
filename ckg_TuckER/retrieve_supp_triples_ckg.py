import os
import torch
import argparse
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import json
from collections import OrderedDict, defaultdict

import pdb
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      


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


def semantic_search(args, corpus_hrt_embeddings, corpus_hr_embeddings, preds_file, retrieval_Top_K, mode, bi_encoder, outfile):
    """
    Bi-Encoders (Retrieval) 
    produce embeddings independently for your paragraphs and for your search queries.
    如果输入的 queries 的 mode 是 'train', 而且 这些 queries 的数目和 corpus 是对应的，则不用输入 idx;
    否则若输入的 queries 的 mode 是 'train', 而且 其实只含有一个query,则需要输入其在 corpus 中对应的 idx
    """
    search_results = []
    preds_data = read_preds(preds_file)
    fjson = open(outfile, 'w')

    print("search and writing file:", outfile)
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    for i, data in enumerate(tqdm(preds_data)):
        if data['rank'] <= 50:
            head_id, rel_id, tail_id = data['trip_id']
            head, rel, tail = data['trip_str']
            head_name = args.ent2name[head]
            tail_name = args.ent2name[tail]

            if rel.endswith("_reverse"):
                rel = rel.split("_reverse")[0]
                template = args.rel2template[rel]
                question = template.replace("[Y]", head_name)  
                # question_hr = question.replace("[X]", "_____")  
                # question_hr = question_hr[:-2] + " ?"  # TODO
                reverse = True
            else:
                template = args.rel2template[rel]
                question = template.replace("[X]", head_name) 
                # question_hr = question.replace("[Y]", "_____")  
                # question_hr = question_hr[:-2] + " ?"  # TODO
                reverse = False

            ######### search for query-supported triples   
            query_hr = question
            query_hr_embedding = bi_encoder.encode(query_hr, convert_to_tensor=True)
            cos_scores_hr = util.cos_sim(query_hr_embedding, corpus_hr_embeddings)[0]
            if mode =='train':
                raise NotImplementedError
                cos_scores_hr[i] = -1 # TODO
                print(args.train_triples_hr_str[i])

            top_results_hr = torch.topk(cos_scores_hr, k=min(retrieval_Top_K, corpus_hr_embeddings.shape[0]))
            results_hr = []
            seen = set()
            for score, idx in zip(top_results_hr[0], top_results_hr[1]):
                triple_str_id = idx.item()
                triple_id = args.train_triples_hr_str2triple_id[triple_str_id]
                if triple_id not in seen:
                    top_j = {
                    'triple_id': triple_id, 
                    # 'triple_str': args.train_triples_str[triple_id], 
                    # 'triple_str_defi': args.train_triples_str_defi[triple_id], 
                    'score': score.item()
                    }
                    seen.add(triple_id)
                    results_hr.append(top_j)
            data['query_support_triples'] = results_hr


            ######### search for cand-supported triples        
            candi_info_dict = OrderedDict()
            for j, cand_info in enumerate(data['cands']):
                cand_id, cand, cand_score = cand_info
                cand_triples = args.ent2triplestr[cand]
                cand_name = args.ent2name[cand]
                if reverse:
                    query_cand = question.replace("[X]", cand_name)
                else:
                    query_cand = question.replace("[Y]", cand_name)
                query_cand_embedding = bi_encoder.encode(query_cand, convert_to_tensor=True)
                if mode!='train':
                    cand_triples_id = [item[0] for item in cand_triples]                    # 取全部
                    cand_triples_str = [item[1] for item in cand_triples]       
                    cand_triples_str_defi = [item[2] for item in cand_triples]       
                else:
                    raise NotImplementedError
                    cand_triples_id = [item[0] for item in cand_triples if i==item[0] or i-len(query_data)//2==item[0]]   # 不能取原始三元组
                    cand_triples_str = [item[1] for item in cand_triples if i==item[0] or i-len(query_data)//2==item[0]]
            
                # 从corpus中 取出cand对应的三元组的编码，然后与问题匹配 TODO 
                cand_triples_emb = corpus_hrt_embeddings[cand_triples_id]

                cos_scores = util.cos_sim(query_cand_embedding, cand_triples_emb)[0]
                top_results_cand = torch.topk(cos_scores, k=min(retrieval_Top_K, cand_triples_emb.shape[0]))
                results_cand_j = []
                seen = set()

                for score, idx in zip(top_results_cand[0], top_results_cand[1]):
                    triple_id = cand_triples_id[idx.item()]
                    if triple_id not in seen:
                        # triple_str = cand_triples_str[idx.item()]
                        # triple_str_defi = cand_triples_str_defi[idx.item()]
                        top_j = {
                        'triple_id': triple_id, 
                        # 'triple_str': args.train_triples_str[triple_id], 
                        # 'triple_str_defi': args.train_triples_str_defi[triple_id], 
                        'score': score.item()
                        }
                        seen.add(triple_id)
                        results_cand_j.append(top_j)
                candi_info_dict[cand] = {'id':cand_id, 'triples':results_cand_j}
            
            data['cand_info'] = candi_info_dict
            fjson.write("%s\n"%json.dumps(data))


    fjson.close() 

    return search_results




# def bm25_tokenizer(text):
#     # lower case text and remove stop-words from indexing
#     tokenized_doc = []
#     for token in text.lower().split():
#         token = token.strip(string.punctuation)

#         if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
#             tokenized_doc.append(token)
#     return tokenized_doc


# def get_corpus_bm25(corpus):
#     tokenized_corpus = []
#     for passage in tqdm(corpus):
#         tokenized_corpus.append(bm25_tokenizer(passage))
#     bm25 = BM25Okapi(tokenized_corpus)
#     return bm25


# def BM25_search(corpus_bm25, queries, retrieval_Top_K, mode):
#     """ (lexical search) """
#     search_results = []
#     for i, query in enumerate(queries):
#         bm25_scores = corpus_bm25.get_scores(bm25_tokenizer(query))
#         if mode == "train":
#             bm25_scores[i] = -1       
#         top_k = np.argpartition(bm25_scores, -retrieval_Top_K)[-retrieval_Top_K:]
#         bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_k]
#         bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
#         search_results.append(bm25_hits)
        
#         # print("Top-3 lexical search (BM25) hits")
#         # for hit in bm25_hits:
#         #     print("\t{:.3f}\t{}".format(hit['score'], corpus[hit['corpus_id']].replace("\n", " ")))
#     return search_results


def retrieval_triples_sbert(args):
    retrieval_Top_K = args.retrieval_Top_K                          # Number of passages we want to retrieve
    
    bi_encoder = SentenceTransformer(args.retrieval_model)
    bi_encoder.max_seq_length = 384     #Truncate long passages to 256 tokens
    
    corpus_hrt = args.train_triples_str     
    corpus_hrt_embeddings = bi_encoder.encode(corpus_hrt, show_progress_bar=True, convert_to_tensor=True)
    corpus_hr = args.train_triples_hr_str
    corpus_hr_embeddings = bi_encoder.encode(corpus_hr, show_progress_bar=True, convert_to_tensor=True)
    
    f_preds_results = [(args.preds_path_json_test, args.outfile_test), 
                        # (args.preds_path_json_train, args.outfile_train),  
                        # (args.preds_path_json_valid, args.outfile_valid)
                        ]
    modes = ['test', 
            #  'train', 'valid'
             ]
    for mode, (preds_file, outfile) in zip(modes, f_preds_results):        # TODO
        ##### Sematic Search for cand and query #####
        semantic_search(args, corpus_hrt_embeddings, corpus_hr_embeddings, preds_file, retrieval_Top_K, mode, bi_encoder, outfile)                             
        print("semantic_search over.")
    
    return


# def get_canonical_triples_BM25(args, basedata):
#     retrieval_Top_K = args.retrieval_Top_K                          # Number of passages we want to retrieve
#     sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)
    
#     corpus_trips = basedata.train_trips
#     corpus = get_queries(basedata, corpus_trips, sep_id_rp)
#     corpus_bm25 = get_corpus_bm25(corpus)
#     for mode in ['train', 'test', 'valid']:
#         if mode == 'train':
#             queries_trips = basedata.train_trips
#         elif mode == 'valid':
#             queries_trips = basedata.valid_trips
#         elif mode == 'test':
#             queries_trips = basedata.test_trips
        
#         queries = get_queries(basedata, queries_trips, sep_id_rp)
        
#         file_new = "retrieval_Top" + str(retrieval_Top_K)
#         file_new

#         print("start BM25_search")
#         dir_name = args.retrieval_dirname
#         search_results_bm = BM25_search(corpus_bm25, queries, retrieval_Top_K, mode)
#         print("BM25_search over.")
#         write_canonical_triples(args, basedata, args.dataset, corpus_trips, queries_trips, corpus, queries, search_results_bm, dir_name, file_name, sep_id_rp)
#         print("write BM25_search_results over.")
#     return
    


def read_ent2defi(f_ent2defi):
    ent2defi = defaultdict(str)
    for line in open(f_ent2defi):
        line = line.strip().split("\t")
        ent2defi[line[0]] = line[1]
    return ent2defi


def get_train_triples_str(f_triple, args):
    triples_hr_str = []
    triples_hr_str2triple_id = []
    triples_str = []
    triples_str_defi = []
    ent2triplestr = defaultdict(list)
    for i, line in enumerate(open(f_triple)):
        line = line.strip().split("\t")
        h, r, t = line
        h_name = args.ent2name[h]
        t_name = args.ent2name[t]
        template = args.rel2template[r]
        trip_str = template.replace("[X]", h_name).replace("[Y]", t_name)
        trip_str_defi = template.replace("[X]", h_name+" (" + args.ent2defi[h]+")").replace("[Y]", t_name+" (" + args.ent2defi[t]+")")
        triples_str.append(trip_str)
        triples_str_defi.append(trip_str_defi)
        ent2triplestr[h].append((i, trip_str, trip_str_defi))
        ent2triplestr[t].append((i, trip_str, trip_str_defi))

        # query  一正一反添加
        triples_hr_str.append(template.replace("[X]", h_name))
        triples_hr_str.append(template.replace("[Y]", t_name))
        triples_hr_str2triple_id.append(i)
        triples_hr_str2triple_id.append(i)
    return triples_str, triples_str_defi, ent2triplestr, triples_hr_str, triples_hr_str2triple_id




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieve support training triples for (h,r,t)')
    parser.add_argument('--data_path', dest='data_path', default='./dataset/', help='directory path of KG datasets')        
    parser.add_argument('--dataset', dest='dataset', default='Wiki27K', help='Dataset Choice')
    parser.add_argument('--saved_model_name_rank', dest='saved_model_name_rank', default='', help='')       
    parser.add_argument('--retrieval_model', dest='retrieval_model', default='/data/liqiwang/pretrained/all-mpnet-base-v2', help='')       
    
    parser.add_argument('--retrieval_Top_K', dest='retrieval_Top_K', default=5, type=int, help='') 
  
    args = parser.parse_args()
 
    # rerank_filedir = args.saved_model_name_rank + '/rerank_data_candi/' 
    # args.frerank_data_train = rerank_filedir + args.dataset + "_rerank_data_train_rs_t2_Top50.json"                
    # args.frerank_data_valid = rerank_filedir + args.dataset + "_rerank_data_valid_rs_t2_Top50.json"                
    # args.frerank_data_test = rerank_filedir + args.dataset + "_rerank_data_test_rs_t2_Top50.json"                                 

    topk = 50
    args.ranks_path_test = os.path.join(args.saved_model_name_rank, 'ranks_test.npy')
    args.preds_path_json_train = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_train.json')     
    args.preds_path_json_valid = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_valid.json')
    args.preds_path_json_test = os.path.join(args.saved_model_name_rank, 'preds_Top' + str(topk) + '_test.json')
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
    
    args.outfile_test = args.preds_path_json_test.replace(".json", "_retrieved.json")
    args.outfile_train = args.preds_path_json_train.replace(".json", "_retrieved.json")
    args.outfile_valid = args.preds_path_json_valid.replace(".json", "_retrieved.json")

    retrieval_triples_sbert(args)
    print("over")

# python retrieve_supp_triples_ckg.py --data_path ./dataset/ --dataset Wiki27K --saved_model_name_rank results/Wiki27K/tucker_256 
