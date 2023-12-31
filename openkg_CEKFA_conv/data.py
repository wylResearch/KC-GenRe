import os
import pathlib
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from collections import Counter
from read_data_utils import *
import logging


class load_basedata():
    def __init__(self, args):
        self.args = args
        self.data_files = self.args.data_files
        self.add_reverse = args.reverse

        self.fetch_data()
        
        self.num_nodes = len(self.ent2id)
        self.num_rels = len(self.rel2id)
        

    def read_canonical_rps(self):
        if self.args.rp_neigh_mth == 'Global':
            if "rp2neighs_scores" in self.data_files["rpneighs_filepath"]:
                self.rps2neighs = self.get_edges_rp_thresholdScore(self.data_files["rpneighs_filepath"]) 
            elif "rpneighs" in self.data_files["rpneighs_filepath"]:
                self.rps2neighs = self.get_edges_rp_maxNum(self.data_files["rpneighs_filepath"]) 
            else:
                self.rps2neighs = None
            logging.info("rps2neighs.shape: (" + str(self.rps2neighs.shape[0]) + "," + str(self.rps2neighs.shape[1]) + ").")
        elif self.args.rp_neigh_mth == 'Local':
            if "rp2neighs_scores" in self.data_files["rpneighs_filepath"]:
                self.rps2neighs = self.get_rps2neighs_score(self.data_files["rpneighs_filepath"])
            elif "rpneighs" in self.data_files["rpneighs_filepath"]:
                self.rps2neighs = self.get_rps2neighs(self.data_files["rpneighs_filepath"])
            else:
                self.rps2neighs = None
        else:
            self.rps2neighs = None


    def fetch_data(self):
        self.ent2id, self.id2ent = get_noun_phrases(self.data_files["ent2id_path"])
        self.rel2id, self.id2rel, self.inverse2rel_map = get_relation_phrases(self.data_files["rel2id_path"], self.add_reverse)
        
        # 第一项是{entid:同cluster的ents}     
        # 第二项是{entid:clustid}
        # 第三项存储簇信息，是个list，如unique_clusts_cesi[i]相当于存储 clustid 为 i 的 ents
        self.canon_clusts, self.entid2clustid_cesi, self.unique_clusts_cesi, self.num_groups_cesi_np = get_clusters(self.data_files["cesi_npclust_path"])
        self.true_clusts, self.entid2clustid_gold, self.unique_clusts_gold , self.num_groups_gold_np = get_clusters(self.data_files["gold_npclust_path"])

        self.edges = self.get_edges_np(self.canon_clusts)    # 更新 self.edges, shape为 (2, X) 其中X为规范化边的数目

        self.train_trips,self.rel2id,self.label_graph, self.train_trips_without_rev, label_graph_other, self.label_filter = get_train_triples(
                                                                                            self.data_files["train_trip_path"],
                                                                                            self.entid2clustid_gold,self.rel2id,
                                                                                            self.id2rel, self.add_reverse)
        self.label_graph_rt2h, self.label_graph_ht2r, self.label_graph_r2ht = label_graph_other

        self.test_trips, self.test_trips_without_rev = get_test_triples(self.data_files["test_trip_path"],self.rel2id, self.id2rel, self.add_reverse)
        self.valid_trips, self.valid_trips_without_rev = get_test_triples(self.data_files["valid_trip_path"],self.rel2id, self.id2rel, self.add_reverse)

    
    def get_edges_np(self, content_clusts):     # 规范化边
        head_list = []
        tail_list = []
        for content in content_clusts:
            if len(content_clusts[content])==1: # 簇里只有自己，则加上自己
                head_list.append(content)
                tail_list.append(content)
            for _, neigh in enumerate(content_clusts[content]):           # 簇里还有邻居，则不加自己
                if neigh!=content:           
                    head_list.append(neigh)
                    tail_list.append(content)   

        head_list = np.array(head_list).reshape(1,-1)
        tail_list = np.array(tail_list).reshape(1,-1)

        edges = np.concatenate((np.array(head_list),np.array(tail_list)),axis = 0)
        return edges
    
    def get_edges_rp_maxNum(self, clust_path):
        head_list = []
        tail_list = []
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            content, neigh_num = int(line[0]), int(line[1])
            if neigh_num == 0:
                head_list.append(content)
                tail_list.append(content)
            else:
                clust = [int(content) for content in line[2:]]  
                for i, neigh in enumerate(clust):                      
                    if i == self.args.maxnum_rpneighs:
                        break
                    elif i < self.args.maxnum_rpneighs:            # 限制上限
                        head_list.append(neigh)
                        tail_list.append(content)   

        head_list = np.array(head_list).reshape(1,-1)
        tail_list = np.array(tail_list).reshape(1,-1)
        edges = np.concatenate((np.array(head_list),np.array(tail_list)),axis = 0)
        return edges
    
    def get_edges_rp_thresholdScore(self, clust_path):
        head_list = []
        tail_list = []
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split(":")
            content = int(line[0].split("\t")[0])
            neighs = line[1].split(";")
            neigh_num = len(neighs)
            if neigh_num == 0:
                head_list.append(content)
                tail_list.append(content)
            else:
                for i, neigh in enumerate(neighs): 
                    _, neigh_id, neigh_str, neigh_score = neigh.split("\t")
                    if float(neigh_score) < self.args.thresholds_rp:            # 设置分数阈值 默认0.8
                        break
                    else:            
                        head_list.append(int(neigh_id))
                        tail_list.append(content)                
        head_list = np.array(head_list).reshape(1,-1)
        tail_list = np.array(tail_list).reshape(1,-1)
        edges = np.concatenate((np.array(head_list),np.array(tail_list)),axis = 0)
        return edges
 
    def get_rps2neighs(self, clust_path):
        self.rel2id["PADRP"] = self.num_rels
        self.id2rel[self.num_rels] = "PADRP"
        pad_neigh_id = self.num_rels
        self.inverse2rel_map[pad_neigh_id] = pad_neigh_id
        self.num_rels += 1
        rps2neighs = {}
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            content, neigh_num = int(line[0]), int(line[1])
            neighs = []
            if neigh_num != 0:
                clust = [int(content) for content in line[2:]]  
                neighs = clust[:min(len(clust), self.args.maxnum_rpneighs)]
            if len(neighs) < self.args.maxnum_rpneighs:
                neighs += [pad_neigh_id] * (self.args.maxnum_rpneighs - len(neighs))
            rps2neighs[content] = neighs
        return rps2neighs
    
    def get_rps2neighs_score(self, clust_path):
        self.rel2id["PADRP"] = self.num_rels
        self.id2rel[self.num_rels] = "PADRP"
        pad_neigh_id = self.num_rels
        self.inverse2rel_map[pad_neigh_id] = pad_neigh_id
        self.num_rels += 1
        rps2neighs = {}
        f = open(clust_path,"r").readlines()
        for line_ in f:
            line = line_.strip().split(":")       
            if len(line)!=2:
                line_ = line_.replace("n :o", "n o")
                line = line_.strip().split(":")
            if len(line)!=2:
                print('---',line_)
                raise Exception
            content = int(line[0].split("\t")[0])
            neighs = line[-1].split(";")
            neigh_num = len(neighs)

            rp_neighs = []
            if neigh_num != 0:
                for i, neigh in enumerate(neighs): 
                    _, neigh_id, neigh_str, neigh_score = neigh.split("\t")
                    if float(neigh_score) < self.args.thresholds_rp:            # 设置分数阈值 默认0.8
                        break
                    elif len(rp_neighs)==self.args.maxnum_rpneighs:            
                        break
                    else:
                        rp_neighs.append(int(neigh_id))
            if len(rp_neighs) < self.args.maxnum_rpneighs:
                rp_neighs += [pad_neigh_id] * (self.args.maxnum_rpneighs - len(rp_neighs))
            rps2neighs[content] = rp_neighs
        return rps2neighs






    ######################################################################################################################
    ######################################################################################################################


class myDataset(Dataset):
    def __init__(self, args, basedata, split, eval_train=False):
        self.args = args
        self.basedata = basedata
        self.split = split 
        if not eval_train:
            self.train_pairs = [[pair[0], pair[1]] for pair in list(basedata.label_graph.keys())] # key是turple
        else:
            self.train_pairs = [[pair[0], pair[1]] for pair in basedata.train_trips]
        self.test_pairs = [[pair[0], pair[1]] for pair in basedata.test_trips]
        self.valid_pairs = [[pair[0], pair[1]] for pair in basedata.valid_trips]
        
        if args.rp_neigh_mth == 'Global':
            self.edges_rp = torch.tensor(basedata.rps2neighs, dtype=torch.long)     
            self.rel_id = torch.arange(0, basedata.num_rels, dtype=torch.long)

    def __getitem__(self, index):
        if self.split == "train":
            return self.train_pairs[index]
        elif self.split == "test":
            return self.test_pairs[index]
        elif self.split == "valid":
            return self.valid_pairs[index]

    def __len__(self):
        if self.split == "train":
            return len(self.train_pairs)
        elif self.split == "test":
            return len(self.test_pairs)
        elif self.split == "valid":
            return len(self.valid_pairs)

     
    def get_batch_train(self, raw_batch):
        labels = np.zeros((len(raw_batch), self.basedata.num_nodes))
        for i in range(len(raw_batch)):             
            h_id, r_id = raw_batch[i]    
            pos_t_ids = list(self.basedata.label_graph[(h_id, r_id)])
            labels[i][pos_t_ids] = 1    
        pairs_id = torch.LongTensor(raw_batch)
        labels = torch.FloatTensor(labels)   
        return {'pairs_id':pairs_id,        
                'labels':labels}
    
    def get_batch_test(self, raw_batch): 
        return {'pairs_id':torch.LongTensor(raw_batch)}

    def get_batch_Global_rps_train(self, raw_batch):
        labels = np.zeros((len(raw_batch), self.basedata.num_nodes))
        for i in range(len(raw_batch)):             
            h_id, r_id = raw_batch[i]    
            pos_t_ids = list(self.basedata.label_graph[(h_id, r_id)])
            labels[i][pos_t_ids] = 1    
        pairs_id = torch.LongTensor(raw_batch)
        labels = torch.FloatTensor(labels)   
        return {'pairs_id':pairs_id,        
                'labels':labels,
                'edges_rp':self.edges_rp,
                'rel_id':self.rel_id,
                }
    
    def get_batch_Global_rps_test(self, raw_batch): 
        return {'pairs_id':torch.LongTensor(raw_batch),
                'edges_rp':self.edges_rp,
                'rel_id':self.rel_id,
                }

    def get_batch_Local_rps_train(self, raw_batch): 
        labels = np.zeros((len(raw_batch), self.basedata.num_nodes))
        neigh_rels_id = []

        for i in range(len(raw_batch)):             
            h_id, r_id = raw_batch[i]    
            pos_t_ids = list(self.basedata.label_graph[(h_id, r_id)])
            labels[i][pos_t_ids] = 1 

            if r_id in self.basedata.rps2neighs:
                neighs = self.basedata.rps2neighs[r_id]
            else:
                neighs_rev = self.basedata.rps2neighs[self.basedata.inverse2rel_map[r_id]]
                neighs = [self.basedata.inverse2rel_map[n_rev] for n_rev in neighs_rev]
            neigh_rels_id.append(neighs)
                    
        pairs_id = torch.LongTensor(raw_batch)
        neigh_rels_id = torch.LongTensor(neigh_rels_id)
        labels = torch.FloatTensor(labels)

        return {'pairs_id':pairs_id,          
                'neigh_rels_id':neigh_rels_id,
                'labels':labels,
                }

    def get_batch_Local_rps_test(self, raw_batch): 
        neigh_rels_id = []
        for i in range(len(raw_batch)):             
            h_id, r_id = raw_batch[i]    
            if r_id in self.basedata.rps2neighs:
                neighs = self.basedata.rps2neighs[r_id]
            else:
                neighs_rev = self.basedata.rps2neighs[self.basedata.inverse2rel_map[r_id]]
                neighs = [self.basedata.inverse2rel_map[n_rev] for n_rev in neighs_rev]
            neigh_rels_id.append(neighs)
        pairs_id = torch.LongTensor(raw_batch)
        neigh_rels_id = torch.LongTensor(neigh_rels_id)
        return {'pairs_id':pairs_id,          
                'neigh_rels_id':neigh_rels_id,
                }

      
class DatasetFromScores_Retrieval_ReRank(Dataset):
    """ only for test data """
    def __init__(self, args, basedata, scores, trips, data1=None):
        if data1 != None:   # v1 或者 v2 类型的retrieval数据
            self.retrieval_data = data1 
            assert scores.shape[0] == trips.shape[0] #== len(self.retrieval_data)
        else:
            self.retrieval_data = None
        
        self.args = args
        self.basedata = basedata
        self.num_nodes = basedata.num_nodes
        self.trips = trips
        self.scores = scores
        self.sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)
        self.SEP = "[SEP]"

    
    def __getitem__(self, index):
        if self.retrieval_data != None:
            return self.trips[index], self.scores[index], self.retrieval_data[index]
        else:
            raise Exception

        
    def __len__(self):
        return len(self.trips)
    
    def get_batch_retrival_data(self, rawbatch):      
        for i in range(len(rawbatch)):   
            test_trips, test_scores, retrieval_data = rawbatch[i]
            h, r, t = test_trips[0], test_trips[1], test_trips[2]       
            test_scores_sorted, test_sorts_indices = torch.sort(torch.tensor(test_scores), dim=0, descending=True)
            candi_TopK = test_sorts_indices[:self.args.rerank_Top_K]
            
            rerank_input =[]
            if r < self.sep_id_rp:
                for candi_t in candi_TopK.tolist():
                    rerank_input.append([self.basedata.id2ent[h] + ' ' + self.basedata.id2rel[r] + ' '+ self.basedata.id2ent[candi_t], retrieval_data[2]])
            else:
                for candi_t in candi_TopK.tolist():
                    r_inv = self.basedata.inverse2rel_map[r]        
                    rerank_input.append([self.basedata.id2ent[candi_t] + ' ' + self.basedata.id2rel[r_inv] + ' '+ self.basedata.id2ent[h], retrieval_data[2]])
    
        hr = torch.LongTensor([h,r]).unsqueeze(0).repeat(self.args.rerank_Top_K,1)
        t = candi_TopK.long().unsqueeze(1)
        triples_id = torch.cat([hr,t],dim=1)
        return {'triples_id':triples_id, 
                'test_scores_sorted': test_scores_sorted,
                'test_sorts_indices':test_sorts_indices,   
                'rerank_input':rerank_input,   
                }
    

class DatasetFromScores_ReRank_Bert(Dataset):
    def __init__(self, args, basedata, scores, trips, mode):
        self.args = args
        self.basedata = basedata
        self.num_nodes = basedata.num_nodes
        self.trips = trips
        self.scores = scores
        self.mode = mode
        self.sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)
        if mode != 'test':
            self.rerank_inputs, self.labels = self.process_train_valid()

        if args.bert_model_dir != None:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)

        self.SEP = "[SEP]"
        self.CLS_ID = self.tokenizer.cls_token_id
        self.SEP_ID = self.tokenizer.sep_token_id
        self.PAD_ID = self.tokenizer.pad_token_id
        
    def __getitem__(self, index):    
        if self.mode == 'test':
            return self.trips[index], self.scores[index]
        else:
            return self.rerank_inputs[index], self.labels[index]

    def __len__(self):
        if self.mode == 'test':
            return len(self.trips)
        else:
            return len(self.rerank_inputs)
    
    def get_triple_string(self, h, r, t):
        if r < self.sep_id_rp:
            rerank_input = self.basedata.id2ent[h] + ' ' + self.basedata.id2rel[r] + ' '+ self.basedata.id2ent[t]
        else:
            r_inv = self.basedata.inverse2rel_map[r] 
            rerank_input = self.basedata.id2ent[t] + ' ' + self.basedata.id2rel[r_inv] + ' '+ self.basedata.id2ent[h]
        return rerank_input
        
    def process_train_valid(self):
        samples = []
        labels = []
        triples_set_pos = set()
        triples_set_neg = set()
        
        for i in range(len(self.trips)):   
            score, trip = self.scores[i], self.trips[i]
            if self.mode == 'train':
                h, r = trip   
                true_tails = self.basedata.label_graph[(h,r)]
            else:
                h, r, t = trip    
                true_tails = self.basedata.label_graph[(h,r)] if (h,r) in self.basedata.label_graph else set()
                true_tails = true_tails.union(set([t]))
            
            scores_sorted, sorts_indices = torch.sort(torch.tensor(score), dim=0, descending=True)
            candi_TopK = sorts_indices[:(self.args.rerank_training_neg_K+len(true_tails))].tolist()
            candi_TopK_neg = []
            for candi in candi_TopK:
                if len(candi_TopK_neg) == self.args.rerank_training_neg_K:
                    break
                if candi not in true_tails:
                    candi_TopK_neg.append(candi)
            
            samples_i = []
            labels_i = []
            r_inv = self.basedata.inverse2rel_map[r]   
            for t in true_tails:
                if (h,r,t) not in triples_set_pos and (t,r_inv,h) not in triples_set_pos:
                    samples_i.append(self.get_triple_string(h, r, t))
                    labels_i.append(1)
                    triples_set_pos.add((h,r,t))
                    triples_set_pos.add((t,r_inv,h))
            for t in candi_TopK_neg:
                if (h,r,t) not in triples_set_neg and (t,r_inv,h) not in triples_set_neg:
                    samples_i.append(self.get_triple_string(h, r, t))
                    labels_i.append(0)
                    triples_set_neg.add((h,r,t))
                    triples_set_neg.add((t,r_inv,h))
            
            samples.extend(samples_i)
            labels.extend(labels_i)    
        return samples, labels


    def get_batch_rerank_data_bert_test(self, rawbatch):      
        for i in range(len(rawbatch)):   
            test_trips, test_scores = rawbatch[i]
            h, r, t = test_trips[0], test_trips[1], test_trips[2]       
            test_scores_sorted, test_sorts_indices = torch.sort(torch.tensor(test_scores), dim=0, descending=True)
            candi_TopK = test_sorts_indices[:self.args.rerank_Top_K]
            
            rerank_inputs =[]
            for candi_t in candi_TopK.tolist():
                rerank_inputs.append(self.get_triple_string(h, r, candi_t))

        hr = torch.LongTensor([h,r]).unsqueeze(0).repeat(self.args.rerank_Top_K,1)
        t = candi_TopK.long().unsqueeze(1)
        triples_id = torch.cat([hr,t],dim=1)

        rerank_inputs_code = self.tokenizer.batch_encode_plus(rerank_inputs, padding='longest', truncation=True, max_length=20)
        rerank_inputs_ids = torch.LongTensor(rerank_inputs_code["input_ids"])
        return {'triples_id':triples_id, 
                'test_scores_sorted': test_scores_sorted,
                'test_sorts_indices':test_sorts_indices,   
                'rerank_inputs':rerank_inputs_ids,   
                }

    
    def get_batch_rerank_data_bert_train_valid(self, rawbatch):    
        rerank_inputs = []
        labels = []  
        for i in range(len(rawbatch)):   
            rerank_input, label = rawbatch[i]
            rerank_inputs.append(rerank_input)
            labels.append(label)
        rerank_inputs_code = self.tokenizer.batch_encode_plus(rerank_inputs, padding='longest', truncation=True, max_length=60)
        rerank_inputs_ids = torch.LongTensor(rerank_inputs_code["input_ids"])
        labels = torch.FloatTensor(labels)   
        return {'labels':labels, 
                'rerank_inputs':rerank_inputs_ids,   
                }

