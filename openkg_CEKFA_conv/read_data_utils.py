import numpy as np
from collections import OrderedDict, defaultdict

def get_noun_phrases(ent2id_path):
    f = open(ent2id_path, "r").readlines()
    phrase2id = {}
    id2phrase = {}
    for line in f[1:]:
        line = line.strip().split("\t")
        phrase,ID = line[0], line[1]
        phrase2id[phrase] = int(ID)
        id2phrase[int(ID)] = phrase
    return phrase2id, id2phrase

def get_relation_phrases(rel2id_path, add_reverse=True):
    inverse2rel_map = {}
    f = open(rel2id_path,"r").readlines()
    phrase2id = {}
    id2phrase = {}
    phrases = []
    for line in f[1:]:
        line = line.strip().split("\t")
        phrase,ID = line[0], line[1]
        phrases.append(phrase)
        phrase2id[phrase] = int(ID)
        id2phrase[int(ID)] = phrase
    cur_id = max(id2phrase.keys())
    if add_reverse:
        for phrase in phrases:
            newphrase =  "inverse of " + phrase
            cur_id += 1 
            phrase2id[newphrase] = cur_id                       # 添加反关系
            id2phrase[cur_id] = newphrase
            inverse2rel_map[cur_id] = phrase2id[phrase]    # 反关系id 映射到 关系id
            inverse2rel_map[phrase2id[phrase]] = cur_id    # 关系id 映射到 反关系id    
    return phrase2id, id2phrase, inverse2rel_map

def get_clusters(clust_path):
    # 第一项是{entid:同cluster的ents}     
    # 第二项是{entid:clustid}
    # 第三项存储簇信息，是个list，如unique_clusts_cesi[i]相当于存储 clustid 为 i 的 ents
    content_clusts = {}
    contentid2clustid = {}
    content_list = []                                   # 存储已经记录cluster信息的ents
    unique_clusts = []           
    ID = -1                     
    f = open(clust_path,"r").readlines()
    for line in f:
        line = line.strip().split()
        clust = [int(content) for content in line[2:]]      # 同一个cluster下的所有实体, 包括自己
        content_clusts[int(line[0])] = clust                # {entid:同一个cluster下的所有ent, 包括自己}
        if line[0] not in content_list:
            ID+=1                                       # 簇的 id
            unique_clusts.append(clust)                 # 存储 簇信息，unique_clusts[i]相当于存储cluster id为i的ents
            content_list.extend(line[2:])               # 注意是extend！
            for content in clust: contentid2clustid[content] = ID
    return content_clusts, contentid2clustid, unique_clusts, ID+1

def get_train_triples(triples_path, entid2clustid, rel2id, id2rel, add_reverse=True):
    trip_list = []
    trip_list_without_rev = []
    label_graph = defaultdict(set)
    label_graph_rt2h = defaultdict(set)
    label_graph_ht2r = defaultdict(set)
    label_graph_r2ht = defaultdict(set)
    label_filter = defaultdict(set)
    f = open(triples_path,"r").readlines()
    for trip in f[1:]:
        trip = trip.strip().split()
        e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
        if add_reverse:
            r_inv = "inverse of " + id2rel[r]       # 添加 反关系
            if r_inv not in rel2id: raise Exception
            r_inv = rel2id[r_inv]                   # 反关系的 id

        label_graph[(e1,r)].add(e2)
        label_graph_rt2h[(r,e2)].add(e1)
        label_graph_ht2r[(e1,e2)].add(r)
        label_graph_r2ht[r].add((e1,e2))
        label_filter[(e1,r)].add(entid2clustid[e2])
        trip_list.append([e1,r,e2])
        trip_list_without_rev.append([e1,r,e2])

        if add_reverse:
            label_graph[(e2,r_inv)].add(e1)
            label_graph_rt2h[(r_inv,e1)].add(e2)
            label_graph_ht2r[(e2,e1)].add(r_inv)
            label_graph_r2ht[r_inv].add((e2,e1))
            label_filter[(e2,r_inv)].add(entid2clustid[e1])
            trip_list.append([e2,r_inv,e1])
    
    return np.array(trip_list),rel2id,label_graph, trip_list_without_rev, [label_graph_rt2h, label_graph_ht2r, label_graph_r2ht], label_filter

def get_test_triples(triples_path,rel2id,id2rel,add_reverse=True):
    trip_list = []
    trip_list_without_rev = []
    f = open(triples_path,"r").readlines()
    for trip in f[1:]:
        trip = trip.strip().split()
        e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
        trip_list.append([e1,r,e2])
        trip_list_without_rev.append([e1,r,e2])
        if add_reverse:
            r_inv = "inverse of " + id2rel[r]
            if r_inv not in rel2id: raise Exception
            r_inv = rel2id[r_inv] 
            trip_list.append([e2,r_inv,e1])
    return np.array(trip_list), trip_list_without_rev
