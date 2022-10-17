# @Author  :   Yuyue Zhao
# @email   :   yyzha0@mail.ustc.edu.cn

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gzip
import re
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from easydict import EasyDict as edict
import pandas as pd
from datetime import date
from sklearn import cluster
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM

# Dataset names.
BEAUTY = 'beauty'
CELL = 'cell'
CLOTH = 'cloth'
CD = 'cd'

DATASET_DIR = {
    BEAUTY: './data/Amazon_Beauty',
    CELL: './data/Amazon_Cellphones',
    CLOTH: './data/Amazon_Clothing',
    CD: './data/Amazon_CDs',
}

# Model result directories.
TMP_DIR = {
    BEAUTY: './tmp/Amazon_Beauty',
    CELL: './tmp/Amazon_Cellphones',
    CLOTH: './tmp/Amazon_Clothing',
    CD: './tmp/Amazon_CDs',
}

COMPLETE_REVIEW = {
    BEAUTY: 'reviews_Beauty_5.json.gz',
    CELL: 'reviews_Cell_Phones_and_Accessories_5.json.gz',
    CLOTH: 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
    CD: 'reviews_CDs_and_Vinyl_5.json.gz',
}

TIME_TRAIN = 'time_train.csv'
TIME_TEST = 'time_test.csv'


Y = 2000
seasons = [(0, (date(Y,  1,  1),  date(Y,  3, 20))),  #'winter'
        (1, (date(Y,  3, 21),  date(Y,  6, 20))),  #'spring'
        (2, (date(Y,  6, 21),  date(Y,  9, 22))),  #'summer'
        (3, (date(Y,  9, 23),  date(Y, 12, 20))),  #'autumn'
        (0, (date(Y, 12, 21),  date(Y, 12, 31)))]  #'winter'

def save_timeClusters(dataset, clusters_num, clusters_label):
    time_file = TMP_DIR[dataset] + '/time_clusters.pkl'
    time2save = clusters_num, clusters_label
    with open(time_file, 'wb') as f:
        pickle.dump(time2save, f)

def save_clus_weight(dataset, user_clus_weight, usage='train'):
    uc_file = TMP_DIR[dataset] + '/' + usage + '_user_clus_weight.pkl'
    with open(uc_file, 'wb') as f:
        pickle.dump(user_clus_weight, f)

def save_clus_dict(dataset, uid_clus_pid, usage='train'):
    uc_file = TMP_DIR[dataset] + '/' + usage + '_uid_clus_pid_dict.pkl'
    with open(uc_file, 'wb') as f:
        pickle.dump(uid_clus_pid, f)

def save_features_label(dataset, fea_label):
    uc_file = '../AddedEval' + '/_' + dataset + 'feature_label.pkl'
    with open(uc_file, 'wb') as f:
        pickle.dump(fea_label, f)



class ExtractTime4Concate(object):

    def __init__(self, data_dir, dataset_str, set_name='train'):
        self.data_dir = data_dir
        self.dataset_str =dataset_str
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        uidMap, pidMap = self.generate_id_hashmap()
        pair2add = self.pair2Add()
        user_item_time = self.generate_pair_time()
        self.review_with_time = self.map_uid_pid_pair_restore_time(user_item_time, pair2add, uidMap, pidMap)
        # print(self.review_with_time[:10])


    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            # In Python 3, must use decode() to convert bytes to string!
            return [line.decode('utf-8').strip() for line in f]
    
    def parse(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            for l in f:
                yield eval(l)
    
    def generate_id_hashmap(self):
        '''
        Transfer USER, PRODUCT to uid, pid
        Output: up_map{user: {USER: uid}, 
                        product: {PRODUCT: pid}}
        '''
        user_product_files = edict(
            user = 'users.txt.gz',
            product = 'product.txt.gz',
        )
        user_product = edict(
            user = [],
            product = [],
        )
        up_map = edict(
            user = {},
            product = {},
        )
        for name in user_product_files:
            # print(name)
            user_product[name] = self._load_file(user_product_files[name])
            for index in range(len(user_product[name])):
                #print(index)
                up_map[name][user_product[name][index]] = index
        return up_map['user'], up_map['product']

    def pair2Add(self):
        '''
        Load review file(train/test) and Extract (uid,pid) pair
        '''
        pair2add = [] # (user_idx, product_idx)
        for raw in self._load_file(self.review_file):
            arrRaw = raw.split('\t')
            user_idx = int(arrRaw[0])
            product_idx = int(arrRaw[1])
            pair2add.append([user_idx, product_idx])
        # print('Purchase history size in review data is:', pair2add[:10])
        return pair2add
        

    def generate_pair_time(self):
        sourcepair = COMPLETE_REVIEW[self.dataset_str]
        u_p_time = []
        print(self.parse(sourcepair))
        for review in self.parse(sourcepair):
            # print(review)
            # print(review['asin'])
            u_p_time.append([review['reviewerID'], review['asin'], review['reviewTime']])
        for rawIdx in range(len(u_p_time)):
            u_p_time[rawIdx][2] = self._time_Extract(u_p_time[rawIdx][2])
            # print(u_p_time[rawIdx])
            # pass
        return u_p_time 
    
    def _transfer2dictPair(self, pair_time_List, uidMap, pidMap):
        pairDict = {}
        for raw in pair_time_List:
            tmpList = (uidMap[raw[0]], pidMap[raw[1]])
            if tmpList in pairDict:
                pairDict[tmpList].append(raw[2])
            pairDict[tmpList] = [raw[2]]
        # print(list(pairDict.items())[:100])
        return pairDict

    def time2stamp(self, List2Transfer):
        return List2Transfer[2] + "-" + List2Transfer[0] + "-" + List2Transfer[1]
    
    def _time_Extract(self, timestamp):
        '''
        Extract timestamp to [month, day, year]
        '''
        t = re.findall(r"\d+", timestamp)
        if len(t) == 3:
            # a = self.time2stamp(t)
            return self.time2stamp(t)
        elif len(t) < 3:
            for i in range(3-len(t)):
                t.append(0)
        else:
            t = t[:3]
        # t = self.time2stamp(t)
        # t = t[2] + "-" + t[0] + "-" + t[1]
        # print(t)
        return self.time2stamp(t)

    
    def map_uid_pid_pair_restore_time(self, metafile, lackTfile, uidMap, pidMap):
        '''
        Input,
            metafile: list[(USER, PRODUCT), time[M, D, Y]]]
            lackTfile: list[[uid, pid]]
            uidMap: dict{USER, uid}
            pidMap: dict{PROFUCT, pid}
        Output, 
            pairWithTime: [[uid, pid, [time1, time2, ...]]]
        '''
        metaDict = self._transfer2dictPair(metafile, uidMap, pidMap)
        for idx in range(len(lackTfile)):
            pairID = (lackTfile[idx][0], lackTfile[idx][1])
            # check except.
            # if pairID not in metaDict:
            #     continue
            lackTfile[idx].append(metaDict[pairID])
        return lackTfile

# class TimeStatistics(object):
#     def __init__(se)
def listFlatten(timeList):
    '''Transfer [[319, 8099, ['2014-04-18', '2014-06-19']] ->
    [[319, 8099, '2014-04-18'],[319, 8099, '2014-06-19']]'''
    flatList = []
    for item in timeList:
        if len(item[2]) > 1:
            for time in item[2]:
                flatList.append([item[0], item[1], time])
        else:
            flatList.append([item[0], item[1], item[2][0]])
    return flatList

def list2csv(list2store, filename):
    name = ['UID', 'PID', 'PURCHASE_Time']
    tmp = pd.DataFrame(columns=name, data=list2store)
    tmp.to_csv(filename, encoding='gbk')

def get_season(dt):
    dt = dt.date() 
    dt = dt.replace(year=Y) 
    return next(season for season, (start, end) in seasons if start <= dt <= end)

def get_fre_deta(timeCSV):
    dac_time = timeCSV.PURCHASE_Time.value_counts()
    dac_time_date = pd.to_datetime(dac_time.index)

    dac_time_day = dac_time_date - dac_time_date.min()
    time2num = {}
    time2relative = {}
    serial2PrefixSum = {}
    for i in range(len(dac_time)):
        time2num[dac_time.index[i]] = [dac_time.values[i]]
    for i in range(len(dac_time)):
       
        time2relative[dac_time_day.days[i]] = dac_time.index[i]
    mapIndex = sorted(time2relative.keys())
    # timestamp2hit = time2num.copy()
    serial2PrefixSum[0] = 0
    for i in range(1, mapIndex[-1]+1):
    
        cur = time2num.get(time2relative.get(i, 0), 0)
        if cur:
            serial2PrefixSum[i] = serial2PrefixSum[i-1] + cur[0]
        else:
            serial2PrefixSum[i] = serial2PrefixSum[i-1]

    for gap in [90, 30, 7, 1]:
        structuralWithGap(gap, time2num, time2relative, mapIndex, serial2PrefixSum)
    return time2num

def structuralWithGap(gap, time2Num, time2relative, mapIndex, serial2PrefixSum):

    second_order_serial2PrefixSum = {} 
    init_left = (serial2PrefixSum[2*gap] - serial2PrefixSum[0] - 2*serial2PrefixSum[gap]) / gap
    for i in range(mapIndex[-1]+1):
        if i <= 2*gap:
            second_order_serial2PrefixSum[i] = init_left
        else:
            second_order_serial2PrefixSum[i] = (serial2PrefixSum[i] - 2 * serial2PrefixSum[i - gap] + serial2PrefixSum[i - 2*gap]) / gap

    init_left_2 = (second_order_serial2PrefixSum[2*gap] - second_order_serial2PrefixSum[0] - 2*second_order_serial2PrefixSum[gap]) / gap
    for idx in mapIndex:
        if idx <= 2*gap:
            gap_left = init_left
            gap_left_2 = init_left_2
        else:
            gap_left = (serial2PrefixSum[idx] - 2 * serial2PrefixSum[idx - gap] + serial2PrefixSum[idx - 2*gap]) / gap
            gap_left_2 = (second_order_serial2PrefixSum[idx] - 2 * second_order_serial2PrefixSum[idx - gap] + second_order_serial2PrefixSum[idx - 2*gap]) / gap
        
        time2Num[time2relative[idx]].append(gap_left)
        time2Num[time2relative[idx]].append(gap_left_2)

def timeAnalysis(csvFile, cluster_feature):
    timeCSV = pd.read_csv(csvFile)
    # 提取特征 年/月/日
    fullTime = pd.to_datetime(timeCSV.PURCHASE_Time)
    user_item_timestamp = np.array(timeCSV['PURCHASE_Time'])
    time2num = get_fre_deta(timeCSV)
    if cluster_feature == 'all':
        print('Extracting ' + cluster_feature + ' feature...')
        # =============================== Structural Features ================================= [90, 30, 7, 1]
        add_df = pd.DataFrame(columns=['pur_frequancy', 'order1_90', 'order2_90', 'order1_30', 'order2_30', 'order1_7', 'order2_7', 'order1_1', 'order2_1'], data = np.array([time2num[i] for i in user_item_timestamp]))
        timeCSV['pur_frequancy'], timeCSV['order1_90'], timeCSV['order2_90'], timeCSV['order1_30'], timeCSV['order2_30'], timeCSV['order1_7'], timeCSV['order2_7'], timeCSV['order1_1'], timeCSV['order2_1'] = add_df['pur_frequancy'], add_df['order1_90'], add_df['order2_90'], add_df['order1_30'], add_df['order2_30'], add_df['order1_7'], add_df['order2_7'], add_df['order1_1'], add_df['order2_1']

        # =============================== Stastical Features ================================= 
        timeCSV['tfa_year'] = np.array([x.year for x in fullTime])
        timeCSV['tfa_month'] = np.array([x.month for x in fullTime])
        timeCSV['tfa_day'] = np.array([x.day for x in fullTime])
        timeCSV['tfa_weekday'] = np.array([x.isoweekday() for x in fullTime])
        
        tfa_weekday = pd.get_dummies(timeCSV.tfa_weekday, prefix = 'tfa_weekday')  # one hot encoding 
        timeCSV = pd.concat((timeCSV, tfa_weekday), axis = 1)    

        timeCSV['tfa_season'] = np.array([get_season(x) for x in fullTime])
        tfa_season = pd.get_dummies(timeCSV.tfa_season, prefix = 'tfa_season') # one hot encoding 
        timeCSV = pd.concat((timeCSV, tfa_season), axis = 1)
        # df.drop(['tfa_season'], axis = 1, inplace = True)       
        # ================================== Over ============================================
    elif cluster_feature == 'w-stat':
        print('Extracting ' + cluster_feature + ' feature...')
        timeCSV['tfa_year'] = np.array([x.year for x in fullTime])
        timeCSV['tfa_month'] = np.array([x.month for x in fullTime])
        timeCSV['tfa_day'] = np.array([x.day for x in fullTime])
        timeCSV['tfa_weekday'] = np.array([x.isoweekday() for x in fullTime])
        
        tfa_weekday = pd.get_dummies(timeCSV.tfa_weekday, prefix = 'tfa_weekday')  # one hot encoding 
        timeCSV = pd.concat((timeCSV, tfa_weekday), axis = 1)     

        timeCSV['tfa_season'] = np.array([get_season(x) for x in fullTime])
        tfa_season = pd.get_dummies(timeCSV.tfa_season, prefix = 'tfa_season') # one hot encoding 
        timeCSV = pd.concat((timeCSV, tfa_season), axis = 1)
    elif cluster_feature == 'w-stru':
        print('Extracting ' + cluster_feature + ' feature...')
        add_df = pd.DataFrame(columns=['pur_frequancy', 'order1_90', 'order2_90', 'order1_30', 'order2_30', 'order1_7', 'order2_7', 'order1_1', 'order2_1'], data = np.array([time2num[i] for i in user_item_timestamp]))
        timeCSV['pur_frequancy'], timeCSV['order1_90'], timeCSV['order2_90'], timeCSV['order1_30'], timeCSV['order2_30'], timeCSV['order1_7'], timeCSV['order2_7'], timeCSV['order1_1'], timeCSV['order2_1'] = add_df['pur_frequancy'], add_df['order1_90'], add_df['order2_90'], add_df['order1_30'], add_df['order2_30'], add_df['order1_7'], add_df['order2_7'], add_df['order1_1'], add_df['order2_1']
    else:
        print('Wrong Cluster Feature Setting!!!')

    return timeCSV

def hierarchicalTime(args, fileTimePlus):

    if args.cluster_feature == 'all':
        fileTime = pd.DataFrame(fileTimePlus, columns=['pur_frequancy', 'order1_90', 'order2_90', 'order1_30', 'order2_30', 'order1_7', 'order2_7', 'order1_1', 'order2_1', 'tfa_year', 'tfa_month', 'tfa_day', 'tfa_weekday',  'tfa_weekday_1', 'tfa_weekday_2', 'tfa_weekday_3', 'tfa_weekday_4', 'tfa_weekday_5', 'tfa_weekday_6', 'tfa_weekday_7', 'tfa_season', 'tfa_season_0', 'tfa_season_1', 'tfa_season_2', 'tfa_season_3'])
        fileTime['tfa_year'] = fileTime['tfa_year'] - fileTime['tfa_year'].min()
    elif args.cluster_feature == 'w-stat':
        fileTime = pd.DataFrame(fileTimePlus, columns=['tfa_year', 'tfa_month', 'tfa_day', 'tfa_weekday',  'tfa_weekday_1', 'tfa_weekday_2', 'tfa_weekday_3', 'tfa_weekday_4', 'tfa_weekday_5', 'tfa_weekday_6', 'tfa_weekday_7', 'tfa_season', 'tfa_season_0', 'tfa_season_1', 'tfa_season_2', 'tfa_season_3'])
        fileTime['tfa_year'] = fileTime['tfa_year'] - fileTime['tfa_year'].min()
    elif args.cluster_feature == 'w-stru':
        fileTime = pd.DataFrame(fileTimePlus, columns=['pur_frequancy', 'order1_90', 'order2_90', 'order1_30', 'order2_30', 'order1_7', 'order2_7', 'order1_1', 'order2_1'])
    print('Cluster Feature is : ' + args.cluster_feature)

    timestamp = pd.DataFrame(fileTimePlus, columns=['PURCHASE_Time'])
    timestamp = np.array(timestamp)
    x = np.array(fileTime)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) 
 
    suitableCluser = args.cluster_num
    print('time cluster number is: ' + str(suitableCluser))
    models = GMM(suitableCluser, covariance_type='full', random_state=12).fit(x)
    labels = models.predict(x)
    # fea_label = timestamp, x, labels
    # save_features_label(args.dataset, fea_label)
    return models, suitableCluser, labels


def generate_clus_dict(timeData, clus_label):
    uid_pid_clu = pd.DataFrame(timeData, columns=['UID', 'PID'])
    uid_pid_clu['clu_label'] = clus_label
    uid_pid_clu_list = uid_pid_clu.values.tolist()
    ucp_hash = {}
    for [uid, pid, clu] in uid_pid_clu_list:
        if uid not in ucp_hash:
            ucp_hash[uid] = {clu : [pid]}  # ucp_hash : {uids{c1: pid, c2:pid, ...}, ...}
        else:
            if clu in ucp_hash[uid]:
                ucp_hash[uid][clu].append(pid)
            else:
                ucp_hash[uid][clu] = [pid]
    return ucp_hash

def generate_user_agent_num(ucp_hash):
    u_c_weight = ucp_hash
    for u in u_c_weight:
        tmp_u_tot = 0
        for c in u_c_weight[u]:
            tmp_u_tot = tmp_u_tot + len(u_c_weight[u][c])
        for c in u_c_weight[u]:
            u_c_weight[u][c] = len(u_c_weight[u][c])/tmp_u_tot
    return u_c_weight


def writeTimeConfig(fileName, Clusters_Num):
    with open(fileName, 'w') as f:
        f.write('CLUSNUM = ' + str(Clusters_Num) + '\n')
        f.write('USER = \'user\' \n')
        f.write('PRODUCT = \'product\' \n')
        f.write('WORD = \'word\' \n')
        f.write('RPRODUCT = \'related_product\' \n')
        f.write('BRAND = \'brand\' \n')
        f.write('CATEGORY = \'category\' \n')

    PURCHASE = {}
    MENTION = {}
    DESCRIBED_AS = {}

    # Write Relations
    with open(fileName, 'a+') as f:
        f.write('PRODUCED_BY = \'produced_by\' \n')
        f.write('BELONG_TO = \'belongs_to\' \n')
        f.write('ALSO_BOUGHT = \'also_bought\' \n')
        f.write('ALSO_VIEWED = \'also_viewed\' \n')
        f.write('BOUGHT_TOGETHER = \'bought_together\' \n')
        f.write('SELF_LOOP = \'self_loop\' \n')
        for i in range(Clusters_Num):
            # f.write('PURCHASE_' + str(i) + '= \'purchase_' + str(i) + '\' \n')
            # f.write('MENTION_' + str(i) + '= \'mention_' + str(i) + '\' \n')
            # f.write('DESCRIBED_AS_' + str(i) + '= \'described_as_' + str(i) + '\' \n')
            PURCHASE[i] = 'purchase_' + str(i)
            MENTION[i] = 'mention_' + str(i)
            DESCRIBED_AS[i] = 'described_as_' + str(i)
        f.write('PURCHASE = ' + str(PURCHASE) + '\n')
        f.write('MENTION = ' + str(MENTION) + '\n')
        f.write('DESCRIBED_AS = ' + str(DESCRIBED_AS) + '\n')



    with open(fileName, 'a+') as f:
        f.write('KG_RELATION = { \n')

        # USER
        f.write('USER: { \n')
        for i in range(Clusters_Num):
            f.write('\'' + PURCHASE[i] + '\'' + ': PRODUCT, \n')
            f.write('\'' + MENTION[i] + '\'' + ': WORD, \n')
        f.write('}, \n')

        # WORD
        f.write('WORD: { \n')
        for i in range(Clusters_Num):
            f.write('\'' + MENTION[i] + '\'' + ': USER, \n')
            f.write('\'' + DESCRIBED_AS[i] + '\'' + ': PRODUCT, \n')
        f.write('}, \n')

        # PRODUCT
        f.write('PRODUCT: { \n')
        for i in range(Clusters_Num):
            f.write('\'' + PURCHASE[i] + '\'' + ': USER, \n')
            f.write('\'' + DESCRIBED_AS[i] + '\'' + ': WORD, \n')
        f.write('PRODUCED_BY: BRAND, \n')
        f.write('BELONG_TO: CATEGORY, \n')
        f.write('ALSO_BOUGHT: RPRODUCT, \n')
        f.write('ALSO_VIEWED: RPRODUCT, \n')
        f.write('BOUGHT_TOGETHER: RPRODUCT, \n')
        f.write('}, \n')

        # BRAND
        f.write('BRAND: { \n')
        f.write('PRODUCED_BY: PRODUCT, \n')
        f.write('}, \n')

        # CATEGORY
        f.write('CATEGORY: { \n')
        f.write('BELONG_TO: PRODUCT, \n')
        f.write('}, \n')

        # RPRODUCT
        f.write('RPRODUCT: { \n')
        f.write('ALSO_BOUGHT: PRODUCT, \n')
        f.write('ALSO_VIEWED: PRODUCT, \n')
        f.write('BOUGHT_TOGETHER: PRODUCT, \n')
        f.write('}} \n')




def test_knn_cluster(cluster_info, clus_fea, timeData):
    if clus_fea == 'all':
        fileTime = pd.DataFrame(timeData, columns=['pur_frequancy', 'order1_90', 'order2_90', 'order1_30', 'order2_30', 'order1_7', 'order2_7', 'order1_1', 'order2_1', 'tfa_year', 'tfa_month', 'tfa_day', 'tfa_weekday',  'tfa_weekday_1', 'tfa_weekday_2', 'tfa_weekday_3', 'tfa_weekday_4', 'tfa_weekday_5', 'tfa_weekday_6', 'tfa_weekday_7', 'tfa_season', 'tfa_season_0', 'tfa_season_1', 'tfa_season_2', 'tfa_season_3'])
        fileTime['tfa_year'] = fileTime['tfa_year'] - fileTime['tfa_year'].min()
    elif clus_fea == 'w-stat':
        fileTime = pd.DataFrame(timeData, columns=['tfa_year', 'tfa_month', 'tfa_day', 'tfa_weekday',  'tfa_weekday_1', 'tfa_weekday_2', 'tfa_weekday_3', 'tfa_weekday_4', 'tfa_weekday_5', 'tfa_weekday_6', 'tfa_weekday_7', 'tfa_season', 'tfa_season_0', 'tfa_season_1', 'tfa_season_2', 'tfa_season_3'])
        fileTime['tfa_year'] = fileTime['tfa_year'] - fileTime['tfa_year'].min()
    elif clus_fea == 'w-stru':
        fileTime = pd.DataFrame(timeData, columns=['pur_frequancy', 'order1_90', 'order2_90', 'order1_30', 'order2_30', 'order1_7', 'order2_7', 'order1_1', 'order2_1'])

    x = np.array(fileTime)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    test_cluster_label = cluster_info.predict(x)

    return test_cluster_label


def train_preparation(args):
    # Storage time.
    timestamp = ExtractTime4Concate(DATASET_DIR[args.dataset], dataset_str=args.dataset, set_name='train')
    timeDict = timestamp.review_with_time
    list2csv(listFlatten(timeDict), DATASET_DIR[args.dataset] + "/" + TIME_TRAIN)
    # Analysis time.
    timeData = timeAnalysis(DATASET_DIR[args.dataset] + "/" + TIME_TRAIN, args.cluster_feature)
    gmmModel, timeNum, timeClassifyLabel = hierarchicalTime(args, timeData)
    # timeClusterUidPidReview(timeData, timeClassifyLabel)

    save_timeClusters(args.dataset, timeNum, timeClassifyLabel)
    ucp_hash = generate_clus_dict(timeData, timeClassifyLabel)
    save_clus_dict(args.dataset, ucp_hash)
    uc_weight = generate_user_agent_num(ucp_hash)
    save_clus_weight(args.dataset, uc_weight)
    writeTimeConfig('timeUtils.py', timeNum)
    return gmmModel


def test_preparation(args, cluster_info):
    timestamp = ExtractTime4Concate(DATASET_DIR[args.dataset], dataset_str=args.dataset, set_name='test')
    timeDict = timestamp.review_with_time
    list2csv(listFlatten(timeDict), DATASET_DIR[args.dataset] + "/" + TIME_TEST)
    # Analysis time.
    timeData = timeAnalysis(DATASET_DIR[args.dataset] + "/" + TIME_TEST, args.cluster_feature)
    test_cluster_label = test_knn_cluster(cluster_info, args.cluster_feature, timeData)
    test_ucp_hash = generate_clus_dict(timeData, test_cluster_label)
    save_clus_dict(args.dataset, test_ucp_hash, usage='test')
    uc_weight = generate_user_agent_num(test_ucp_hash)
    save_clus_weight(args.dataset, uc_weight, usage='test')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beauty', help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--cluster_num', type=int, default=14, help='The Num of Clusters.')
    parser.add_argument('--cluster_feature', type=str, default='w-stat', help='Temporal feature used for modeling behavior.')
    args = parser.parse_args()
    gmmModel = train_preparation(args)
    test_preparation(args, gmmModel)
    print("Done.")

if __name__ == "__main__":
    main()
