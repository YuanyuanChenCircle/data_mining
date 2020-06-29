# import pandas as pd
# import numpy as np
# import random, string
# import re
# import math
# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules
#
#
# # def do_association_rule(df):
# #     print(df[0:10])
# #     print(type(df))
# #     print("###############")
# #
# #     frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
# #     print(frequent_itemsets)
# #
# #     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
# #     print(rules)
# #
# #     return frequent_itemsets,rules
#
#
# # !/usr/bin/env python
# # -*-  coding:utf-8  -*-
# # version: Python 3.6.4 on win64
# # author:  Suranyi    time:  8/6
# # title:  Apriori算法
#
# import os,itertools
# import numpy as np
# import pandas as pd
#
# class Apriori(object):
#     def __init__(self, itemSets, minSupport=0.5, minConf=0.7, sort = False):
#         self.itemSets = itemSets
#         self.minSupport = minSupport
#         self.minConf = minConf
#         self.sort = sort
#         self.__Initialize()
#
#     def __Initialize(self):
#         self.__item()
#         self.__creat_matrix()
#         self.update(minSupport=self.minSupport, minConf=self.minConf)
#
#     def __item(self):
#         '''获取项目元素列表'''
#         self.item = []#创建那个一维非空
#         for itemSet in self.itemSets:
#             for item in itemSet:
#                 if item not in self.item:
#                     self.item.append(item)
#         self.item.sort()#为什么排序****************
#
#     def __creat_matrix(self):
#         '''将项集转为pandas.DataFrame数据类型'''
#         self.data = pd.DataFrame(columns=self.item)#data为转换为的一维数
#         for i in range(len(self.itemSets)):#原始的列表
#             self.data.loc[i, self.itemSets[i]] = 1#选择坐标赋值为1
#
#     def __candidate_itemsets_l1(self):
#         '''创建单项频繁项集及L1'''
#         self.L1 = self.data.loc[:, self.data.sum(axis=0) / len(self.itemSets) >= self.minSupport]
#         self.L1_support_selects = dict(self.L1.sum(axis=0) / len(self.itemSets))  # 只作为分母，不进行拆分
#
#     def __candidate_itemsets_lk(self):
#         '''根据L1创建多项频繁项集Lk，非频繁项集的任何超集都不是频繁项集'''
#         last_support_selects = self.L1_support_selects.copy()  # 初始化
#         while last_support_selects:
#             new_support_selects = {}
#             for last_support_select in last_support_selects.keys():
#                 for L1_support_name in set(self.L1.columns) - set(last_support_select.split(',')):
#                     columns = sorted([L1_support_name] + last_support_select.split(','))  # 新的列名：合并后排序
#                     count = (self.L1.loc[:, columns].sum(axis=1) == len(columns)).sum()
#                     if count / len(self.itemSets) >= self.minSupport:
#                         new_support_selects[','.join(columns)] = count / len(self.itemSets)
#             self.support_selects.update(new_support_selects)
#             last_support_selects = new_support_selects.copy()  # 作为新的 Lk，进行下一轮更新
#
#     def __support_selects(self):
#         '''支持度选择'''
#         self.__candidate_itemsets_l1()
#         self.__candidate_itemsets_lk()
#         self.item_Conf = self.L1_support_selects.copy()
#         self.item_Conf.update(self.support_selects)
#
#     def __confidence_selects(self):
#         '''生成关联规则，其中support_selects已经按照长度大小排列'''
#         for groups, Supp_groups in self.support_selects.items():
#             groups_list = groups.split(',')
#             for recommend_len in range(1, len(groups_list)):
#                 for recommend in itertools.combinations(groups_list, recommend_len):
#                     items = ','.join(sorted(set(groups_list) - set(recommend)))
#                     Conf = Supp_groups / self.item_Conf[items]
#                     if Conf >= self.minConf:
#                         self.confidence_select.setdefault(items, {})
#                         self.confidence_select[items].setdefault(','.join(recommend),{'Support': Supp_groups, 'Confidence': Conf})
#
#     def show(self,**kwargs):
#         '''可视化输出'''
#         if kwargs.get('data'):
#             select = kwargs['data']
#         else:
#             select = self.confidence_select
#         items = []
#         value = []
#         for ks, vs in select.items():
#             items.extend(list(zip([ks] * vs.__len__(), vs.keys())))
#             for v in vs.values():
#                 value.append([v['Support'], v['Confidence']])
#         index = pd.MultiIndex.from_tuples(items, names=['Items', 'Recommend'])
#         self.rules = pd.DataFrame(value, index=index, columns=['Support', 'Confidence'])
#         if self.sort or kwargs.get('sort'):
#             result = self.rules.sort_values(by=['Support', 'Confidence'], ascending=False)
#         else:
#             result = self.rules.copy()
#         return result
#
#     def update(self, **kwargs):
#         '''用于更新数据'''
#         if kwargs.get('minSupport'):
#             self.minSupport = kwargs['minSupport']
#             self.support_selects = {}  # 用于储存满足支持度的频繁项集
#             self.__support_selects()
#         if kwargs.get('minConf'):
#             self.minConf = kwargs['minConf']
#             self.confidence_select = {}  # 用于储存满足自信度的关联规则
#             self.__confidence_selects()
#         print(self.show())
#         if kwargs.get('file_name'):
#             file_name = kwargs['file_name']
#             self.show().to_excel(f'/../table/{file_name}.xlsx')
#         self.apriori_rules = self.rules.copy()
#
#     def __get_Recommend_list(self,itemSet):
#         '''输入数据，获取关联规则列表'''
#         self.recommend_selects = {}
#         itemSet = set(itemSet) & set(self.apriori_rules.index.levels[0])
#         if itemSet:
#             for start_str in itemSet:
#                 for end_str in self.apriori_rules.loc[start_str].index:
#                     start_list = start_str.split(',')
#                     end_list = end_str.split(',')
#                     self.__creat_Recommend_list(start_list, end_list, itemSet)
#
#     def __creat_Recommend_list(self,start_list,end_list,itemSet):
#         '''迭代创建关联规则列表'''
#         if set(end_list).issubset(itemSet):
#             start_str = ','.join(sorted(start_list+end_list))
#             if start_str in self.apriori_rules.index.levels[0]:
#                 for end_str in self.apriori_rules.loc[start_str].index:
#                     start_list = start_str.split(',')
#                     end_list = end_str.split(',')
#                     self.__creat_Recommend_list(sorted(start_list),end_list,itemSet)
#         elif not set(end_list) & itemSet:
#             start_str = ','.join(start_list)
#             end_str = ','.join(end_list)
#             self.recommend_selects.setdefault(start_str, {})
#             self.recommend_selects[start_str].setdefault(end_str, {'Support': self.apriori_rules.loc[(start_str, end_str), 'Support'], 'Confidence': self.apriori_rules.loc[(start_str, end_str), 'Confidence']})
#
#     def get_Recommend(self,itemSet,**kwargs):
#         '''获取加权关联规则'''
#         self.recommend = {}
#         self.__get_Recommend_list(itemSet)
#         self.show(data = self.recommend_selects)
#         items = self.rules.index.levels[0]
#         for item_str in items:
#             for recommends_str in self.rules.loc[item_str].index:
#                 recommends_list = recommends_str.split(',')
#                 for recommend_str in recommends_list:
#                     self.recommend.setdefault(recommend_str,0)
#                     self.recommend[recommend_str] += self.rules.loc[(item_str,recommends_str),'Support'] * self.rules.loc[(item_str,recommends_str),'Confidence'] * self.rules.loc[item_str,'Support'].mean()/(self.rules.loc[item_str,'Support'].sum()*len(recommends_list))
#         result = pd.Series(self.recommend,name='Weight').sort_values(ascending=False)
#         result.index.name = 'Recommend'
#         result = result/result.sum()
#         result = 1/(1+np.exp(-result))
#         print(result)
#         if kwargs.get('file_name'):
#             file_name = kwargs['file_name']
#             excel_writer = pd.ExcelWriter(f'{os.getcwd()}/{file_name}.xlsx')
#             result.to_excel(excel_writer,'推荐项目及权重')
#             self.rules.to_excel(excel_writer, '关联规则树状表')
#             self.show().to_excel(excel_writer, '总关联规则树状表')
#             self.show(sort = True).to_excel(excel_writer, '总关联规则排序表')
#             excel_writer.save()
#         return result
#
# def str2itemsets(strings, split=','):
#     '''将字符串列表转化为对应的集合'''
#     itemsets = []
#     for string in strings:
#         itemsets.append(sorted(string.split(split)))
#     return itemsets
#
#
# # def do_association_rule(df):
# #     # 1.导入数据
# #     # data = pd.read_excel(r'apriori算法实现.xlsx', index=False)
# #
# #     # 2.关联规则中不考虑多次购买同一件物品，删除重复数据,不考虑一个物品在一个销售单中出现多次，只考虑不同物品在一个销售单之间的关联
# #     data = df.drop_duplicates()
# #
# #     # 3.初始化列表
# #     itemSets = []
# #
# #     # 3.按销售单分组，只有1件商品的没有意义，需要进行过滤
# #     groups = data.groupby(by='Sales order details')
# #     # print(groups)
# #     # print("#######")
# #     for group in groups:#访问分组对象
# #         # print(type(group))
# #         if len(group[1]) >= 2:
# #             # print(type(group[1]))
# #             # print(group[1])
# #             itemSets.append(group[1]['Commodity code'].tolist())#tolist转换为列表，两层列表
# #
# #
# #     # 4.训练 Apriori
# #     ap = Apriori(itemSets, minSupport=0.03, minConf=0.5)
# #     print("#####################")
# #     print(ap)
# #
# #     re = ap.get_Recommend('2BYP206,2BYW001-,2BYW013,2BYX029'.split(','))


'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *
import numpy as np


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)  # use frozen set so we
    # can use it as a key in a dict


def scanD(D, Ck, minSupport):
    ssCnt = {}
    # print(list(D))
    # print(map(set,D))
    D1 = map(set, D)
    # print(D1)
    D1 = list(D1)
    # print(D1)
    print(len(D1))

    # print(len(list(D)))
    # print("######")
    # print(len(list(D)))
    # print(Ck)
    # print(list(Ck))
    # print(minSupport)
    i = 0
    for tid in D1:
        # print(tid)
        i = i + 1
        # print(i)
        # print(tid)
        # print(Ck)
        Ck = list(Ck)
        for can in Ck:
            # print(can)
            if can.issubset(tid):
                # print('*' * 9)
                # if not ssCnt.has_key(can): ssCnt[can]=1
                if can not in ssCnt:
                    ssCnt[can] = 1

                else:
                    ssCnt[can] += 1
                    # print("6666666")
    # print(D)
    # print(ssCnt)
    # print(list(D1))
    print("######")
    print(len(D1))

    numItems = 8124
    # print(numItems)
    retList = []
    supportData = {}
    for key in ssCnt:
        # print(key)
        support = ssCnt[key] / numItems
        # print(support)
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support

    print(retList)
    return retList, supportData


def aprioriGen(Lk, k):  # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2];
            L2 = list(Lk[j])[:k - 2]
            L1.sort();
            L2.sort()
            if L1 == L2:  # if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j])  # set union
    return retList


def apriori(dataSet, minSupport=0.3):
    C1 = createC1(dataSet)
    # print(dataSet)
    # D = map(set, dataSet)
    D = dataSet

    # print(list(D))
    # D = list(D)
    # print(len(list(D)))
    print("#####")
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  # supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):  # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # create new list to return
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # calc confidence
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # try further merging
        Hmp1 = aprioriGen(H, m + 1)  # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print("--------")  # print a blank line

# from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# #votesmart.apikey = 'get your api key first'
# def getActionIds():
#     actionIdList = []; billTitleList = []
#     fr = open('recent20bills.txt')
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum) #api call
#             for action in billDetail.actions:
#                 if action.level == 'House' and \
#                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print('bill: %d has actionId: %d' % (billNum, actionId))
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print("problem getting bill %d" % billNum)
#         sleep(1)                                      #delay to be polite
#     return actionIdList, billTitleList

# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician)
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print('getting votes for actionId: %d' % actionId)
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not transDict.has_key(vote.candidateName):
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except:
#             print("problem getting actionId: %d" % actionId)
#         voteCount += 2
#     return transDict, itemMeaning
def do_association_rule(df):
    df = np.array(df)
    da = df.tolist()
    L,s = apriori(da,minSupport = 0.3)
    return L,s