# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:24:30 2021

@author: Administrator
"""

from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import datetime
import logging
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tqdm import tqdm,trange
import random
import gc
from util import *
import pickle
import time
import warnings
warnings.filterwarnings("ignore")
# torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.FloatTensor)
from tqdm import tqdm,trange
from torch.nn import CrossEntropyLoss, MSELoss
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_useful=['tweet_id','user_id','thread','event','is_source_tweet','is_rumor','in_reply_tweet','in_reply_user','created']
'''
    一个event 一个graph,一个thread一个样本
    每个thread中抽取到的user状态可能不同    
'''
#所有用户的静态特征
basef1=['user.tweets_count','user.verified','user.followers_count','user.listed_count','user.friends_count',
    'user.desc_length','user.has_bg_img','user.created_at']

def generateProfileX(event,data):
    '''
    拿到所有用户的静态特征，以及user_id_graphid、graphid_user_id两个映射字典
    event 事件
    data 原始twwiter数据
    如果指定位置没有当前文件，则重新取，如果有则直接读
    '''
    if os.path.exists("middle/tweets/node_feature/%s.pkl"% event):
        with open("middle/tweets/node_feature/%s.pkl"% event, 'rb') as f:
            node_feature=pickle.load(f)
            return node_feature['data'],node_feature['user_graph'],node_feature['graph_user']
    else:
        baseF=['user_id','user.tweets_count','user.verified','user.followers_count','user.listed_count','user.friends_count',
         'user.desc_length','user.has_bg_img','user.created_at']
        
        with open("middle/tweets/user_meta_%s.pkl"% event, 'rb') as f:
            reply_user=pickle.load(f)
        data.drop_duplicates(subset=['user_id'], keep='last', inplace=True)
        reply_user=reply_user[~reply_user['user_id'].isin(data['user_id'].values)]
        print('原始用户数',data['user_id'].nunique(),'扩展用户数',reply_user['user_id'].nunique())
        
        temp=pd.concat([data[baseF], reply_user], ignore_index=True)
        print('data.shape',data.shape)
        temp.fillna(0, inplace=True)
        
        #对各个特征列归一化
        temp[basef1]=temp[basef1].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        
        user_id_graphid={}
        graphid_user_id={}
        
        userlist=temp['user_id'].unique()
        print('用户节点总数',len(userlist))
        for i in range(len(userlist)):
#            print(i)
            user_id_graphid[userlist[i]]=i
            graphid_user_id[i]=userlist[i]
        temp['graphid']=temp['user_id'].map(user_id_graphid)
        #存储节点特征
        with open("middle/tweets/node_feature/%s.pkl"% event, 'wb') as f:
            pickle.dump({'data':temp[baseF+['graphid']],
                         'user_graph':user_id_graphid,
                         'graph_user':graphid_user_id
                         }, f)
        
        return temp[baseF+['graphid']],user_id_graphid,graphid_user_id



def getstart(event,train_pro=0.8):
    '''
    event 具体事件
    train_pro 训练比例
    加载静态数据（节点不变的属性，数据划分结果、异质邻接图）
    划分数据集 8：2划分thread
    return :
    node_feature, 节点的静态属性，dataframe类型，包含graphid和user的属性
    train_thread,训练的进程
    test_thread,测试的进程
    global_graph,封装了取邻接矩阵的各种方法
    
    '''
    data=fetch_tweets(event)
    node_feature,user_id_graphid,graphid_user_id=generateProfileX(event,data)
    Threadlists=data['thread'].unique()
#    Threadlists=data['thread'].values
    random.shuffle(Threadlists)
    train_thread=Threadlists[:int(len(Threadlists)*train_pro)]
    test_thread=list(filter(lambda x: x not in train_thread, Threadlists))
    print('训练样本',len(train_thread),'测试样本:',len(test_thread))
    global_graph=MyGraph(user_id_graphid,graphid_user_id)
    return node_feature,train_thread,test_thread,global_graph

#class TensorDataset(Dataset):
#    """Dataset wrapping data and target tensors.
#
#    Each sample will be retrieved by indexing both tensors along the first
#    dimension.
#
#    Arguments:
#        data_tensor (Tensor): contains sample data.
#        target_tensor (Tensor): contains sample targets (labels).
#    """
#
#    def __init__(self,event, node_feature,thread,global_graph):
#        
#        self.event=event
#        self.node_feature=node_feature
#        self.thread=thread
#        
#        self.graph=global_graph
#        self.data=self.threadprocess()
#    def threadprocess(self):
#        '''
#        把数据处理成样本 sample:(A,X,state,Y)
#        A 邻接矩阵，[A1,A2,A3]分别表示朋友关系矩阵、转推关系矩、评论关系矩
#        X 用户节点特征矩阵
#        state 用户传播谣言状态，（即用户是否发推,用户传播了谣言为1，没有传播为-1）
#        spread_time  归一化的谣言散布时间（虽然经过处理但是时间的先后顺序没变）
#        Y 当前用户是否是源头
#        
#        '''
#        data=fetch_tweets(self.event)
#        data['created_1']=data[['created']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#        static_f=self.node_feature.sort_values(by = ['graphid'],axis = 0,ascending = True)[basef1].values
#        input_ = []
#        for t in self.thread:
#            temp=data[data['thread']==t]
#            A_3=self.graph.csv_dict(temp,'in_reply_user','user_id')
#            A_3=self.graph.getAdjNormornize(self.graph.getadju(A_3))
#            A_1,A_2=self.graph.process_Adjacent(self.event)
#            
#            #构造state
#            state=np.zeros((self.graph.node_num,1))
#            spread_time=np.zeros((self.graph.node_num,1))
#            #u_sum_in[np.where(u_sum_in == 0)] = 1
#            for u_uid in temp['user_id'].values:
#                u=self.graph.n_g[u_uid]
#                state[u][0]=state[u][0]+1
#                spread_time[u][0]=temp[temp['user_id']==u_uid]['created_1'].values[0]
#            
#            #构造X 包含静态部分和动态变化部分
#            X=np.hstack((static_f,state)).astype(float)#self.graph.node_num ,len(basef1)+1
##            print(static_f.shape,'state',state.shape,'   X.shape',X.shape)
#            #构造Y
#            Y=np.zeros((self.graph.node_num,1))
#            #找到源头用户
#            for u in temp[temp['is_source_tweet']==1]['user_id'].unique():
#                Y[self.graph.n_g[u]][0]=1
#            
#            input_.append((
#                    torch.tensor(A_1),
#                    torch.tensor(A_2),
#                    torch.tensor(A_3),
#                    torch.tensor(X),
#                    torch.tensor(state) ,
#                    torch.tensor(spread_time),
#                    torch.tensor(Y)))
#        return input_
#     
#    
#    def __getitem__(self, index):
#        batch=self.data[index]
#        return batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6]
#
#    def __len__(self):
#        return len(self.data) 


#########################model##############################
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from collections import OrderedDict
from torch.nn.modules.module import Module
from torch_geometric.nn import GraphConv,SAGEConv,GCNConv,GATConv,TAGConv,GINConv

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        '''
        nfeat:输入x特征矩阵维度
        nhid：中间层维度
        nclass：输出特征维度
        dropout：dropout的比例
        
        '''
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
#        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)        

class Mycluster(nn.Module):
    def __init__(self, nfeat, hidden, nclass):
        '''
        nfeat:输入x特征矩阵维度
        hidden：中间层维度
        nclass：分类数
        
        '''
        super(Mycluster, self).__init__()

        self.gc1 = GraphConvolution(nfeat, hidden)
        self.mlp2=nn.Linear(hidden, nclass)
        

    def forward(self, x,adj):
        x = F.relu(self.gc1(x, adj))
        x=self.mlp2(x)
        return F.relu(x)    #n,nclass
# class Centrility(nn.Module):
#     def __init__(self, nfeat, hidden,node_num):
#         '''
#         nfeat:输入x特征矩阵维度
#         hidden：中间层维度
#         node_num：节点数
        
#         '''
#         super(Centrility, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, hidden)
#         self.mlp1=nn.Linear(node_num*hidden, node_num)
#         self.mlp1.apply(self._init_weights)
        
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=0.002)


#     def forward(self, x,adj):
#         x = self.gc1(x, adj)
#         x=torch.flatten(x,0)
#         x=self.mlp1(x)
#         return F.log_softmax(x,dim=-1)    #n,nclass



class RumorDetect(nn.Module):
    def __init__(self, node_num,nfeat, nhid, nclass, dropout,device,num_clusters=25,hidden=32):
        '''
        node_num图中节点总数
        nfeat:输入x特征矩阵维度
        nhid：中间层维度
        nclass：输出特征维度
        dropout：dropout的比例
        
        '''
        super(RumorDetect, self).__init__()
        
        # self.w1=nn.Linear(node_num, node_num) 
        # self.w2=nn.Linear(node_num, node_num)
        # self.w3=nn.Linear(node_num, node_num)
        

        self.gcn = GCN(nfeat, nhid, nclass, dropout)
        self.norm= nn.BatchNorm1d(nclass)
        
        self.norm1= nn.BatchNorm1d(node_num)
        self.mlp1=nn.Linear(node_num, 2048*2)#用于最后输出
       
        self.norm2= nn.BatchNorm1d(2048*2)
        self.mlp2=nn.Linear(2048*2, node_num*2)
        
        self.mycluster=Mycluster(nclass, hidden, num_clusters)
        self.clusternorm1= nn.BatchNorm1d(num_clusters)
        
        self.Centrility=nn.Linear(num_clusters, 1)
        self.num_clusters=num_clusters
        self.dropout = dropout
        self.device=device
        self.node_num=node_num
        self.k=2
        


    def forward(self, x, a1,a2,a3,state):
        '''
       x  特征矩阵  b,n,nfeat
       a1, 朋友关系矩阵 b,n,n
       a2,转推关系矩 b,n,n
       a3 评论关系矩 b,n,n
       A=a1W1+a2W2+a3W3
       
       a 超图邻接矩阵
       x, 经过学习的节点特征矩阵
       state_orginal  节点原始状态 b,n,1
       '''
#        print('a1:',a1)
#        print('a2:',a2)
#        print('a3:',a3)
        adj=F.relu(a1+a2+a3)
        # adj=F.relu(self.w1(a1)+self.w2(a2)+self.w3(a3))
#        adj=self.normalize_adj_torch(adj)
        out_seq = []
#        newgraph=[]
#        cluster_ids=[]
        C_s=[]#各个图中聚类中心性分数
        T_mask=[]#各个图中最可疑的类的标记
        c_id=[]#最可疑类的id
        A_new=[]
        for t,Ahat in enumerate(adj):
            node_embs = x[t]
            # print('Ahat',node_embs)
#            a1=(Ahat >0.3).nonzero()
            A_new.append(Ahat)
#            print('model_size',a1.size())
            node_embs=self.gcn(node_embs, Ahat)#n,class
            node_embs=self.norm(node_embs)
            #聚类
            myclusters_x =self.mycluster(node_embs,Ahat) # 预测分类n,cluster
            myclusters_x=self.clusternorm1(myclusters_x)
            
            cluster_ids_x=torch.argmax(F.softmax(myclusters_x,dim=1),dim=1)#n
#            print('myclusters_x.size:',myclusters_x)
            #计算图的中心性
#            ss=torch.div(Ahat.sum(dim=0),Ahat.size()[1]).unsqueeze(1)#度中心性torch.div(a.sum(dim=0),a.size()[1])
            #得到各个节点的中心性分数
            ss=F.sigmoid(self.Centrility(myclusters_x))
            cluster_score=[]#聚类分数
            for i in range(self.num_clusters):
                seletx_i=torch.eq(cluster_ids_x,i).unsqueeze(1).float()#n,1
                s_i=torch.mm(ss.t(),seletx_i).squeeze(1)#拿到当前类的中心性分数，实际是当前类中所有点中心性的总和
#                print('中间变量',s_i)
                s_i=torch.div(s_i,torch.sum(seletx_i,dim=0).clamp_(1))
                # print('单个聚类的中心性s_i：',s_i)
                cluster_score.append(s_i)
            cluster_score=torch.stack(cluster_score)#num_clusters,1
            C_s.append(cluster_score)
        
            #得到中心值最大的类的所有点位置的标记，属于可疑类的位置标记为1，max_cluster_mask->n*1
#            max_cluster_id=torch.argmax(cluster_score,0)#目标类的id (1,)
            max_cluster_id=torch.topk(cluster_score,self.k,0)[1].squeeze(1)
            c_id.append(max_cluster_id)
            max_cluster_mask=torch.zeros_like(cluster_ids_x)#torch.zeros(cluster_ids_x.size()[0],1).to(self.device)
            for c_i in max_cluster_id:
                max_cluster_mask=max_cluster_mask+torch.eq(cluster_ids_x,c_i).float()
            max_cluster_mask=max_cluster_mask.unsqueeze(1)    
            # print('max_cluster_mask',max_cluster_mask)
            T_mask.append(max_cluster_mask)
            out_seq.append(node_embs)
            del node_embs,cluster_ids_x,ss,cluster_score,seletx_i,s_i,max_cluster_id,max_cluster_mask
            gc.collect()
        x=torch.stack(out_seq)
        del out_seq
        gc.collect()
        C_s=torch.stack(C_s)# batch-size,num_clusters,1
        T_mask=torch.stack(T_mask)# batch-size,n,1
        c_id=torch.stack(c_id)# batch-size,1
        
        
        s= F.relu(torch.matmul(x,x.transpose(1, 2)))#节点间传播概率 b,n,n
        output= torch.matmul(s,state).squeeze(2)#原始节点感染状态 output b,n,1
        output=self.norm1(output)
        output=F.relu(self.mlp1(output))
        output=self.norm2(output)
        
        output=self.mlp2(output)
        output=F.sigmoid(output.view(-1,self.node_num,2))
        return C_s,T_mask,c_id,output,A_new

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.lg=nn.LogSigmoid()
    def forward(self, inputs, targets,device):
        '''
        @params:
            inputs: 经过relu层后得到的中心性分数，b,cluster_num,1
            targets: 目标聚类,b,3 数字代表具体聚类id
        @return:
            res: 平均到每个label的loss
        '''
#        print('inputs',inputs)
#        print('targets',targets)
        inputs= self.lg(inputs).squeeze(2)# b,cluster_num
        mask=torch.zeros(inputs.size()[0], inputs.size()[1]).to(device).scatter_(1,targets,1)  #b,cluster_num 对应目标聚类位置为1
        t_score=torch.mul(inputs,mask).sum(dim=1).unsqueeze(1).clamp_(1e-6)#b,1
#        print('中心性最高的节点得分',t_score)
         
        res = -(torch.div( t_score,inputs).sum(dim=1)).mean()#标量
        del inputs,mask,t_score
        gc.collect()
        return res
####################model end###########################

class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self,event, node_feature,thread,global_graph,istrain=True):
        self.istrain=istrain
        
        self.event=event
        self.node_feature=node_feature
        self.thread=thread
        
        self.graph=global_graph
        data=fetch_tweets(self.event)
        data['created_1']=data[['created']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        self.static_f=self.node_feature.sort_values(by = ['graphid'],axis = 0,ascending = True)[basef1].values
        self.temdata=data#self.threadprocess()
      
    def __getitem__(self, index):
        '''
        把数据处理成样本 sample:(A,X,state,Y)
        A 邻接矩阵，[A1,A2,A3]分别表示朋友关系矩阵、转推关系矩、评论关系矩
        X 用户节点特征矩阵
        state 用户传播谣言状态，（即用户是否发推,用户传播了谣言为1，没有传播为-1）
        spread_time  归一化的谣言散布时间（虽然经过处理但是时间的先后顺序没变）
        Y 当前用户是否是源头
        
        '''
#        print(index)
        batch=self.thread[index] #一个thread 列表
        temp=self.temdata[self.temdata['thread']==batch]
        A_3=self.graph.csv_dict(temp,'in_reply_user','user_id')
        A_3=self.graph.getadju(A_3)
#        A_3=self.graph.getAdjNormornize(self.graph.getadju(A_3))
        A_1,A_2=self.graph.process_Adjacent(self.event)
            
        #构造state
        state=-np.ones((self.graph.node_num,1))
        spread_time=np.zeros((self.graph.node_num,1))
        #u_sum_in[np.where(u_sum_in == 0)] = 1
        for u_uid in temp['user_id'].unique():
            u=self.graph.n_g[u_uid]
            state[u][0]=1
            spread_time[u][0]=temp[temp['user_id']==u_uid]['created_1'].values[0]
            
        #构造X 包含静态部分和动态变化部分
        X=np.hstack((self.static_f,state)).astype(float)#self.graph.node_num ,len(basef1)+1
#            print(static_f.shape,'state',state.shape,'   X.shape',X.shape)
        #构造Y
        Y=np.zeros((self.graph.node_num,1))
        #找到源头用户
        for u in temp[temp['is_source_tweet']==1]['user_id'].unique():
            Y[self.graph.n_g[u]][0]=1
        
        
        return (    torch.tensor(A_1, dtype=torch.float32),
                    torch.tensor(A_2, dtype=torch.float32),
                    torch.tensor(A_3, dtype=torch.float32),
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(state, dtype=torch.float32) ,
                    torch.tensor(spread_time, dtype=torch.float32),
                    torch.tensor(Y))

    def __len__(self):
        return len(self.thread) 



step=5
train_batch_size=16
output_dir='model'



def infer(model,eval_dataset,eval_batch_size=2,topk=10):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=0)
#    eval_loss = 0.0
    pre_user=[]#预测的源头用户
    Groud_user=[]#真实的源头用户情况
    A_graph=[]#邻接矩阵,按边存储
    model.eval()      
    for batch in eval_dataloader:
        if eval_dataset.istrain:
            A1,A2,A3,X,S,Spread_Time,Y =(x.to(device) for x in batch)
            Groud_user.append(Y.squeeze(2).cpu().numpy()) #b,n
            del Spread_Time,Y
            gc.collect()
        else:
            A1,A2,A3,X,S,Spread_Time=(x.to(device) for x in batch)
#        A_graph.append(A1.cpu().numpy())
#            Groud_user.append(Y.squeeze(2).cpu().numpy()) #b,n
        del batch
        gc.collect()
        with torch.no_grad():
            _,T_mask,_,state_orginal,A_new  = model(X,A1,A2,A3,S)#state_orginal ,b,n,2
            del A1,A2,A3,S
            gc.collect()
            state_orginal=F.softmax(state_orginal,dim=2)
#            A_graph.extend(A_new)
            A_graph.extend(A_new)
            #找到发布时间top的用户节点并预测为源
#            source_user=torch.topk(torch.mul(T_mask,Spread_Time),topk,1,False)[1]# batch_size,topk,1
            pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)
            source_user=torch.argmax(pred,dim=2)#b,n
#            source_user=torch.zeros(source_user.size()[0], X.size()[1],1).to(device).scatter_(1,source_user,1)#batch_size,n,1
            pre_user.append(source_user.cpu().numpy())
        
    
    pre_user=np.concatenate(pre_user,0)
#    A_graph=np.concatenate(A_graph,0)
    if eval_dataset.istrain:
        Groud_user=np.concatenate(Groud_user,0)
#        print('infer_size;pre_user:',pre_user,'Groud_user:',Groud_user,'A_graph:',A_graph)
        return pre_user,Groud_user,A_graph
    else:
        return pre_user,A_graph

#from _tkinter import _flatten
 
 




def eval(pre_user,Groud_user,A_graph,N):
    '''
    Groud_user  len 
    pre_user len topk
    A_graph  len ,n,n
    '''
    #求出acc
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    results={}
    accu=0
    precision=0
    recall=0
    f1_micro=0
    dis=0
    for i,t in enumerate(Groud_user):
        accu=accu+accuracy_score(t,pre_user[i])
        precision=precision+precision_score(t,pre_user[i])
        recall=recall+recall_score(t,pre_user[i])
        f1_micro=f1_micro+f1_score(t,pre_user[i])
        
        dis=dis+error_distance(i,A_graph[i],pre_user[i],t)
    results['accuracy']=accu/(1.0 * len(Groud_user))
    results['precision']=precision/(1.0 * len(Groud_user))
    results['recall']=recall/(1.0 * len(Groud_user))
    results['eval_f1']=f1_micro/(1.0 * len(Groud_user))
    results['eval_error_distance']=dis/(1.0 * len(Groud_user))
    
#    for i,t in enumerate(Groud_user):
#        myp=np.where(t==1)[0]#具体的类
#        for p in myp:
#            f1=f1+f1_score(t,pre_user[i])
#            f1_macro=f1_macro+f1_score(t,pre_user[i], average='macro')
#            f1_micro=f1_micro+f1_score(t,pre_user[i], average='micro')
#            dis=dis+error_distance(A_graph[i].cpu().numpy(),pre_user[i],t)
#    results['eval_f1']=f1
#    results['eval_f1_macro']=f1_macro
#    results['eval_f1_micro']=f1_micro
#    results['eval_error_distance']=dis        
    return results
def reload(model,output_dir):
        #读取在验证集结果最好的模型
    load_model_path=os.path.join(output_dir, "pytorch_model_gcn.bin")
    logger.info("Load model from %s",load_model_path)
    model_to_load = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_load.load_state_dict(torch.load(load_model_path))   
    return model
def one_hot(x, class_count):
    
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
    return torch.eye(class_count)[x,:]    


def train(model,train_dataset,train_batch_size,epoch,dev_dataset,topk=10,warm_up=0,
          display_steps=5,eval_steps=20,max_grad_norm=1.0,lr=0.01,output_dir='model',event='germanwings-crash',logprintfile=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=0)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epoch)
    logger.info("学习率：",lr)
    print("  Num examples =",len(train_dataset),"  Num Epochs =",epoch,"\n",file=logprintfile)
    optimizer =torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fun=CrossEntropyLoss()
    loss_f2=MyLoss()
    global_step = 0
    model.zero_grad() 
    tr_loss,best_f1,avg_loss = 0.0, 0.01,0.0
    for idx in range(epoch):     
        tr_num=0
        train_loss=0
        model.train()
        for step, batch in enumerate(train_dataloader):
            
            A1,A2,A3,X,S,Spread_Time,Y =(x.to(device) for x in batch)

            C_s,T_mask,c_id,state_orginal,_  = model(X,A1,A2,A3,S)#state_orginal ,b,n,2
            del batch,A1,A2,A3,S
            gc.collect()
            
            if (step+1)>warm_up:
#                #找到发布时间top的用户节点并预测为源
#                source_user=torch.topk(torch.mul(T_mask,Spread_Time),topk,1,False)[1]# batch_size,topk,1
#                #源节点整成one-hot
#                source_user=torch.zeros([source_user.size()[0], X.size()[1],1],requires_grad=True).to(device).scatter_(1,source_user,1)#batch_size,n,1
#                #微调
#                pred=torch.mul(source_user.repeat(1,1,2),state_orginal) #b,n,2
#                #中心性损失，目标是使预测的源头类尽量大于其他类的中心性min -sum(log(Ci-Cj))
#                loss =loss_fun(pred.view(pred.size()[0]*X.size()[1],-1) ,Y.view(Y.size()[0]*X.size()[1]).long())
                #找原始情况也是感染的
                pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)#这样就把其他聚类给屏蔽了
                # pred=F.log_softmax(pred,dim=2)
                
                loss =loss_fun(pred.view(pred.size()[0]*X.size()[1],-1) ,Y.view(Y.size()[0]*X.size()[1]).long())
            else:
#                #粗调
#                #源节点预测损失
                loss =loss_f2(C_s,c_id,device)
#                
#            source_user=torch.topk(torch.mul(T_mask,Spread_Time),topk,1,False)[1]# batch_size,topk,1
            
#            #找原始情况也是感染的
#            pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)#这样就把其他聚类给屏蔽了
#          
#            
#            loss2 =loss_fun(pred.view(pred.size()[0]*X.size()[1],-1) ,Y.view(Y.size()[0]*X.size()[1]).long())
            
         #粗调
#            loss1 =loss_f2(C_s,c_id,device)
#            print('pred',(torch.argmax(pred,dim=1)==1).nonzero(),'Y:',Y)
#            Y=torch.argmax(Y.squeeze(2),dim=1)#b,n->b
            
#            print(loss2.requires_grad)
#            loss = torch.add(loss1,loss2)
            optimizer.zero_grad()
#            loss=loss1
#            print('step:',step,'  loss1：',loss1,'loss2',loss2)
#            print('loss1',loss1,'loss2',loss2,"\n",file=logprintfile)
#            loss = loss.requires_grad_()
            loss.backward()#先写到这里，后续再补充！！
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)         
            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            #输出log
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            
            if (step+1) % display_steps == 0:
                logger.info("At Training:  epoch {} step {} loss {}".format(idx,step+1,avg_loss))
                print("At Training:  epoch {} step {} loss {}".format(idx,step+1,avg_loss),"\n",file=logprintfile)
            optimizer.step()
            
            global_step += 1
            #测试验证结果
            if (step+1) % eval_steps == 0 and dev_dataset is not None:
                #输出验证集预测的时间和分类结果
                pre_user,Groud_user,A_graph= infer(model,dev_dataset)
                #输出预测的f1和error distance
                results=eval(pre_user,Groud_user,A_graph,topk)
                #打印结果                  
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,8))                    
                    #保存最好的年龄结果和模型
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        print("  "+"*"*20,"\n",file=logprintfile)
                        logger.info("  Best f1:%s",round(best_f1,8))
                        logger.info("  Best error_distance:%s",round(results['eval_error_distance'],8))
                        print("  Best f1:",round(best_f1,8),"  Best error:",round(results['eval_error_distance'],8),"\n",file=logprintfile)
                        print("  "+"*"*20,"\n",file=logprintfile)
                        logger.info("  "+"*"*20)                          
                        try:
                            os.system("mkdir -p {}".format(output_dir))
                        except:
                            pass
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model_gcn_%s.bin"%event)
                        torch.save(model_to_save.state_dict(), output_model_file)

def main(Istest,event,epoch=2):

    node_feature,train_thread,test_thread,global_graph=getstart(event)
    m_model = RumorDetect(global_graph.node_num,len(basef1)+1, 64, 32, 0.0002,device,30)
    m_model.to(device)
    if not Istest:     
        s=open('GCN_traing_log_%s.txt'%event,'w')
        print("The number of train_thread:",len(train_thread),' The number of test_thread:',len(test_thread))
        #,event, node_feature,thread,global_graph
        train_dataset = TensorDataset(event,node_feature,train_thread,global_graph) 
        vaild_dataset = TensorDataset(event,node_feature,test_thread,global_graph) 
        
        train(m_model,train_dataset,train_batch_size,epoch,vaild_dataset,output_dir=output_dir,event=event,logprintfile=s)
        s.close()
        del train_dataset,vaild_dataset
        gc.collect()
    
    #
    logger.info("训练结束  "+"*"*20)
#    logger.info("开始预测。。。。")
#     
#    test_dataset = TensorDataset(windowsize,step,his=test_set) 
#    model=reload(m_model,output_dir+'_'+event)
#    result_probs,clasify= infer(model,test_dataset)#需要把result_probs转换成时间字符串类型的
#    res_df=pd.DataFrame({'car_no':test_dataset.car_id,'label':clasify,'timess':result_probs})
#    res_df.to_csv('sub_self_attention_1.csv',index=False)
#    transformersubmit(res_df,test_set)

if __name__ == "__main__":
    events=[    "germanwings-crash",
             "sydneysiege",
            "ottawashooting",
            "ferguson",
            "charliehebdo",
        ]

    for event in events:
        Istest=False
        main(Istest,event)    
    
    