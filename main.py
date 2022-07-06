# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:24:30 2021
pip install transformers==2.8.0 pandas gensim scikit-learn filelock gdown
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
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
from kmeans import lloyd
import warnings
warnings.filterwarnings("ignore")
# torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.FloatTensor)
from tqdm import tqdm,trange
from torch.nn import CrossEntropyLoss, MSELoss
#from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def getAdjNormornize(A):
    '''
    矩阵归一化,Laplace
    '''
    R = np.sum(A, axis=1)
    R_sqrt = 1/np.sqrt(R) 
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(A.shape[0])
    return I - D_sqrt * A * D_sqrt

def getdataForpro(dataset,num_source,train_pro=0.9):
    with open("../%s/source_%d/adj.pkl"% (dataset,num_source), 'rb') as f:
        A=pickle.load(f)
    Threadlists=[i for i in range(1000)]
    random.shuffle(Threadlists)
    train_thread=Threadlists[:int(len(Threadlists)*train_pro)]
    test_thread=list(filter(lambda x: x not in train_thread, Threadlists))
    print('训练样本',len(train_thread),'测试样本:',len(test_thread))
    
    return train_thread,test_thread,A
   



#############model########################
    


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


class GCN(nn.Module):
    def __init__(self,modelname, nfeat, nhid, nclass, dropout):
        '''
        modelname:使用的模型名称
        nfeat:输入x特征矩阵维度
        nhid：中间层维度
        nclass：输出特征维度
        dropout：dropout的比例
        ['GNN','GCN','GAT','GraphSAGE','TAGCN']
        '''
        super(GCN, self).__init__()
        if modelname=='GNN':
            self.gc1 = GraphConv(nfeat, nhid)
            self.gc2 = GraphConv(nhid, nclass)
        elif modelname=='GCN':
            self.gc1 = GCNConv(nfeat, nhid)
            self.gc2 = GCNConv(nhid, nclass)
        elif modelname=='GAT':
            self.gc1 = GATConv(nfeat, nhid)
            self.gc2 = GATConv(nhid, nclass)
        elif modelname=='GraphSAGE':
            self.gc1 = SAGEConv(nfeat, nhid)
            self.gc2 = SAGEConv(nhid, nclass)
        elif modelname=='TAGCN':
            self.gc1 = TAGConv(nfeat, nhid)
            self.gc2 = TAGConv(nhid, nclass)
        elif modelname=='GINConv':
            self.gc1 = GINConv(self.MLP(nfeat, nhid))
            self.gc2 = GINConv(self.MLP(nhid, nclass))
        self.dropout = dropout
    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)
            


class Mycluster(nn.Module):
    def __init__(self, modelname,nfeat, hidden, nclass):
        '''
        nfeat:输入x特征矩阵维度
        hidden：中间层维度
        nclass：分类数
        
        '''
        super(Mycluster, self).__init__()
        if modelname=='GNN':
            self.gc1 = GraphConv(nfeat, hidden)
        elif modelname=='GCN' or modelname=='GINConv':
            self.gc1 = GCNConv(nfeat, hidden)
        elif modelname=='GAT':
            self.gc1 = GATConv(nfeat, hidden)
        elif modelname=='GraphSAGE':
            self.gc1 = SAGEConv(nfeat, hidden)
        elif modelname=='TAGCN':
            self.gc1 = TAGConv(nfeat, hidden)

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
    def __init__(self,modelname, node_num,nfeat, nhid, nclass, dropout,device,num_clusters=2,hidden=64):
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

        self.gcn = GCN(modelname,nfeat, nhid, nclass, dropout)
        self.norm= nn.BatchNorm1d(nclass+nfeat)
        
        self.norm1= nn.BatchNorm1d(node_num)
        self.mlp1=nn.Linear(node_num, 512*4)#用于最后输出
       
        self.norm2= nn.BatchNorm1d(512*4)
        self.mlp2=nn.Linear(512*4, node_num*2)
        
        self.mycluster=Mycluster(modelname,nclass+nfeat, hidden, num_clusters)
        self.clusternorm1= nn.BatchNorm1d(num_clusters)
        
        self.Centrility=nn.Linear(nclass+nfeat, 1)
        self.num_clusters=num_clusters
        self.dropout = dropout
        self.device=device
        self.node_num=node_num
        self.k=2
        


    def forward(self, x,state, a1,a2=None,a3=None,kmeans=False,Centrilityname="us"):
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
        if a2==None:
            adj=a1#F.relu(a1)
        else:
            adj=F.relu(a1+a2+a3)
        a1=(adj >0).nonzero().t()
        
        
        # adj=F.relu(self.w1(a1)+self.w2(a2)+self.w3(a3))
#        adj=self.normalize_adj_torch(adj)
        out_seq = []
#        newgraph=[]
#        cluster_ids=[]
        C_s=[]#各个图中聚类中心性分数
        T_mask=[]#各个图中最可疑的类的标记
        c_id=[]#最可疑类的id
        # A_new=[]
        for t,embs in enumerate(x):
            # print('Ahat',node_embs)
            
            # A_new.append(a)
#            print('model_size',a1.size())
            node_embs=self.gcn(embs, a1)#n,class
            node_embs=torch.cat((embs,node_embs),1)#n,(class+nfeat)
            node_embs=self.norm(node_embs)
            #聚类
            if kmeans:
                cluster_ids_x, _ = lloyd(node_embs, self.num_clusters,device)

            else:

                myclusters_x =self.mycluster(node_embs,a1) # 预测分类n,cluster
                myclusters_x=self.clusternorm1(myclusters_x)

                cluster_ids_x=torch.argmax(F.softmax(myclusters_x,dim=1),dim=1)#n
#            print('myclusters_x.size:',myclusters_x)
            #计算图的中心性
#            ss=torch.div(Ahat.sum(dim=0),Ahat.size()[1]).unsqueeze(1)#度中心性torch.div(a.sum(dim=0),a.size()[1])
            
            
            #得到各个节点的中心性分数
            if Centrilityname=="degree":
            	ss=torch.div(adj.sum(dim=0),adj.size()[1]).unsqueeze(1)#adj
            elif Centrilityname=="eig":#特征向量中心性
            	evals,_=torch.eig(adj,eigenvectors=False)
            	ss=evals[:,0].unsqueeze(1)

            else:
            	#本文设计的中心性计算
            	ss=F.sigmoid(self.Centrility(node_embs))#[34, 1]
            	# print("ss",ss.size())
            
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
        # output=self.norm1(output)
        output=F.relu(self.mlp1(output))
        # output=self.norm2(output)
        
        output=self.mlp2(output)
        output=output.view(-1,self.node_num,2)
        # output=F.sigmoid(output.view(-1,self.node_num,2))
        return C_s,T_mask,c_id,output#,A_new



#############model end########################

def readfacebook(FILE_NAME = '../facebook/facebook_combined.txt'):
    edges = []
    with open(FILE_NAME) as netfile:
        print('file opened')
        for i, line in enumerate(netfile):
            words = line.split()
            edges.append([int(words[0]), int(words[1])])
        print('Reading edges finished')
    return np.array(edges, dtype=np.int)



class G_TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self,datasetname,thread,global_graph,num_source,istrain=True,alpha=0.4):
        self.istrain=istrain
        
        self.datasetname=datasetname
        self.thread=thread
        self.alpha=alpha
        self.num_source=num_source
       
        self.A=self.graph=np.array(global_graph)#邻接矩阵

        self.S=getAdjNormornize(self.A) 
         #填空值
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.S=imp.fit_transform(self.S)
       
            
      
    def __getitem__(self, index):
        '''
        把数据处理成样本 sample:(S,X,Y)
        S 邻接矩阵
        X 用户节点特征矩阵
        state 用户传播谣言状态，（即用户是否发推,用户传播了谣言为1，没有传播为-1）
        spread_time  归一化的谣言散布时间（虽然经过处理但是时间的先后顺序没变）
        Y 当前用户是否是源头
        
        '''
#        print(index)
        batch=self.thread[index] #一个thread 列表
        with open("../%s/source_%d/"% (self.datasetname,self.num_source)+"Thread_%d.pkl"% batch, 'rb') as f:
            data=pickle.load(f)
        '''
        {'source_id':list_s,
                         'net_state':net_state,
                         }, 
        '''
        #构造state
        state=-np.ones((len(self.graph),1))
        
        v34=np.zeros((len(self.graph),2))
        #u_sum_in[np.where(u_sum_in == 0)] = 1
        for u,s in enumerate(data['net_state']):
            if s==1:
                state[u][0]=1
                v34[u][1]=0
                v34[u][0]=1
            else:
                state[u][0]=-1
                v34[u][0]=0
                v34[u][1]=-1
        d2=np.dot((1-self.alpha)*np.linalg.inv(np.eye(state.shape[0])-self.alpha*self.S),state)
        X=np.hstack((state,d2))
        d3=np.dot((1-self.alpha)*np.linalg.inv(np.eye(state.shape[0])-self.alpha*self.S),v34[:,0])
        X=np.hstack((X,np.expand_dims(d3,axis=1)))
        d4=np.dot((1-self.alpha)*np.linalg.inv(np.eye(state.shape[0])-self.alpha*self.S),v34[:,1])
        X=np.hstack((X,np.expand_dims(d4,axis=1))).astype(float)
#        print('X:*******',X)
#            print(static_f.shape,'state',state.shape,'   X.shape',X.shape)
 
        if self.istrain:
                   #构造Y
            Y=np.zeros((len(self.graph),1))
            #找到源头用户
            for u in data['source_id']:
                Y[u][0]=1
                
#            Y=self.graph.n_g[temp[temp['is_source_tweet']==1]['user_id'].unique()[0]]
            return   (
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(state, dtype=torch.float32) ,
                    torch.tensor(Y))
        else:
            return (   
                        torch.tensor(X, dtype=torch.float32),
                        torch.tensor(state, dtype=torch.float32) 
                        )
    def __len__(self):
        return len(self.thread) 



#step=20
train_batch_size=16
output_dir='model'



def infer(model,eval_dataset,eval_batch_size=4):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=0)
#    eval_loss = 0.0
    pre_user=[]#预测的源头用户
    Groud_user=[]#真实的源头用户情况
    model.eval()      
    A=torch.tensor(eval_dataset.A,dtype=torch.float32)
    # adj_nor=torch.tensor(eval_dataset.S,dtype=torch.float32)
    for batch in eval_dataloader:
        if eval_dataset.istrain:
            X,state,Y =(x.to(device) for x in batch)
            Groud_user.append(Y.squeeze(2).cpu().numpy()) #b,n
            del Y
        else:
            X =(x.to(device) for x in batch)
#            Groud_user.append(Y.squeeze(2).cpu().numpy()) #b,n
        del batch
        with torch.no_grad():
            
            _,T_mask,_,state_orginal=  model(X,state,A.to(device))
            state_orginal=F.softmax(state_orginal,dim=2)
            pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)
            source_user=torch.argmax(pred,dim=2)#b,n
#            source_user=torch.zeros(source_user.size()[0], X.size()[1],1).to(device).scatter_(1,source_user,1)#batch_size,n,1
            pre_user.append(source_user.cpu().numpy())# batch_size,topk
        del X
        gc.collect()
    pre_user=np.concatenate(pre_user,0)
    if eval_dataset.istrain:
        Groud_user=np.concatenate(Groud_user,0)
#        print('infer_size;pre_user:',pre_user,'Groud_user:',Groud_user,'A_graph:',A_graph)
        return pre_user,Groud_user,A
    else:
        return pre_user,A

#from _tkinter import _flatten
 
 
def calulateF(pre,label):
    pn=np.where(pre==1)[0]
    lan=np.where(label==1)[0]
    count=0
    for p in pn:
        if p in lan:
            count += 1
    precision = count / len(lan)
    recall = count /(len(pn)+1)
    f_score = (2 * precision * recall) / (precision + recall+ 0.001)
    return precision,recall,f_score


def eval(pre_user,Groud_user,A_graph):
    '''
    Groud_user  len ,n
    pre_user len n
    A_graph n,n
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
        pt,rt,ft=calulateF(pre_user[i],t)
        precision=precision+pt#precision_score(t,pre_user[i])
        recall=recall+rt#recall_score(t,pre_user[i])
        f1_micro=f1_micro+ft#f1_score(t,pre_user[i])
        
        dis=dis+error_distance(A_graph,pre_user[i],t)
    results['accuracy']=accu/(1.0 * len(Groud_user))
    results['precision']=precision/(1.0 * len(Groud_user))
    results['recall']=recall/(1.0 * len(Groud_user))
    results['eval_f1']=f1_micro/(1.0 * len(Groud_user))
    results['eval_error_distance']=dis/(1.0 * len(Groud_user))
#    results['accuracy']=accuracy_score(Groud_user,pre_user)
#    results['precision']=precision_score(Groud_user,pre_user,average='macro')
#    results['recall']=recall_score(Groud_user,pre_user,average='macro')
#    results['eval_f1']=f1_score(Groud_user,pre_user,average='macro')
#    results['eval_error_distance']=error_distance(A_graph[0],pre_user,Groud_user)/len(Groud_user)
    
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
def reload(model,output_dir,modelname):
        #读取在验证集结果最好的模型
    load_model_path=os.path.join(output_dir, "pytorch_model_%s.bin"% modelname)
    logger.info("Load model from %s",load_model_path)
    model_to_load = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_load.load_state_dict(torch.load(load_model_path))   
    return model
def one_hot(x, class_count):
    
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
    return torch.eye(class_count)[x,:]    
def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)




def train(model,train_dataset,train_batch_size,epoch,dev_dataset,
          display_steps=2,eval_steps=3,max_grad_norm=1.0,lr=0.0001,l2_alpha=0.1,output_dir='model',event='germanwings-crash',logprintfile=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=0)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epoch)
    print("  Num examples =",len(train_dataset),"  Num Epochs =",epoch,"\n",file=logprintfile)
    

    optimizer =torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8,weight_decay=0.08)
#    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataloader)*epoch*0.2),num_training_steps=int(len(train_dataloader)*epoch)) 

#    wec=torch.FloatTensor([1/(train_dataset.graph.node_num-1),1])
    loss_fun=CrossEntropyLoss()
#    loss_fun=nn.MultiLabelSoftMarginLoss(reduction='mean')
    logger.info("***** Running training *****")
    loss_fun.to(device)
    global_step = 0
    adj=torch.tensor(train_dataset.A,dtype=torch.float32)
#    print('S.dimension',S.size())
    model.zero_grad() 
    tr_loss,best_f1,avg_loss = 0.0, 0,0.0
    for idx in range(epoch):     
        tr_num=0
        train_loss=0
        model.train()
        for step, batch in enumerate(train_dataloader):
            
            X,state,Y =(x.to(device) for x in batch)
#            print("模型输入A：",A)
#            print("模型输入X：",X)
            start = time.time()
            
            C_s,T_mask,c_id,state_orginal  =model(X,state,adj.to(device))#state_orginal ,b,n,2
            end = time.time()
            # print("train time:",end-start)
            del batch,state
            gc.collect()
#            print('原始输出',pred)
#            pred=pred.squeeze(2)
#            print(pred)#b,n,1
            pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)#这样就把其他聚类给屏蔽了
                # pred=F.log_softmax(pred,dim=2)
                
            loss =loss_fun(pred.view(pred.size()[0]*X.size()[1],-1) ,Y.view(Y.size()[0]*X.size()[1]).long())
            optimizer.zero_grad()

            loss.backward()#先写到这里，后续再补充！！
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)         
            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            #输出log
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,8)
            
            if (step+1) % display_steps == 0:
                logger.info("At Training:  epoch {} step {} loss {}".format(idx,step+1,avg_loss))
                print("At Training:  epoch {} step {} loss {}".format(idx,step+1,avg_loss),"\n",file=logprintfile)
            optimizer.step()
            optimizer.zero_grad()
#            scheduler.step()  
            global_step += 1
            
            if (global_step) % eval_steps == 0 and dev_dataset is not None:
            # #测试验证结果
            # if (step+1) % eval_steps == 0 and dev_dataset is not None:
                #输出验证集预测的结果
                pre_user,Groud_user,A_graph= infer(model,dev_dataset)
                #输出预测的f1和error distance
                results=eval(pre_user,Groud_user,A_graph)
                
                for key, value in results.items():
                    logger.info("step %d 测试结果  %s = %s",global_step, key, round(value,8))      
                    print("step {} 测试结果  {} = {}".format(global_step,key, round(value,8)),"\n",file=logprintfile)
                
                #保存最好的年龄结果和模型
                if results['eval_f1']>best_f1:
                    best_f1=results['eval_f1']
                    logger.info("  "+"*"*20)  
                    print("  "+"*"*20,"\n",file=logprintfile)
                    #打印结果                  
                    # for key, value in results.items():
                    #     logger.info("测试结果  %s = %s", key, round(value,8))      
                    #     print("测试结果  {} = {}".format(key, round(value,8)),"\n",file=logprintfile)
                    logger.info("  Best f1:%s",round(best_f1,8))
                    logger.info("  Best error_distance:%s",round(results['eval_error_distance'],8))
                    print("  Best f1:",round(best_f1,8),"\n",file=logprintfile)
                    print("  "+"*"*20,"\n",file=logprintfile)
                    logger.info("  "+"*"*20)                          
                    # try:
                    #     os.system("mkdir -p {}".format(output_dir))
                    # except:
                    #     pass
                    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    # output_model_file = os.path.join(output_dir, "pytorch_model_gcn_%s.bin"%event)
                    # torch.save(model_to_save.state_dict(), output_model_file)





def main(Istest,dataset,modelname,i,epoch=20):
    if dataset=='twitter':
        print('please run the train_GCN.py etc.')
    elif dataset=='public_karate'or dataset=='facebook' :
        num_source=[1,2,3,5][1]
        train_thread,test_thread,A=getdataForpro(dataset,num_source)
        m_model = RumorDetect(modelname,len(A),4, 512, 512, i,device)
        m_model.to(device)
        if not Istest: 
            print(modelname,"current dataset:",dataset,"source_num",num_source)
            s=open('num_source_%d'% num_source+"_"+modelname+'_traing_log_%s.txt'% dataset,'w')
            print("The number of train_thread:",len(train_thread),' The number of test_thread:',len(test_thread))
            #,event, node_feature,thread,global_graph
            train_dataset = G_TensorDataset(dataset,train_thread,A,num_source) 
            vaild_dataset =  G_TensorDataset(dataset,test_thread,A,num_source) 
            
            train(m_model,train_dataset,train_batch_size,epoch,vaild_dataset,output_dir=output_dir,event=dataset,logprintfile=s)
            s.close()
            del train_dataset,vaild_dataset
            gc.collect()
        
        
    #
    logger.info("训练结束  "+"*"*20)
    print(modelname," on",dataset, "  !!")

if __name__ == "__main__":
    
    dataset=['twitter','public_karate','facebook'][1]
    modelname=['GNN','GCN','GAT','GraphSAGE','TAGCN','GINConv'][4]
    
    for i in [0.02]:#,0.5,0.8]:
        print('dropout rate',i)
        main(False,dataset,modelname,i)
    
    # main(False,dataset,modelname)
    
    
    
    # for mm in modelname:
    #     main(False,dataset,mm)    
    
    