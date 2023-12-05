import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import tensorflow as tf
import random
from collections import defaultdict, Counter
import math
from tqdm import tqdm
random.seed(13)
'''
pairs 0 1 2 3
0 2是user
1 3是item

transac 
pair包含transac对

'''
def load_data(path):

    wish = pd.read_csv(path + 'wish_dense.csv').values
    transac = pd.read_csv(path + 'transac_dense.csv').values
    have = pd.read_csv(path + 'have_dense.csv').values

    userIDs,itemIDs = {},{}
    interactions = []
    wishPerU,havePerU = defaultdict(list), defaultdict(list)
    userPeruser = defaultdict(list)

    random.shuffle(transac)
    train = transac[:int(0.85*len(transac))]
    test = transac[int(0.85*len(transac)):]

    itemsPerUser = defaultdict(list)
    usersPerItem = defaultdict(list)

    for d in train: #giver reciever item time
        ug, ur, i, time = d[0],d[1],d[2],d[3]
        if not ug in userIDs: userIDs[ug] = len(userIDs)
        if not ur in userIDs: userIDs[ur] = len(userIDs)
        if not i in itemIDs: itemIDs[i] = len(itemIDs)
        interactions.append((ur,i,time))
        userPeruser[ug].append(ur)
        itemsPerUser[ur].append(i)
        usersPerItem[i].append(ur)
    for d in test:
        ug, ur, i, time = d[0],d[1],d[2],d[3]
        if not ug in userIDs: userIDs[ug] = len(userIDs)
        if not ur in userIDs: userIDs[ur] = len(userIDs)
        if not i in itemIDs: itemIDs[i] = len(itemIDs)
        userPeruser[ug].append(ur)
        itemsPerUser[ur].append(i)
        usersPerItem[i].append(ur)
    for d in wish:
        if d[0] in userIDs and d[1] in itemIDs:
            wishPerU[userIDs[d[0]]].append(itemIDs[d[1]])
    for d in have:
        if d[0] in userIDs and d[1] in itemIDs:
            havePerU[userIDs[d[0]]].append(itemIDs[d[1]])


    return userIDs,itemIDs,interactions,wishPerU,havePerU,itemsPerUser,usersPerItem,userPeruser,transac,test
# read and process data
path = 'data/ratebeer/'

userIDs,itemIDs,interactions,wishPerU,havePerU,itemsPerUser,usersPerItem,userPeruser,transac,test = load_data(path)


len(itemIDs)
#13560
maxw = 20
wish_list = []
for u in userIDs:
    wish_list.append(wishPerU[u][:maxw])
padded_wish = [sublist + [0] * (maxw - len(sublist)) for sublist in wish_list]
padded_wish = tf.convert_to_tensor(padded_wish)

mask = tf.math.greater(padded_wish, 0)

class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.attenI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        u_tf = tf.convert_to_tensor([u])
        i_tf = tf.convert_to_tensor([i])
        wish_att_idx = tf.nn.embedding_lookup(padded_wish, u_tf)
        att_mask = tf.nn.embedding_lookup(mask, u_tf)
        wish_att = tf.gather(self.attenI,wish_att_idx)
        item = tf.nn.embedding_lookup(self.attenI, i_tf)
        A = tf.multiply(item[:, tf.newaxis,:],wish_att)
        mask_float = tf.cast(att_mask, dtype=tf.float32)
        A = tf.multiply(mask_float[:,:, tf.newaxis],A)
        A = tf.reduce_sum(A,1)
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1) + tf.reduce_sum(A,1)[0]
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI) +\
                            tf.nn.l2_loss(self.attenI))

    def atten(self,sampleU, sampleI):
        wish_att_idx = tf.nn.embedding_lookup(padded_wish, sampleU)
        att_mask = tf.nn.embedding_lookup(mask, sampleU)
        wish_att = tf.gather(self.attenI,wish_att_idx)
        item = tf.nn.embedding_lookup(self.attenI, sampleI)
        A = tf.multiply(item[:, tf.newaxis,:],wish_att)
        mask_float = tf.cast(att_mask, dtype=tf.float32)
        A = tf.multiply(mask_float[:,:, tf.newaxis],A)
        A = tf.reduce_sum(A,1)
        return tf.reduce_sum(A,1)



    
    def score(self, sampleU, sampleI):    
        
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        A = self.atten(u, i)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1) + A
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))
items = list(itemIDs.keys())
users = list(userIDs.keys())
def trainingStepBPR(model, interactions, optimizer):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()
'''
bpr + user相似度 + wish + 多次交换 + 时间周期：活跃度
'''
modelBPR = BPRbatch(5, 0.00001)
Optimizer = tf.keras.optimizers.Adam(0.1)
# for i in tqdm(range(100)):
#     obj = trainingStepBPR(modelBPR, interactions, Optimizer)
#     if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))
def in_wish(i,u):    
    if i in wishPerU[u]:
        return 1
    return 0
    
#两个用户之间的相似度
def Jaccard(s1, s2):
    try:
        return len(s1.intersection(s2))/len(s1.union(s2))
    except:
        return 0
    
def Jaccard_v(uu,gg):
    ans = 0
    count = 0
    s = usersPerItem[gg]
    ansm = 0
    for gt in itemsPerUser[uu]:    
        if gt == gg:
            continue
        count += 1
        s1 = usersPerItem[gt]
        ans += Jaccard(s,s1)
        ansm = max(ansm, Jaccard(s, s1))
    try:
        ans = ans/count
    except:
        return 0,ansm
    return ans,ansm


def similarity_u(ui,uj):
    si = set(itemsPerUser[ui])
    sj = set(itemsPerUser[uj])
    return Jaccard(si,sj)


#需要预测的item和user拥有或希望的item的相似度

def recurring(ui,uj,alpha):
    u_number = Counter(userPeruser[ui])
    number = u_number[uj]
    return math.log10(number+10)*alpha
   
def predict(ug,ur,i,model,use_recur = False, use_Su = False):
    s = model.predict(ur,i)
    if use_recur:
        s += recurring(ug,ur,1)
    if use_Su:
        s += similarity_u(ug, ur)
    return s
def auc(model, test_transac,use_recur = False,use_Su = False):
    correct = 0
    for d in test_transac: #giver reciever item time
        ug, ur, i, time = d[0],d[1],d[2],d[3]
        ug,ur,i = userIDs[ug], userIDs[ur],itemIDs[i]
        j = random.choice(items) # negative sample
        while j in itemsPerUser[ur]:
            j = random.choice(items)
        uj = random.choice(users)
        while uj in userPeruser[ur]:
            uj = random.choice(users)
        j = itemIDs[j]
        uj = userIDs[uj]
        s_pos = predict(ug,ur,i,model,use_recur,use_Su)
        s_neg = predict(uj,ur,j,model,False, use_Su)
        if s_pos > s_neg:
            correct += 1
    return correct/len(test_transac)
#auc for bpr + attentioni
random.seed(13)
a0 = auc(modelBPR, test)
a0

#auc for bpr + recurring
random.seed(13)
a0 = auc(modelBPR, test,True)
a0
#0.8737931034482759
#auc for bpr + Similarity user
random.seed(13)
a0 = auc(modelBPR, test,False, True)
a0
#
