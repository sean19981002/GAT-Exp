import dgl
import torch
import torch.nn.functional as F
import torch
import numpy as np
from copy import deepcopy
import random
import time
import matplotlib.pyplot as plt
import json
from functions import *
from eval import *
from gat import *
from tqdm import tqdm
import time

device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

graph_list = Load_GraphList(file_name='graph_list.bin')

# tweet6 -> 8hr hasn't done
t = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
tweet_list = [4, 6, 18]
graph_list = [graph_list[i-1] for i in tweet_list]
#graph_list = [graph_list[i-1] for i in tweet_list]
metrics = evaluation()
start = time.time()

for g, tweet_id in zip(graph_list, tweet_list):
    
    active_matrix = g.ndata['active']
    labels = g.ndata['label']
    features = g.ndata['feature']
    hashtags = g.ndata['hashtags']
    folder = 'all/tweet %d/' % tweet_id
    
    run_time = ['first/', 'second/', 'third/']
    #run_time = ['first/']

    for h in range(0, 5): # time stamp
        for times in run_time:

            # Selecting Inactive users to train and test => 1:9
            active_vec = deepcopy(active_matrix[:, h])
            active_user = (active_vec == 1).nonzero().flatten().tolist()
            inactive_user = (active_vec == 0).nonzero().flatten().tolist()

            # chunk inactive users into 10 batch
            batch = shuffle_chunks(input= inactive_user, batch_num= 5)

            time_path = folder + '%d hr/' % t[h]
            time_path += times


            confusion_dict = dict()
            # Selecting test and train set      
            for i in range(len(batch)):
                    
                f = open(time_path + 'Batch %d.txt' % (i+1), 'a')
                print('Active user   :', len(active_user), file= f)
                print('Inactive user :', len(inactive_user), file= f)

                test_idx = i
                train_idx = [j for j in range(len(batch)) if j!= i]
                test_set = deepcopy(batch[i])
                train_set = deepcopy(active_user)
                for idx in train_idx:
                    train_set += deepcopy(batch[idx])

                test_p, test_n = PN_split(ind= test_set, label= labels)

                print('Test || P: %d || N: %d || P+N: %d' % (len(test_p), len(test_n), len(test_set)), file= f)

                # find Positive and Negative samples from training set
                p, n = PN_split(ind= train_set, label= labels)
                print('Train before NS || P: %d || N: %d || Total: %d' % (len(p), len(n), len(train_set)), file= f)

                # Negative Sampling
                NS = Negative_Sampling(pos_sample= p, neg_samples= n, n_ratio= 3)
                train_set = deepcopy(NS) + deepcopy(p) 

                print('Train after NS || P: %d || N: %d || Total: %d' % (len(p), len(NS), len(train_set)), file= f)

                nodes = deepcopy(train_set) + deepcopy(test_set)
                nodes = sorted(nodes)
                delete_node = set(n) - set(NS)
                delete_node = list(delete_node)
                print("Deleted Nodes: %d" % len(delete_node), file= f)

                # process data graph
                data_g, datag_f, datag_label, datag_active, iso_d = Data_Graph_Process(
                    del_nodes= delete_node, nodes_set= nodes,
                    g= g, feature= features, active_matrix= active_matrix, label= labels, hashtags= hashtags
                )
                print('To GAT:', file= f)
                print('   nodes :', data_g.num_nodes(), file= f)
                print('   edges :', data_g.num_edges(), file= f)
                print('   Isolated :', len(iso_d), file= f)
                
                # Train, Test mask
                train_mask, test_mask = train_test_split(train_set= train_set, test_set= test_set, nodes= nodes)

                # run model
                net = GAT(data_g.to(device= device), in_dim=datag_f.size()[1], hidden_dim=16, out_dim=2,num_heads=3)
                
                # create optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
                x = list()
                y = list()
                train_res = list()
                test_res = list()
                confusion_list = dict()
                max_epoch = 2000
                for epoch in tqdm(range(max_epoch), total= max_epoch, desc = time_path + 'Batch %d' % (i+1)):
                    t0 = time.time()

                    logits = net(datag_f.float().to(device))
                    logp = F.log_softmax(logits, 1) # shape = |V| * classes
                    loss = F.nll_loss(logp[train_mask].to(device), datag_label[train_mask].to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    
                    print('=' * 80, file= f)
                    print("epoch {} | loss: {:.5f} | runtime: {:.2f}(s)".format(epoch, loss.item(), time.time() - t0), file= f)
                    x.append(epoch)
                    y.append(loss.item())

                    tp, tn, fp, fn = metrics.confusion_matrix(logits= logp, labels= datag_label, mask= test_mask)
                    print('  Test: ', file= f)
                    confusion_to_file(tp, tn, fp, fn, f)
                    test_res.append([epoch, loss.item(), tp, tn, fp, fn])

                    confusion_list = metrics.confusion_list(logits= logp, labels= datag_label, mask= test_mask)
                
                write_to_pd(test_res, time_path, i+1)
                plotting(x, y, time_path, 'Batch %d' % (i+1))
                with open(time_path + "confusion_list.json", 'w') as json_fp:
                    confusion_dict['Batch %d' % (i+1)] = confusion_list
                    json.dump(confusion_dict, fp= json_fp)
                f.close()
end = time.time()
print("All Time :", end-start)

                