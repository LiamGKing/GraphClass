import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, SAGPooling, global_mean_pool, JumpingKnowledge)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class SAGPool(torch.nn.Module):
    def __init__(self, num_layers, hidden, ratio=0.8):
        super(SAGPool, self).__init__()
        self.conv1 = GraphConv(1001, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [SAGPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 31)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    
colnames = ['id', 'assembly', 'genus', 'species', 'seqfile', 'cntfile', 'meta']
samples = pd.read_csv('/home/liam/100test/processed_data/metadata.csv', names=colnames)
species = samples.species.tolist()
species = species[1001:]
specdict = {}
indspec = {}
ind = 0
for s in set(species):
    specdict[s] = ind
    indspec[ind] = s
    ind += 1
colors = []
with open('/home/liam/pytorchtest/added.txt.gfa_current_colors', 'r') as f:
    for l in f:
        temp = str.rsplit(l)[0]
        temp = str.split(temp, '.fna')[0]
        colors.append(temp)

donegraphs = []
for graph in range(188):
    print("Graph#%i Start" % (graph))
    graphname = 'graph' + str(graph) + '.gfa'
    with open(graphname, 'r') as f:
        numnodes = 0
        fromarr = []
        toarr = []
        for l in f:
            temp = str.split(l)
            if temp[0] == 'S':
                numnodes = int(temp[1])
            elif temp[0] == 'L':
                fromarr.append(int(temp[1]) - 1)
                toarr.append(int(temp[3]) - 1)
    
    g = np.zeros((numnodes,1001),dtype=np.float32)
    graphname = 'query_graph' + str(graph) + '.fasta_subgraph.fasta'
    print("Graph#%i Colors" % (graph))
    with open(graphname, 'r') as f:
        ind = 0
        for l in f:
            if ind % 2 == 0:
                temp = str.split(l,'|')
                num = int(str.split(temp[0],'>')[1])
                col = str.rsplit(temp[1])[0]
                col = str.split(col, '.fna')[0]
                g[num][colors.index(col)] = 1
            ind += 1
    print("Graph#%i Data Create" % (graph))
    fromarr = np.array(fromarr)
    toarr = np.array(toarr)
    edge_index = np.zeros((2,len(toarr)), dtype=np.long)
    edge_index[0] = fromarr
    edge_index[1] = toarr
    y = np.zeros(1,dtype=np.float32)
    y[0] = specdict[species[graph]]
    y = torch.tensor(y, dtype=torch.long)
    x = torch.tensor(g, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    donegraphs.append(data)
    
model = SAGPool(5, 256)
random.shuffle(donegraphs)
n = (len(donegraphs)//10) * 3
train_data = donegraphs[n:]
test_data = donegraphs[:n]
train_loader = DataLoader(train_data, batch_size=1)
test_loader = DataLoader(test_data, batch_size=1)
device = torch.device('cuda')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(40):
    model.train()
    loss_all = 0
    for d in train_loader:
        d = d.to(device)
        optimizer.zero_grad()
        output = model(d)
        loss = F.nll_loss(output, d.y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    print(loss_all)
    
model.eval()
correct = 0
for d in test_loader:
    d = d.to(device)
    pred = model(d).max(dim=1)[1]
    print(indspec[pred.item()], " vs. ", indspec[d.y[0].item()])
    correct += pred.eq(d.y).sum().item()

print("Accuracy: %f" % (correct/len(test_loader.dataset)))
print((correct/len(test_loader.dataset)))

