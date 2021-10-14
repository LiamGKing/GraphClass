import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def create(datadir):
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
    for graph in range(290):
        print("Graph#%i Start" % (graph))
        graphname = '/home/liam/pytorchtest/graphs/graph' + str(graph) + '.gfa'
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
        graphname = '/home/liam/pytorchtest/graphs/query_graph' + str(graph) + '.fasta_subgraph.fasta'
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
    
    torch.save(donegraphs, '/home/liam/pytorchtest/graphdataset.pkl')
    return datadir

