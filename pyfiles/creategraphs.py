import os
import pandas as pd

def create(datadir):
    colnames = ['id', 'assembly', 'genus', 'species', 'seqfile', 'cntfile', 'meta']
    readcsv = pd.read_csv('/home/liam/100test/processed_data/metadata.csv', names=colnames)
    paths = readcsv.seqfile.tolist()
    paths = paths[1001:]
    species = readcsv.species.tolist()
    species = species[1001:]
    
    ind = 0
    for s in paths:
        cmd = "/home/liam/bifrost/bifrost/build/src/Bifrost build -r "
        pth = '/home/liam/100test' + s
        pst = ' -o /home/liam/pytorchtest/graphs/graph' + str(ind) + ' -t 64 -c -v'
        full = cmd + pth + pst
        os.system(full)
        ind += 1
    
    return datadir

