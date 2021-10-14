import os
import sys
import pandas as pd
import numpy as np

def create(datadir):
    for x in range(368):
        openpth = '/home/liam/pytorchtest/graphs/graph' + str(x) + '.gfa'
        writepth = '/home/liam/pytorchtest/graphs/graph' + str(x) + '.fasta'
        with open(openpth, 'r') as f:
            print("File#%i" % (x))
            ind = 0
            with open(writepth, 'w') as w:
                for l in f:
                    temp = str.rsplit(l)
                    if temp[0] == 'S':
                        seq = '>' + str(ind)
                        w.write(seq + '\n')
                        w.write(temp[2] + '\n')
                        ind += 1
    return datadir