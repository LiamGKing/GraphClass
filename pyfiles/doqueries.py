import os
import sys
import pandas as pd
import numpy as np

def query(datadir):
    pths = ""
    for x in range(315, 367):
        pths += '-q graph' + str(x) + '.fasta '
    
    cmd = '/home/liam/BlastFrost/build/BlastFrost -g /home/liam/pytorchtest/added.txt.gfa -f /home/liam/pytorchtest/added.txt.bfg_colors '
    flags = '-o query -t 128 -v -e'
    full = cmd + pths + flags
    os.system(full)
    return datadir