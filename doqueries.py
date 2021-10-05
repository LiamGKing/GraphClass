import os
import sys
import pandas as pd
import numpy as np

for x in range(367):
    pths = ""
    pths += '-q graph' + str(x) + '.fasta '

cmd = '/home/liam/BlastFrost/build/BlastFrost -g /home/liam/pytorchtest/added.txt.gfa -f /home/liam/pytorchtest/added.txt.bfg_colors '
flags = '-o nosubgraph -t 128 -v -e'
full = cmd + pths + flags
os.system(full)