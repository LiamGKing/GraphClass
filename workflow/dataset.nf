
process DATASET {
    echo true
	input:
	  val(datadir)

	output:
	  stdout emit: filename

	script:
	"""
#!python3
import sys
sys.path.append("$baseDir")
from pyfiles import graphdataset
data = "$datadir"
out = graphdataset.create(data)
print(out, end = '')
	"""
}