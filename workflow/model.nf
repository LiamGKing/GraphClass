
process MODEL {
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
from pyfiles import model

data = "$datadir"
out = model.build(data)
print(out, end = '')
	"""
}