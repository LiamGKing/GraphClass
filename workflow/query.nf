
process QUERY {
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
	from pyfiles import doqueries

	out = doqueries.query(datadir)
	print(out, end = '')
	"""
}