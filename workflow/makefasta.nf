
process MAKEFASTA {
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
	from pyfiles import makefasta

	out = makefasta.create(datadir)
	print(out, end = '')
	"""
}