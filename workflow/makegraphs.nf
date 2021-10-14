
process MAKEGRAPHS {
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
	from pyfiles import creategraphs

	out = creategraphs.create(datadir)
	print(out, end = '')
	"""
}