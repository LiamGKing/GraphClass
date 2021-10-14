#!/usr/bin/env nextflow
params.k = 5
params.datadir = "$baseDir"
params.download = false 
params.dataready = true

nextflow.enable.dsl = 2

include { METADATA } from './workflow/metadata'
include { DOWNLOAD } from './workflow/download'
include { MAKEGRAPHS } from './workflow/makegraphs'
include { MAKEFASTA } from './workflow/makefasta'
include { QUERY } from './workflow/query'
include { DATASET } from './workflow/dataset'
include { MODEL } from './workflow/model'


workflow {
    if(params.dataready == true) {
        MODEL(params.datadir)
    } else {
        	if(params.download == true) {
        		DOWNLOAD(params.datadir, params.genera)
        		METADATA(params.k, DOWNLOAD.out)
        	} else {
        		METADATA(params.k, params.datadir)
        	}
        	MAKEGRAPHS(METADATA.out)
        	MAKEFASTA(MAKEGRAPHS.out)
        	QUERY(MAKEFASTA.out)
        	DATASET(QUERY.out)
        	MODEL(DATASET.out)
	}
}

workflow.onComplete {
	log.info ( workflow.success ? "\nDone!" : "Oops .. something went wrong" )
}