#### This repository contains the code and data for the chromatin imaging analyses in "CTCF Mediates Dosage and Sequence-context-dependent Transcriptional Insulation through Formation of Local Chromatin Domains" published in Nature Genetics.

### File manifest

**manuscript_processing_script.ipynb**: Jupyter notebook containing the pipeline for analyzing the chromatin tracing imaging data

**DNA-RNA Colocalization.ipynb**: Jupyter notebook for doing the spot calling in RNA imaging and colocalizing with chromosomes identified from DNA imaging

**Common Tools**: This folder contains python modules that are imported by the Jupyter notebooks

**4CBS_traces.csv, 4CBS-downstream_traces.csv, 4CBS-mutant_traces.csv**: These files contain the processed data (coordinates of all 5kb imaged regions) for the 3 DNA-only chromatin tracing experiments.

**4CBS_traces_rep2_RNA.csv, 4CBS-downstream_traces_rep2_RNA.csv**: These files contain the processed data for the 2 DNA-RNA imaging experiments. Six binary columns indicate whether an RNA spot was detected in the Sox2, GFP, and mCherry channels for the CAST and 129 alleles of the cell. Note that for any given trace, only 3 of the 6 columns are relevant (based on whether the trace is for a 129 or CAST allele). The Sox2 channel targeted both alleles, while the mCherry targeted only 129 and GFP targeted only CAST.
