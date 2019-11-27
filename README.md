# dNLILP
Written by Ali Payani

Based on paper : https://arxiv.org/pdf/1906.03523.pdf


The software is tested using python 3.6 and Tensorflow 1.13
Each experiment is a single Python file located in the root folder and all the experiments use the common library for dNL-ILP located in Lib folder.

List of current experiments:



1)  Classification tasks:
	the only flag needed to be specified in the fold number in the crossvalidation scheme (0-4) for everything except for Mutagenesis which is 0-9
	
	example: 
		>python3 filename TEST_SET_INDEX=0 
		where filename can be any of {classify_mutagenesis.py, classifiy_cora.py, classifiy_uwcse.py, classifiy_imdb.py, classify_movielens.py}
		

2) other experiments can be envoked simply by running the corresponding py file


	



List of packages required for running the program:
numpy, pandas, sklearn, tensorflow
