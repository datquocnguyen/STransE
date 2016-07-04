# STransE: a novel embedding model of entities and relationships in knowledge bases

This STransE program provides the implementation of the embedding model STransE for knowledge base completion, as described in my NAACL-HLT 2016 paper:

 Dat Quoc Nguyen, Kairit Sirts, Lizhen Qu and Mark Johnson. 2016. [STransE: a novel embedding model of entities and relationships in knowledge bases](http://www.aclweb.org/anthology/N16-1054). In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, NAACL-HLT 2016, pp. 460-466. [[.bib]](http://www.aclweb.org/anthology/N16-1054.bib)

The program also provides the implementation of the embedding model [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data).

Please cite my NAACL-HLT 2016 paper whenever STransE is used to produce published results or incorporated into other software.

I would highly appreciate to have your bug reports, comments and suggestions about STransE. As a free open-source implementation, STransE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

##Usage

###Compile the program


Suppose that `g++` is already set to run in command line or terminal. After you clone or download (and then unzip) the program, you have to compile the program by executing:

	SOURCE_DIR$ g++ -I ../SOURCE_DIR/ STransE.cpp -o STransE -O2 -fopenmp -lpthread

Note that the actual command starts from `g++`. Here `SOURCE_DIR` is simply used to denote the source code directory. Examples:

	STransE$ g++ -I ../STransE/ STransE.cpp -o STransE -O2 -fopenmp -lpthread

	STransE-master$ g++ -I ../STransE-master/ STransE.cpp -o STransE -O2 -fopenmp -lpthread

###Run the program

To run the program, we perform:


	$./STransE -model 1_OR_0 -data CORPUS_DIR_PATH -size <int> -l1 1_OR_0 -margin <double> -lrate <double> [-init 1_OR_0] [-nepoch <int>] [-evalStep <int>] [-nthreads <int>]

	//For Windows OS: use ./STransE.exe instead of ./STransE

where hyper-parameters in [ ] are optional!

**Required parameters:** 

`-model`: Specify the embedding model STransE or TransE. It gets value 1 or 0, where 1 denotes STransE while 0 denotes TransE.

`-data`: Specify path to the dataset directory. Find the dataset format instructions in the `Datasets` folder inside the source code directory. 

`-size`: Specify the number of vector dimensions.

`-l1`:  Specify the `L1` or `L2` norm. It gets value 1 or 0, where 1 denotes `L1`-norm while 0 denotes `L2`-norm.

`-margin`: Specify the margin hyper-parameter.

`-lrate`: Specify the SGD learning rate.

**Optional parameters:** 

`-init`: Use when `-model` gets value 1 (i.e. for STransE). It gets value 1 or 0 in which the default value is 1. The value 1 means that the entity and relation vectors are initialized from external files (e.g. `entity2vec.init` and `relation2vec.init` in the `Datasets` folder inside the source code directory), while the value 0 means that the entity and relation vectors are randomly initialized.

`-nepoch`: Specify the number of training epochs. The default value is 2000.

`-evalStep`: Specify a step to save and evaluate the model, e.g., evaluating the model after each step of 500 training epochs. The default value is 2000.

`-nthreads`: Specify the number of CPU cores used for evaluation. The default value is 1. Note that the evaluation process (i.e. evaluating link/entity prediction in knowledge bases) is slow. If you can afford to run the program with many CPU cores, the evaluation process will be much faster, so you can even evaluate the model after each training epoch. 

### Evaluation scores

For evaluating link/entity prediction, the program provides ranking-based metrics as evaluation scores, including the mean rank, the mean reciprocal rank, Hits@1, Hits@5 and Hits@10 in two setting protocols "Raw" and "Filtered". 

### Reproduce the STransE results 

To reproduce the STransE results published in my NAACL-HLT 2016 paper, execute:

	$ ./STransE -model 1 -data Datasets/WN18/ -size 50 -margin 5 -l1 1 -lrate 0.0005

	$ ./STransE -model 1 -data Datasets/FB15k/ -size 100 -margin 1 -l1 1 -lrate 0.0001

