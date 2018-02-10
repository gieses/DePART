# DeepRThSAX

DeepRThSAX is a high-level python package for retention time prediction in hydrophilic strong anion exchange chromatography (hSAX). DeepRThSAX uses several packages for the pre-processing of (MaxQuant) peptide tables and the final machine learning prediction (custom format peptide tables are also supported).There are convenience functions implemented to use the same preprocessing as used in the manuscript (*in preparation*) within a single function call.
The core function of DeepRThSAX is the training & prediction of the 'elution time' peptides from hSAX fractionation. Elution time is in our workflow exchangeable with the fraction from an off-line fractionation experiment.


Getting Started
---

DeepRThSAX requires the installation of the following packages and ideally python 3.5.

Dependencies
---
* numpy
* scipy
* matplotlib
* seaborn
* pyteomics
* sklearn
* pandas
* keras
* joblib
* Biopython

 Installation
 ----
 At the moment the code is not wrapped into a python package. However,
 useage is straight forward. Here is a complete working example based on an
 MaxQuant evidence file. The training can also be done with arbitrary CSV
 files as long as the "Sequence" and "Fraction" column are there. For this
 set the 'from_CSV' option to True.
 
```
import RTlib as RT
import RTMLlib as ML

#this is an input file, e.g. from an maxquant evidence file
infile = "sample_data/evidence_SAX_Trost.txt"

#specify the outpath here
outpath = "sample_data/"

#this is used as prefix for the results
name = "Test"

#process the data and prepare the given evidence /csv file
# sequence -> feature matrix
matrix, all_data, train_df, valid_df = \
                RT.preprocess_manuscript(infile, outpath, name, n_test=-1, 
                                         mods=False, target="Fraction", 
                                         correct=True, scale=False, 
                                         from_CSV=False, min_obs=300)
                
#use the training dataframe for CV
CV_results = ML.cross_validation(train_df, valid_df, nkfold=5, n_jobs=5)

#train the classifier on the complete data
val_results, nnmodel = ML.train_validation(train_df, valid_df, epochs=100, 
                                           batch_size=512, plot=True)
```
 Notes
 ----
 Changing the architecture and the parameters of neural network can be done
 by editing the RTMLlib. A few parameters can be changed by the function call,
 e.g. the activation function (swish is also supported).
 

Authors
----
* Sven Giese

License
----
This project is licensed under the MIT License - see the LICENSE.md file for details