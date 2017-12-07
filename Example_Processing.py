#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:13:18 2017

@author: sgiese
"""
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