# -*- coding: utf-8 -*-

import numpy as np
from pyteomics import parser
import re
import pandas as pd
from DePART.preprocessing import FeatureFactory as FF
from DePART.reader import ReaderFactory as RF
import sys

def to_minimal_df(df_in):
    """
    Extracts a minimal df partof the given dataframe to be used with the
    implemented machine learning workflow. Also ensures that the indices are 
    compareable
    """
    df1 = df_in.copy().reset_index()
    if "Modified sequence" in df_in.columns:
        df2 = df_in[["Sequence", "Fraction", "Modified sequence"]]
    else:
        df2 = df_in[["Sequence", "Fraction"]]
    return(df1, df2)
    
    
def preprocess_df(infile, reader="MaxQuant", min_obs=300):
    """
    Preprocess a MaxQuant dataframe. Removes duplicates by highest
    intensity, requires at least nobs per Fraction and returns
    a minimal df that is used for the machine learning part.
    
    Parameters:
    ------------------------
    df_evidence: df, Dataframe containing the columns Seuence and Fraction
    """
    
    #read data
    if reader == "MaxQuant":
        df_evidence = RF.MaxQuant_Reader(infile, remove_mods=False)
        
    elif reader =="CSV":
        df_evidence = RF.CSV_Reader(infile)
        
    else:
        sys.exit("Option {} is not supported. Please choose either \
                 MaxQuant or CSV.".format(reader))
    #process training data    
    #select most intense
    df_filter = filter_low_int_ids(df_evidence)
    #minimum obs filter per fraction
    df_filter = filter_obs(df_filter, min_obs=min_obs)    
    #get a minimal df for processing
    df_meta, df = to_minimal_df(df_filter)    
    #generate features
    Generator = FF.FeatureGenerator()
    df_features = Generator.create_all_features(df)
    return(df_meta, df, df_features)
    
    
def mark_most_intense_id(df_evidence):
    """
    Function to filter peptides according to their identified fractions.
    
    Pepties were filtered according to the following criteria.
    1) unique peptides were kept
    2) peptides that were identified in 2 fractions were kept if
        the fractions were adjacent and disregarded if they were not.
        In the first case the peptide with the highest intensity wins and will
        be added to the data.
    3) peptides that were identified in more than 2 fractions were disregarded
    """
    indic_dic = {}
    #clean evidence
    #Peptide ID is the right identifier for the groups of peptides
    indicator = [False] * df_evidence.shape[0]
    
    for ii, grpi in df_evidence.groupby("Peptide ID"):
        nfractions = grpi.shape[0]
        #keep peptide, only identified in one fraction
        if nfractions == 1:
            indicator[grpi.index[0]] = True
            #indic_dic[ii] = "Keep"
        
        #exactly two fractions
        elif nfractions == 2:
            
            #has no intensity? just keep first occurrence
            if grpi.Intensity.dropna().shape[0] == 0:
                keepid = grpi.index[0]
            else:
                #depends on intensity whcih peptide id to take
                nuniquefracs = len(np.unique(grpi.Fraction))
                #case one: same fraction
                if nuniquefracs == 1:
                    keepid = np.argmax(grpi.Intensity)
                #case two: different fraction
                else:
                    #adjacent fraction
                    if np.abs(np.diff(grpi.Fraction.sort_values())[0]) == 1:
                        keepid = np.argmax(grpi.Intensity)
                    #further away
                    else:
                        continue
                        #indic_dic[ii] = "Not Adjacent"
                #store results
                indic_dic[keepid] = "Keep"
                indicator[keepid] = True
                
        #ignore all peptides with ids in more than two fractions
        else:
            #indic_dic[ii] = ">2Fractions"
            pass
    return(indicator)
    
    
def filter_low_int_ids(df_in):
    """
    Description.
    
    Requires the input format to have the columns:
        Sequence, Fraction, Intensity, Peptide ID
    """
    req_cols = set(["Sequence", "Fraction", "Intensity", "Peptide ID"])
    if len(set(df_in.columns) & req_cols)==4:
        #filter only peptides that occur in 1 or 2 fractions.
        #for two fractions take the fraction with the highest intensity
        df_in["Fraction.Filter"] = mark_most_intense_id(df_in)
        return(df_in[df_in["Fraction.Filter"]==True])
    else:
        print ("Your data frame is missing one of the required columns:\
               Sequence, Fraction, Intensity, Peptide ID")
        return(None)


def filter_obs(df_in, min_obs=300):
    """
    Filters the data based on the minimum number of observations per fraction.
    
    Parameter:
    .-------------------------
    """
    df_model = df_in.copy()
    min_frac = df_model.Fraction.min()
    max_frac = df_model.Fraction.max() + 1
    print ("NRows BEFORE min obs filter: {}".format(df_model.shape[0]))
    obs = {i:j for i,j in zip(np.arange(min_frac, max_frac), 
                              np.bincount(df_model.Fraction)[1:])}
    df_model["nobs"] = [obs[i] for i in df_model.Fraction]
    df_model = df_model[df_model["nobs"] >=min_obs]
    print ("NRows AFTER min obs filter: {}".format(df_model.shape[0]))
    return(df_model)
    
def remove_brackets(seq):
    """
    Removes all brackets (and underscores...) from protein sequences.
    """
    return(re.sub("[\(\)\[\]_]", "", seq))
   
#%%
def extract_modifications(sequences, verbose=False):
    """
    Performs simple checks on the sequences. Assumes that the
    sequences has the "Amod" format, where A is an amino acid and
    mod a lower case modification. All brackets are removed.
    Mods need to be strictly lower case.
    
    sequences = ["ELVIS", "PEPT(phos)ID(bs3)E", 
                 "ELVISCcmPEPTIbs3DE", "PEPT[unimod:21]IDE"]
    """
    
    sequences = sequences.apply(FF.simply_alphabet)
    #init data structures
    patterns = []
    npatterns = 0
    all_patterns = {}
    detected_mods = []
    
    #some formatting    
    AA_set = set(parser.std_amino_acids)
    test_val = np.zeros(len(sequences))
    sequences = np.array(sequences)
    
    #matches all nterminal mods, e.g. glD or acA
    nterm_pattern = re.compile(r'\b([a-z]+)([A-Z])')
    
    #test each sequence for non-AA letters
    for ii, seqi in enumerate(sequences):
        nterm_match = re.findall(nterm_pattern, seqi)
        #nterminal acetylation
        if len(nterm_match) == 0:
            pass
        else:
            all_patterns[nterm_match[0][1]] = nterm_match[0][0]
            seqi = seqi.replace(nterm_match[0][0], "")
        test_val[ii] = np.sum([1 for aai in seqi if aai not in AA_set])
    
    #keep the problematic ones
    prob_sequences = sequences[np.where(test_val>0)]
    #replace all kind of brackets
    prob_sequences = pd.Series([remove_brackets(seqi) for seqi in prob_sequences])

    for probi in prob_sequences:      
        #other mods
        pattern = re.compile("([A-Z])([^A-Z]+)")
        matches = re.findall(pattern, probi)
        if verbose:
            print (probi)
            print (matches)
            
        aa, mod = zip(*matches)
        patterns.append(mod)
        npatterns += len(mod)
        for aai,modi in zip(aa, mod):
            all_patterns[aai] = modi
            
        if len(mod) == 0:
            detected_mods.append(None)
        else:
            detected_mods.append(mod)
    
    if npatterns > 0:
        print ("We identified the following modifications: {}".format(all_patterns.items()))
    else:
        print ("There were no identifyable amino acids/modifications in your data: {}. We cannot deal with these...".format(",".join(prob_sequences.head().values)))   
    return(all_patterns, detected_mods)
   
def replace_numbers(seq):
    rep = {"1":"one",
          "2":"two",
          "3":"three",
          "4":"four",
          "5":"five",
          "6":"six",
          "7":"seven",
          "8":"eight",
          "9":"nine",
          "0":"zero"}
    pattern = re.compile("|".join(rep.keys()))
    return(pattern.sub(lambda m: rep[re.escape(m.group(0))], seq))
    
    
def rewrite_modsequences(seq):
    """
    Rewrites modified sequences to modX format. requires the
    input to be preprocessed such that no brackets are in the sequences.
    
    Meant to be used via apply.
    seq = "ELVIS"
    seq = "ELVISCcmASD"
    """
    #TODO, replace numbers with text!  
    return(re.sub("([A-Z])([^A-Z]+)", r'\2\1', seq))
    
    
def replace_nterm_mod(seq):
    """
    Removes the nterminal modification.
    """
    return(re.sub(r'\b([a-z]+)([A-Z])', r'\2', seq))
    
    