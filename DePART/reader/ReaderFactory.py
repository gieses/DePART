# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:46:34 2018

@author: Hanjo
"""
import pandas as pd


def CSV_Reader(infile,seq_column="Sequence", frac_column="Fraction", sep=","):
    """
    Reads a CSV file into the needed format.
    
    Parameters:
    --------------------
    infile: str, location
    seq_column: str, column name of the sequences
    frac_column: str, column name of the retention time
    sep: str, separater in the CSV file
    """
    df_data = pd.read_csv(infile, sep=sep)
    df_data.rename(index=str, columns={seq_column:"Sequence",
                                       frac_column:"Fraction"})
    
    return(df_data)


def MaxQuant_Reader(infile, remove_decoys=True, remove_conts=True,
                    remove_mods=True, verbose=False):
    """
    The MaxQuant evidence file is where the raw data comes from. The
    from evidence function automatically removes contaminants and
    decoys from the evidence file. The input requires the usual columns
    from the evidence file but also a Fraction column. Duplicates 
    are not treated at this point will simply be kept.
    
    Parameters:
    -----------------------
    infile: str,
            path to the evidence file
                
    
    remove_*: bool, remove the respective column in MaxQuant?
    
    remove_mods: bool,
                If True removes all sequences that do not have the tag 
                'Unmodified' (default). 
    """
    df_evidence = pd.read_csv(infile, sep="\t")
    colnames = df_evidence.columns
    
    
    #allow difference MaxQuant evidence input files
    if "Contaminant" in colnames:
        contaminants = "Contaminant"
        reverse = "Reverse" 
    else:
        contaminants = "Potential contaminant"
        reverse = "Reverse"
    
    if verbose:
        print ("Total Peptide IDs: {}".format(df_evidence.shape[0]))
         
    #remove contaminants and decoy hits
    df_evidence = df_evidence[df_evidence[reverse] != "+"]
    df_evidence = df_evidence[df_evidence[contaminants] != "+"]       
    if verbose:
        print ("- reverse/contaminants: {}".format(df_evidence.shape[0]))
    
    #remove modifications
    if remove_mods:
        df_evidence = df_evidence[df_evidence["Modifications"]=="Unmodified"]
        if verbose:
            print ("- modifications: {}".format(df_evidence.shape[0]))
            
    df_evidence = df_evidence.reset_index()
    return(df_evidence) 


def Column_Reader(sequences, fractions):
    """
    Creates a model matrix from a Sequence column.
    
    Parameters:
    -------------------------------
    sequences: ar-like,
                array of sequences (peptides)
                
    """
    df_data = pd.DataFrame([sequences, fractions]).transpose()
    df_data.columns = ["Sequence", "Fractions"]
    return(df_data)
      