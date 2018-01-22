#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:11:34 2017

@author: sgiese
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures
import itertools
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
from pyteomics import parser



class ModelMatrix():
    """
    Model Matrix for the prediction of the hSAX Retention Time.
    """
    def __init__(self, name, loc=""):
        """
        Initializes a Model Matrix which can be imported from a
        MaxQuant Evidence File or an arbitrary CSV file with a "Sequence"
        column which contains the peptide sequences.
        
        Parameter:
        --------------------
        loc: str,
            fiel location of the input data.
        
        name: str,
                A name for the input data anaylsis set-up. This name
                will be used as prefix for output tables as identifier.
        """
        self.loc = loc
        self.modelmatrix = None
        self.X = None
        self.y = None
        self.seq = None
        self.options = None
        self.identifier = name
        self.features = []
    
    def set_Xy(self):
        """
        Sets the X and y attributes which can directly be used as input for
        any sklearn classifier.
        """
        if self.modelmatrix:
            self.X = self.modelmatrix[self.features]
            self.y = self.modelmatrix["Fraction"]
            self.seq = self.modelmatrix["Sequence"]
        else:
            print ("Couldn't set X and y since the model matrix was not set.")
            
    def from_column(self, sequences, dummy_fraction=True):
        """
        Creates a model matrix from a Sequence column.
        
        Parameters:
        -------------------------------
        sequences: ar-like,
                    array of sequences (peptides)
                    
        """
        df_data = pd.DataFrame(sequences)
        df_data.columns = ["Sequence"]
        
        if dummy_fraction:
            df_data["Fraction"] = 42
        
        df_data = df_data[["Sequence", "Fraction"]]
        df_data["Modifications"] = "Unmodified"
        df_data["Modified sequence"] = df_data["Sequence"]
        df_data["PyteomicsSequence"] = df_data["Sequence"]
        df_data["Sequence"] = [i.replace("U","C") for i in df_data["Sequence"]]
        self.modelmatrix = df_data
        
    def from_csv(self, dummy_fraction=True, index_col=None):
        """
        Reads an arbitrary CSV file. The only requirement is a
        'Sequence' column. All other necessary columns will be generated
        """
        
        if index_col:
            df_data = pd.read_csv(self.loc, index_col=index_col)
        else:
            df_data = pd.read_csv(self.loc)
        if dummy_fraction:
            df_data["Fraction"] = 42
        
        df_data = df_data[["Sequence", "Fraction"]]
        df_data["Modifications"] = "Unmodified"
        df_data["Modified sequence"] = df_data["Sequence"]
        df_data["PyteomicsSequence"] = df_data["Sequence"]
        df_data["Sequence"] = [i.replace("U","C") for i in df_data["Sequence"]]
        #filter data for non standard peptides
        df_data = df_data[df_data["Sequence"].apply(marknonstandardaa)]
        self.modelmatrix = df_data
        

    def from_evidence(self, remove_mods=True):
        """
        The MaxQuant evidence file is where the raw data comes from. The
        from evidence function automatically removes contaminants and
        decoys from the evidence file. The input requires the usual columns
        from the evidence file but also a Fraction column. Duplicates 
        are treated such that only the most intense identification from 
        adjacent fractions is kept. Peptides that were identified in 2 (not
        adjacent) ore more fractions are disgarded.
        
        Parameters:
        -----------------------
        infile: str,
                path to the evidence file
                    
        
        has_fraction: bool,
                      if one the input file is assumed to have a fraction column.
                      if not the fraction is take from the specific file fromat
            
        remove_mods: bool,
                    If True removes all sequences that do not have the tag 
                    'Unmodified' (default). 
        """
        summary = []
        df_evidence = pd.read_csv(self.loc, sep="\t")
        colnames = df_evidence.columns
        summary.append("Evidence Input: {}".format(df_evidence.shape))
        
        #allow difference MaxQuant evidence input files
        if "Contaminant" in colnames:
            contaminants = "Contaminant"
            reverse = "Reverse" 
        else:
            contaminants = "Potential contaminant"
            reverse = "Reverse"
                 
        #remove contaminants and decoy hits
        df_evidence = df_evidence[df_evidence[reverse] != "+"]
        df_evidence = df_evidence[df_evidence[contaminants] != "+"]       
        summary.append("- reverse/contaminants: {}".format(df_evidence.shape[0]))
        
        #remove modifications
        if remove_mods:
            df_evidence = df_evidence[df_evidence["Modifications"]=="Unmodified"]
            summary.append("- modifications: {}".format(df_evidence.shape[0]))
    
        df_evidence = df_evidence.reset_index()
        #filter only peptides that occur in 1 or 2 fractions.
        #for two fractions take the fraction with the highest intensity
        df_evidence["Fraction.Filter"] = filter_peptides_fraction(df_evidence)
        df_evidence_maxInt = df_evidence[df_evidence["Fraction.Filter"]==True]
                
        #selection of columns to keep
        xcolumns = ["Sequence", "Length", "Modifications", "Modified sequence",
                    "Fraction", "Retention time", "Charge", "Mass", 
                    "Intensity", "Peptide ID", "Score"]
            
        df_evidence_maxInt = df_evidence_maxInt[xcolumns]
        df_evidence_maxInt = df_evidence_maxInt.sort_values(by="Fraction")
        summary.append("- intensity/adjacent filter: {}\
                       ".format(df_evidence_maxInt.shape[0]))
        
        df_evidence_maxInt["Sequence"] = [i.replace("U","C") for i \
                          in df_evidence_maxInt["Sequence"]]
        
        self.modelmatrix = df_evidence_maxInt
        self.summary = "\n".join(summary)      
        
    def add_features(self, verbose=1):
        """
        This functions adds the pyteomics sequence, loglength, cterm, nterm and 
        noxidation feature to the dataframe. The columns will be added
        in place of the model matrix
        
        Parameters:
        -------------------------
        df: dataframe,
             df from a evidence file
        
        verbose: int
                 Either 1 or 0. With 1 processing info is printed to std.
                 
        Returns:
        -------------------------
        df: processed dataframe
        """
        if verbose:
            print ("Adding pyteomics, loglength, cterm, nterm, noxidation and netcharge features.")
        
        #mutable objects will be changed during processing
        df = self.modelmatrix
        df["PyteomicsSequence"] = df["Modified sequence"].apply(prepare_sequence_for_pyteomics)
        df["loglength"] = [np.log(len(i)) for i in df["Sequence"]]
        df["cterm"] = [1.*add_shortest_distance(i, opt="cterm", verbose=False)/ len(i) for i in df["Sequence"]]
        df["nterm"] = [1.*add_shortest_distance(i, opt="nterm", verbose=False)/ len(i) for i in df["Sequence"]]
        df["noxidation"] = [int(extract_mods_n(i, "Oxidation")) for i in df["Modifications"]]
        df["netcharge"] = [get_net_charge(i) for i in df["Sequence"]]

        #store features
        self.features.extend(["loglength", "cterm","nterm", "netcharge"])
    
    def add_AA_count(self, seq_column="Sequence", correct=False, lcp=-0.20, 
                     mods=1):
        """
        Counts the amino acid in a peptide sequence. Counting uses the
        pyteomics amino_acid composition. Modified residues of the pattern
        "modA" are already supported.
    
        If the modifications should not be considered another sequence column 
        can be used. As read on the pyteomics doc an "lcp" factor can substantially 
        increase the prediction accuracy.
        
        Parameters:
        -----------------------------------
        df: df,
            dataframe with sequences
        seq_column: string,
                    sequence column that is used to generate the features
                    
        correct: bool,
                  if true, the counts are corrected by the lcp (based on pyteomics
                  model idea)
        lcp: float,
             factor used for amino acid count correction
        
        mods: bool,
              1 (default) or zero. If one: oxM and M area treated as different
              entities.
              
        Examples:
        -----------------------------------
        #modification and termini supporting
        >>mystr = "nAAAAAAAAAAAAAAAGAAGcK"
        
        #just aa composition
        >>mystr = "AAAAAAAAAAAAAAAGAAGK"
        
        Returns:
        --------------------------------------
        df: dataframe with amino acid count columns
        """
        df = self.modelmatrix
        #create dataframe with counts
        if mods:
            aa_counts = [parser.amino_acid_composition(i) for i in df[seq_column]]
            
        else:
            aa_counts = [parser.amino_acid_composition(i) for i in df["Sequence"]]
            
        #to dataframe
        aa_count_df = pd.DataFrame(aa_counts)
        aa_count_df = aa_count_df.replace(np.nan, 0)
        
        #if correction, then adjust the raw counts
        if correct:
            cfactor = 1. + lcp * df["loglength"]
            for i in aa_count_df.iterrows():
                aa_count_df.iloc[i[0]] = i[1].values * cfactor.iloc[i[0]]
         
        #if modifications, then we need the true sequence here.
        if mods:
            aa_count_df["PyteomicsSequence"] = df["PyteomicsSequence"].copy().values
        
        aa_count_df = aa_count_df.set_index(df.index)
        pd.DataFrame.merge(df, aa_count_df, how="inner")
        
        self.features.extend([i for i in aa_count_df.columns if i != "PyteomicsSequence"])
        self.modelmatrix = pd.concat([df, aa_count_df], axis=1)
        
        
    def add_positional_effect(self, ntermini=5, residues="all"):
        """
        Adds indicator variables for residues in specific positions from
        and to the c/n-term.  C-terminal 1 is always ignored.
        
        Parameters:
        ---------
        ntermini: int,
                    number of positions away from the termini to consider.
                   
        residues: str,
                 sring of amino acids to consider
                 
        """
        print ("Adding positional feature counts for the amino acids.")
        
        df = self.modelmatrix
        if residues=="all":
            residues = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", 
                        "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        else:
            residues = list(residues)
        residues_hash = {i:0 for i in residues}
        
        #-1 one since last c-term not sued
        nfeatures = (2*ntermini-1) * len(residues)
        #init dic with counts
        #ini dataframe with same row index as df, to overwrite counts
        count_dic = {j+res+str(i):0 for res in residues for i in range(0, ntermini) for j in ["N"]}
        count_dic.update({j+res+str(i):0 for res in residues for i in range(1, ntermini) for j in ["C"]})
        
        count_df = pd.DataFrame(np.zeros((df.shape[0], nfeatures)))
        count_df.columns = sorted(count_dic.keys())
        count_df.index = df.index
        
        #super inefficient
        # todo: fixme
        for ii, rowi in df.iterrows():
            #if the peptides are shorter than 2x ntermini, the
            #counts would overlap. TO avoid this shorten the termini
            #counts when neceessary
            seq = rowi["Sequence"]
            n = len(seq)
            if (n - 2*ntermini) < 0:
                tmp_ntermini = np.floor(n/2.)
            else:
                tmp_ntermini = ntermini
                
            #iterate over number of termini, add count if desired (residues)
            for i in range(0, int(tmp_ntermini)):
                if seq[i] in residues_hash:
                    nterm = "N"+seq[i]+str(i)
                    count_df.set_value(ii, nterm, count_df.loc[ii][nterm]+1)
                    
                if seq[-i-1] in residues_hash:
                    cterm = "C"+seq[-i-1]+str(i)
                    #sinec the last amino acid is usually K/R don't add unnecessary
                    #features here
                    if i != 0:
                        count_df.set_value(ii, cterm, count_df.loc[ii][cterm]+1)
                        
        #correct other counts
        #by substracting the sequence specific counts
        new_df = df.join(count_df)
        new_df = new_df.drop("PyteomicsSequence", axis=1)
        #iterate over columns
        for res in residues:
            tmp_df = new_df.filter(regex="(N|C){}\d".format(res))
            sums = tmp_df.sum(axis=1)
            #correct the internal counts
            new_df[res] = new_df[res] - sums
        
        self.modelmatrix = new_df.copy()
        self.features.extend(count_df.columns)
        
        
    def add_cterm_indicator(self, residue="K", verbose=1):
        """
        Adds either 0 or 1 for the cterminal residue if Lysine or Arginine.
        
        Parameters:
        -------------------------
        residue: char,
                For Trypsin either K/R. Other amino acids are probably
                underpresented.
        """
        if verbose:
            print ("Adding c-term residue indicator for {}".format(residue))
        df = self.modelmatrix
        df["Cterm"+residue] = [1 if i[-1] == residue else 0 for i in \
                              df["Sequence"]]
        self.features.append("Cterm"+residue)
        
        
    def use_correction(self, lcp=-0.2):
        """
        Given a dataframe with amino acid counts the counts will be adjusted
        according to the length correction parameter model (lcp).
        
        Parameters:
        -----------------------------
        minimal_df: dataframe with counts of AA
        
        lcp: float,
            correction value (default: -0.2)
            
        Returns:
        -----------------------------
        dataframe, with corrected counts
        """
        df = self.modelmatrix
        aa = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", 
              "F", "P", "S", "T", "W", "Y", "V"]
        na = "|".join(["(N|C){}\d".format(i) for i in aa]) +"|"
        na += "|".join(["^{}$".format(i) for i in aa])
        
        minimal_df = df.filter(regex=na)
        
        try:
            del minimal_df["Sequence"]
        except:
            pass
        
        loglength = np.log(minimal_df.sum(axis=1))
        cfactor = 1. + lcp * loglength
        for i in minimal_df.iterrows():
            minimal_df.loc[i[0]] = i[1].values * cfactor.loc[i[0]]
        
        try:
            self.modelmatrix = pd.concat([df[["loglength", "netcharge", "nterm", 
                                              "cterm", "Sequence", "Fraction", "Score"]],
                                          minimal_df], axis=1)
        except KeyError:
            self.modelmatrix = pd.concat([df[["loglength", "netcharge", "nterm", 
                                              "cterm", "Sequence", "Fraction"]],
                                                      minimal_df], axis=1)            
            
    def add_patches(self, aromatic=True, acidic=True, basic=True, mixed=True, 
                    counts_only=False, verbose=1):
        """
        Adds counts for patches of amino acids. A pattern is loosely
        defined as string of amino acids of a specific class, e.g. aromatic 
        (FYW). The pattern is only counted if at least two consecutive 
        residues occur: XXXXFFXXXX would be a pattern but also XFFFXXX.
        
        
        Parameters:
        -------------------------
        aromatic, acidic, basic, mixed: bool,
            if True, the respective amino acids are located in the sequence
            and checked for 'patterns'.
            
        counts_only: bool,
                    if True DE and ED are added in a single column "acidic_patterns",
                    if False, DE, ED are counts are added in separate columns.
                    Same is true for the combinations of KRH and WYF.
      
    
        """
        if verbose:
            print ("Adding pattern features...")
            
        df = self.modelmatrix
        combs = []
        if acidic:
            #DE
            ac_aa = ["D", "E"]
            ac_combs = ["".join(i) for i in itertools.combinations_with_replacement(ac_aa, 2)]
            combs.extend(ac_combs)
            
            if counts_only:
                #df["acidic_patterns"] = [count_matches(ac_combs, i) for i in df["Sequence"].values]
                pattern = re.compile("[" + "|".join(ac_aa) +"]{2,}")
                df["acidic_patterns"] = [len(re.findall(pattern, i)) for i in df["Sequence"].values]
                self.features.append('acidic_patterns')
                
        if basic:
            #KRH
            ba_aa = ["K", "R"] #H
            ba_combs = ["".join(i) for i in itertools.combinations_with_replacement(ba_aa, 2)]
            combs.extend(ba_combs)
            if counts_only:
                #df["basic_patterns"] = [count_matches(ba_combs, i) for i in df["Sequence"].values]
                pattern = re.compile("[" + "|".join(ba_aa) +"]{2,}")
                df["basic_patterns"] = [len(re.findall(pattern, i)) for i in df["Sequence"].values]
                self.features.append('basic_patterns')
            
        if aromatic:
            #WFY
            ar_aa = ["W", "Y", "F"]
            ar_combs = ["".join(i) for i in itertools.combinations_with_replacement(ar_aa, 2)]
            combs.extend(ar_combs)
            if counts_only:
                #df["aromatic_patterns"] = [count_matches(ar_combs, i) for i in df["Sequence"].values]
                pattern = re.compile("[" + "|".join(ar_aa) +"]{2,}")
                df["aromatic_patterns"] = [len(re.findall(pattern, i)) for i in df["Sequence"].values]
                self.features.append('aromatic_patterns')
                        
        if mixed:
            #DE / KR
            mx_combs = ["".join(i) for i in list(itertools.product(ac_aa, ba_aa))]
            mx_combs = mx_combs + ["".join(reversed(i)) for i in list(itertools.product(ac_aa, ba_aa))]
            combs.extend(mx_combs)
            if counts_only:
                #df["mixed_patterns"] = [count_matches(mx_combs, i) for i in df["Sequence"].values]
                pattern = re.compile("[" + "|".join(ac_aa+ba_aa) +"]{2,}")
                df["mixed_patterns"] = [len(re.findall(pattern, i)) for i in df["Sequence"].values]            
                self.features.append('mixed_patterns')
                
        if not counts_only:#count the patterns
            for pattern in combs:
                df[pattern] = [str(i).count(pattern) for i in df["Sequence"].values]
                self.features.append(pattern)
        
        
    def add_peptide_properties(self, verbose=1):
        """
        Adds peptide properties to the dataframe as features.
        Amino acids in helix: V, I, Y, F, W, L. 
        Amino acids in Turn: N, P, G, S. 
        Amino acids in sheet: E, M, A, L.
        aromaticity, gravy, pi is also added
        
        """
        df_data = self.modelmatrix
        if verbose:
            print ("Adding biopython features: aromaticity, gravy, helix, turn, sheets and pi")
        prots_biop = [ProteinAnalysis(seqi) for seqi in df_data["Sequence"]]
        

        #percentage helix, turn, sheets by amino acid count
        percentages = [i.secondary_structure_fraction() for i in prots_biop]
        helix, turn, sheets = zip(*percentages)
        df_data["helix"] = helix
        df_data["turn"] = turn
        df_data["sheets"] = sheets

        #non-structure features        
        df_data["aromaticity"] = [i.aromaticity() for i in prots_biop]
        df_data["gravy"] = [i.gravy() for i in prots_biop]
        df_data["pi"] = [i.isoelectric_point() for i in prots_biop]
        df_data.fillna(0.0)
        self.features.extend(["helix", "turn", "sheets", "aromaticity", 
                              "gravy", "pi"])


    def add_turn_indicator(self, verbose=1):
        """
        Computes the average number of amino acids between Proline residues
        
        Example:
        -----------------------
        myseq = "ASDPASDL"
        myseq2= "ASDPASDP"
    
        """
        def indicator_helper(sequence):
            starts = [i.start() for i in re.finditer("P", sequence)]
            #no prolines
            if len(starts) == 0:
                return(0.0)
            #one proline
            if len(starts) == 1:
                return(starts[0] / (len(sequence)*1.))
            else:
                return(np.mean(np.diff(starts)) / (len(sequence)*1.))
        
        if verbose:
            print ("Adding Proline indicator...")
        sequence = self.modelmatrix.Sequence
        self.modelmatrix["TurnIndicator"] = sequence.apply(indicator_helper)
        self.features.append("TurnIndicator")
    
    def return_kmer_count(self, peptides):
        """
        NOT TESTED!
        
        Generates the gapped 3-mer counts from the peptide sequence
        
        Parameters:
        -----------------------------
        peptides: ar-like,
                  list of peptide sequences
        """
        counts = []
        for pep_i in peptides:
            counts_pepi = {}
            for i in range(0, len(pep_i)-3, 1):
                kmeri = pep_i[i:i+1] +"x"+ pep_i[i+2:i+3]
                #print (pep_i[i:i+3])
                #print (kmeri)
                if kmeri in counts_pepi:
                    counts_pepi[kmeri] += 1
                else:
                    counts_pepi[kmeri] = 1
            counts.append(counts_pepi)
            
        counts_df = pd.DataFrame(counts)
        counts_df = counts_df.fillna(0)
        return(counts_df)
         
     
    def return_polynomial_interactions(self):
        """
                NOT TESTED!
                
        Adds ppolynomial interactions as features to the dataframe.
        """
        names = {"x"+str(j):i for j,i in \
                 enumerate(self.modelmatrix.drop(["Sequence", "Fraction"], axis=1).columns) 
                 if (i != "Sequence") and (i != "Fraction")}
        names["1"] = "intercept"
    
        pf = PolynomialFeatures()
        df_pf = pd.DataFrame(pf.fit_transform(self.modelmatrix.drop(["Sequence", 
                                                             "Fraction"], axis=1)))
        df_pf.columns = pf.get_feature_names()
        
        #make column names readable...
        newnames = []
        for coli in df_pf.columns:
            if ("^" not in coli) and (coli.count("x") <2):
                newnames.append(names[coli])
            elif "^" in coli:
                parts = coli.split("^")
                newnames.append("{}^2".format(names[parts[0]]))
            else:
                parts = coli.split(" ")
                newnames.append("{}:{}".format(names[parts[0]], names[parts[1]]))
        df_pf.columns  = newnames
        df_pf["Fraction"] = self.modelmatrix["Fraction"]
        df_pf["Sequence"] = self.modelmatrix["Sequence"]
        return(df_pf)
    
    def filter_obs(self, min_obs=300, return_initial=True):
        """
        Filters the data based on the minimum number of observations per fraction.
        
        Parameter:
        .-------------------------
        """
        df_model_initial = self.modelmatrix.copy()
        df_model = self.modelmatrix
        min_frac = df_model.Fraction.min()
        max_frac = df_model.Fraction.max() + 1
        print ("NRows BEFORE min obs filter: {}".format(df_model.shape[0]))
        obs = {i:j for i,j in zip(np.arange(min_frac, max_frac), 
                                  np.bincount(df_model.Fraction)[1:])}
        df_model["nobs"] = [obs[i] for i in df_model.Fraction]
        df_model = df_model[df_model["nobs"] >=min_obs]
        print ("NRows AFTER min obs filter: {}".format(df_model.shape[0]))
        self.modelmatrix = df_model
        
        if return_initial:
            return(df_model_initial)
        
        
    def add_sandwich(self, aa="FYW", single_value=True, verbose=1):
        """
        Adds sandwich counts based on aromatics ()
        
        Parameters:
        ----------------------

        aa: str,
             amino acids to check fo rsandwiches. Def:FYW
        """
        if verbose:
            print ("adding sandwich feature...")
            
        peptides = self.modelmatrix.Sequence
        patterns = set(np.unique(["""{}x{}""".format(i[0], i[1]) for i in itertools.combinations_with_replacement(aa, 2)] + \
                    ["""{}x{}""".format(i[1], i[0]) for i in itertools.combinations_with_replacement(aa, 2)]))
        
        #count sandwich patterns between all aromatic aminocds and do not
        #distinguish between WxY and WxW.
        if single_value:
            s1 = "("+"|".join(aa)+")"
            counts_df = [len((re.findall((s1+"(\w{1,3})"+s1), pepi))) for pepi in peptides]
            
        else:
            counts = []
            for pep_i in peptides:
                counts_pepi = {}
                for i in range(0, len(pep_i)-3, 1):
                    kmeri = pep_i[i:i+1] +"x"+ pep_i[i+2:i+3]
                    #is this pattern relevant?
                    if kmeri in patterns:
                        #increase
                        if kmeri in counts_pepi:
                            counts_pepi[kmeri] += 1
                        #init
                        else:
                            counts_pepi[kmeri] = 1
                counts.append(counts_pepi)
                
            counts_df = pd.DataFrame(counts)
            counts_df = counts_df.fillna(0)
        self.modelmatrix["sandwich"] = counts_df
        self.features.append("sandwich")
        
        
    def get_uniquie_mods(df):
     """
     Returns the list of unique modifications from a dataframe. df must have
     the column "Modifications"].
     Super slow but quickly coded...
     """
     mod_dic = {}
     mods = df[df["Modifications"] != "Unmodified"]
     for idx, irow in mods.iterrows():
         names = irow["Modifications"].split(",")
         abbrev = re.findall("\(\w+\)", irow["Modified sequence"])
         for namei, abbi in zip(names, abbrev):
             namei_s = re.sub("\d ", "", namei)
             mod_dic[namei_s] = abbi
     mod_dic["n"] = "n-term"
     mod_dic["c"] = "c-term"
        
     mod_df = pd.DataFrame([mod_dic], columns=mod_dic.keys()).transpose().reset_index()
     mod_df.columns = ["Name", "Abbreviation"]
     return(mod_df)

#%% aux functions
def get_net_charge(sequence):
    """
    Computes net charge - or be more accurate an estimate of the contributed
    residue charge in a peptide ignoring the termini.
    
    Parameters:
    ------------------------
    sequence: str,
                Peptide Sequence
    """
    return(sequence.count("D")+sequence.count("E")+(0.3 * sequence.count("F")+
                                                    0.8 * sequence.count("W")+
                                                    0.6 * sequence.count("Y"))-
        sequence.count("K") - sequence.count("R"))
            
def prepare_sequence_for_pyteomics(mod_sequence):
    """
    Prepares the Sequence to work out of the box with pyteomics.
    In particular the Modified sequence patterns are adjusted.
    - "_" are removed
    - "()" are remove
    - "n" is added before the sequence
    - "c" is added before the last amino acid
    
    Parameters:
    -----------------------
    mod_sequence: str,
                  peptide sequence
    """
    
    #inverts the M(ox) to oxM    
    new_seq = re.sub(r'([A-Z])\(([a-z]+)\)', r'\2\1', mod_sequence)
    new_seq = re.sub('[\(\)_]', r'', new_seq)
    new_seq = "n"+new_seq[:-1]+"c"+new_seq[-1:]
    return (new_seq)

def extract_mods_n(modifications, mod_str="Oxidation", verbose=False):
    """
    Extracts the number of Oxidations from the sequence
    """
    match = re.search("(\d)|({})".format(mod_str), modifications)
    
    #no modifications
    if not match:
        n = 0
        
    #at least 1
    elif match.groups()[0] != None:
        n = match.groups()[0]
    
    #exactly one
    else:
        n = 1
        
    if verbose:
        print (n)
        print (modifications)
    return(n)

     
def add_shortest_distance(orig_sequence, opt="cterm", verbose=False):
    """
    Computes the shortest distance of a amino acids to n/cterm.
    E, D, C-term
    K, R, N-term
    
    Parameters:
    ---------------------
    orig_sequence: string,
                  amino acid string
    opt: str,
         either "cterm" or "nterm". Each are defined with a set of amino acids
         
         
    Returns:
    ---------------------
    int: distance to either termini specified
    """
    #NPC
    
    #define shortest distance of tragets to cterm
    if opt=="cterm":
        targets = "|".join(["E", "D"])
        sequence = orig_sequence[::-1]
        match = re.search(targets, sequence)

    #define shortest distance of tragets to nterm
    elif opt=="nterm":
        targets = "|".join(["K", "R"])
        sequence = orig_sequence
        match = re.search(targets, sequence)
        
    else:
        print ("ERROR!")
    
    #if there is a amino acid found...
    if match:
        pos = match.start() + 1
        aa = sequence[pos-1:match.end()]
    else:
        pos = 0
    if verbose:
        print ("#######################################")
        print ("distance from {} ({})".format(opt, targets))
        print (orig_sequence)
        print (pos)
        print (aa)
    return(pos)
    
     
def filter_peptides_fraction(df_evidence):
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
                        pass
                        #indic_dic[ii] = "Not Adjacent"
                #store results
                indic_dic[keepid] = "Keep"
                indicator[keepid] = True
                
        #ignore all peptides with ids in more than two fractions
        else:
            #indic_dic[ii] = ">2Fractions"
            pass
    return(indicator)
    
    
def marknonstandardaa(peptide):
    """
    Returns False if non standard amino acids are in there...
    """
    AAs = {"A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
           "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}
    
    for aap in peptide:
        if aap not in AAs:
            return(False)
        else:
            pass
    return(True)

def get_uniquie_mods(df):
     """
     Returns the list of unique modifications from a dataframe. df must have
     the column "Modifications"].
     Super slow but quickly coded...
     """
     mod_dic = {}
     mods = df[df["Modifications"] != "Unmodified"]
     for idx, irow in mods.iterrows():
         names = irow["Modifications"].split(",")
         abbrev = re.findall("\(\w+\)", irow["Modified sequence"])
         for namei, abbi in zip(names, abbrev):
             namei_s = re.sub("\d ", "", namei)
             mod_dic[namei_s] = abbi
     mod_dic["n"] = "n-term"
     mod_dic["c"] = "c-term"
        
     mod_df = pd.DataFrame([mod_dic], columns=mod_dic.keys()).transpose().reset_index()
     mod_df.columns = ["Name", "Abbreviation"]
     return(mod_df)
     
#%% convenience function
def preprocess_manuscript(infile, outpath, name, n_test=-1, mods=1, target="Fraction",
            correct=True, scale=True, from_CSV=False, add_poly=False, min_obs=300):
    """
    To function to process data files accordin to the published manuscript.
    
    Parameters:
    -----------------------------
    infile: str, 
            Can be a regular CSV path file or preferably a MaxQuant Evidence
            file.
        
    outpath: str, 
            Outpath for storing results.
            
    name: str, 
            Another identifier that is used for storing the results.
            
    n_test: int, 
            Not yet implemented. Used to downsample the input data for
            faster testing.
            
    mods: Bool, 
        If True, modifications are included in the modeling. If False
        they are ignored.
        
    target: str, 
            Target Column for the prediction.
            
    scale: Bool, 
            If True, StandardScaler from sklearn is used to scale the features.
            (default: false)
            
    from_CSV: Bool, 
            If True, the infile argument is expected to be normal CSV
            file with a Sequence column. (default: False)
            
    add_poly: Bool, 
            If True, Polynomial features from the final dataframe are
            added.(default: False)

    min_obs: int, 
            minimum number of peptides to be observed in a fraction to be 
            included in the data analysis. (default: 300)
    """
    #%%
    ##########################################################################
    df_settings = pd.DataFrame([infile, outpath, name, n_test, mods, 
                                correct, scale])
    df_settings.columns = ["Value"]
    df_settings["Name"] = ["infile", "outpath", "name", "n_test", "mods", 
                           "correct", "scale"]
    #==========================================================================
    # CONFIG
    #==========================================================================
    print ("Config...")
    outpath = outpath+"/"+name +"//"
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print ("Created folder {}".format(outpath))
    #==========================================================================
    #     Start processing
    #==========================================================================
    np.random.seed(42)
    matrix = ModelMatrix(name, loc=infile)
    
    if from_CSV:
        matrix.from_csv()
    else:
        matrix.from_evidence()
        #get mods 
        if "Modifications" in matrix.modelmatrix:
            mods_df = get_uniquie_mods(matrix.modelmatrix)
            mods_df.to_csv(outpath+"mods.csv")
    
    print ("Min. Fraction: {}".format(np.min(matrix.modelmatrix.Fraction)))
    print ("Min. Fraction: {}".format(np.max(matrix.modelmatrix.Fraction)))
    
    matrix.add_features()
    matrix.add_AA_count()
    matrix.add_positional_effect(5, "all")
    
    if correct:
        matrix.use_correction(lcp=-0.2)
        
    matrix.add_cterm_indicator(residue="K")
    matrix.add_cterm_indicator(residue="R")
    
    if mods:
        matrix.modelmatrix["Sequence"] = matrix.modelmatrix["PyteomicsSequence"]
        del matrix.modelmatrix["PyteomicsSequence"]
        
    matrix.add_patches(counts_only=True)
    matrix.add_peptide_properties()
    matrix.add_turn_indicator()
    matrix.add_sandwich(single_value=True) 
    
    if add_poly:
        #TODO
        pass
        
    #finalize data
    matrix.modelmatrix.Fraction = matrix.modelmatrix["Fraction"].copy().astype(int).values

    #store the unfilted data 
    df_model_nofilter = matrix.filter_obs(min_obs)
    df_model = matrix.modelmatrix.copy()
    
    print ("Model Matrix: {}".format(df_model.shape))
    print ("Model Matrix (Filter): {}".format(df_model_nofilter.shape))
    
    #%% Store results
    #split data into train for CV and validation, do so 
    try:
        del df_model["nobs"]
    except KeyError:
        pass
    
    train_df, val_df = np.split(df_model.sample(frac=1, random_state=42), [int(.75*len(df_model))])
        
    print ("Storing...")
    print ("Train df with filter: {}".format(train_df.shape))
    print ("Val df with filter: {}".format(val_df.shape))

    
    #store not-filtered data
    df_model.to_csv("{}/Modeldf_{}_{}.csv".format(outpath, name, df_model.shape[1]))
    train_df.to_csv("{}/Traindf_{}_{}.csv".format(outpath, name, train_df.shape[1]))
    val_df.to_csv("{}/Valdf_{}_{}.csv".format(outpath, name,  val_df.shape[1]))
    return(matrix, df_model, train_df, val_df)
    
