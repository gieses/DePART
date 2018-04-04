# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:46:25 2018

@author: Hanjo
"""
import re
import numpy as np
import itertools
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pyteomics import parser
from DePART.preprocessing import PrepFactory as PF

class FeatureGenerator():
    """
    """
    
    def __init__(self, enable_test=True, pos_specific=True,
                 lcp=-0.2, correct=False):
        """
        Initalizes the feature generator. By default the
        Generator will run a small test on the input data after
        calling the fit function.
        """
        self.enable_test = True
        self.options = {"pos_specific":pos_specific, "lcp":lcp,
                        "correct":correct}
        
    def create_all_features(self, df):
        """
        Creates all features.
        
        Parameters:
        ---------------------------
        df: df, requires the column "Sequence".
        """
    
        df["Sequence"] = df["Sequence"].apply(simply_alphabet).values
         
        ff_df = pd.DataFrame()    
        
        #features
        #make sure the sequence column has now strange formatting
        ff_df["Sequence"] = df["Sequence"].apply(get_unmodified_sequence)
        ff_df["estimated_charge"] = ff_df["Sequence"].apply(get_estimated_charge)
        ff_df["loglength"] = ff_df["Sequence"].apply(get_loglength)
        ff_df["cterm"] = ff_df["Sequence"].apply(get_shortest_distance, args=["cterm"])
        ff_df["nterm"] = ff_df["Sequence"].apply(get_shortest_distance, args=["nterm"])
        ff_df["netcharge"] = ff_df["Sequence"].apply(get_residue_charge)
        ff_df["nterm_res"] = ff_df["Sequence"].apply(get_cterm_residue_indicator)
        
        #biopython features
        ff_df["turnP"] = ff_df["Sequence"].apply(get_turn_indicator)
        ff_df["turn"] = ff_df["Sequence"].apply(get_structure_perc, args=["turn"])
        ff_df["helix"] = ff_df["Sequence"].apply(get_structure_perc, args=["helix"])
        ff_df["sheet"] = ff_df["Sequence"].apply(get_structure_perc, args=["sheet"])
        ff_df["pi"] = ff_df["Sequence"].apply(get_pi)
        ff_df["aromaticity"] = ff_df["Sequence"].apply(get_aromaticity)
        #ff_df["weight"] = np.sqrt(ff_df["Sequence"].apply(get_weight))
        #ff_df["flexibility"] = df["Sequence"].apply(FF.get_flexibility)
        #%%
        #attention here we should use the modified sequences
        if "Modified sequence" in df.columns:
            orig_sequences = df["Modified sequence"]
        else:
            orig_sequences = df["Sequence"]

        #save y data fraom original df
        ff_df["Fraction"] = df["Fraction"]
        nterm_mods = extract_nterm_mods(orig_sequences)
        orig_sequences = orig_sequences.apply(PF.remove_brackets)
        
        #add gl/ac features
        for mod in nterm_mods:
            ff_df[mod] = orig_sequences.apply(get_nterm_mod, args=[mod])
            
        orig_sequences = orig_sequences.apply(PF.replace_numbers)
        orig_sequences = orig_sequences.apply(PF.replace_nterm_mod)
        
        mod_dic, _ = PF.extract_modifications(orig_sequences)
        orig_sequences = orig_sequences.apply(PF.rewrite_modsequences)
        #%% do the AA count
        mods = [i[1]+i[0] for i in mod_dic.items()]  
        #print ("Found the following modifications: {}".format(mods))
        ff_df = ff_df.reset_index()
        del ff_df["index"]
        aa_df = get_AA_matrix(orig_sequences, 
                              pos_specific=self.options["pos_specific"], 
                              lcp=self.options["lcp"], 
                              correct=self.options["correct"],
                              residues=parser.std_amino_acids + mods)
        
        ff_df = pd.concat([ff_df, aa_df], axis=1)
        return(ff_df)

    @staticmethod
    def __test__all_sequences__(seq):
        """
        """
        #TODO

def get_nterm_mod(seq, mod):
    """
    Checks for a given nterminal mod. If the sequences contains the mod a 1
    is returned, else 0.
    """
    if seq.startswith(mod):
        return (1)
    else:
        return(0)
    
def get_unmodified_sequence(mod_seq, include_terms=False):
    """
    Prepares the Sequence to work out of the box with pyteomics.
    In particular the Modified sequence patterns are adjusted.
    - "_" are removed
    - "()" are remove
    optional:
    - "n" is added before the sequence
    - "c" is added before the last amino acid
    
    Parameters:
    -----------------------
    mod_seq: str,
        peptide sequence
    """
    
    #inverts the M(ox) to oxM    
    new_seq = re.sub(r'([A-Z])\(([a-z]+)\)', r'\2\1', mod_seq)
    new_seq = re.sub('[\(\)_]', r'', new_seq)
    if include_terms:
        new_seq = "n"+new_seq[:-1]+"c"+new_seq[-1:]
    return (new_seq)
 

def get_loglength(seq):
    """
    Compute loglength of the sequence
    
    Parameters:
    -----------------------
    seq: str,
          peptide sequence
    """
    return(np.log(len(seq)))
    
  
def get_shortest_distance(seq, opt="nterm"):
    """
    Computes the shortest distance of D/E, K/R to the C, N-term.
    """
    return(1.*add_shortest_distance(seq, opt=opt, verbose=False))
    
    
def get_cterm_residue_indicator(seq):
    """
    Returns 1 if Lysine 0 if Arg
    """
    if seq[-1:] == "K":
        return (1)
    else:
        return (0)
    

def get_nmods(mod_seq, mod_str="Oxidation"):
    """
    Get the number of modifications
    """
    return(int(extract_mods_n(mod_seq, mod_str)))
    
    
def get_estimated_charge(seq):
    """
    Computes net charge - or be more accurate an estimate of the contributed
    residue charge in a peptide ignoring the termini.
    
    Parameters:
    ------------------------
    sequence: str,
                Peptide Sequence
    """
    return(seq.count("D")+seq.count("E")+ (0.3 * seq.count("F")+
                                           0.8 * seq.count("W")+
                                           0.6 * seq.count("Y"))- 
        seq.count("K") - seq.count("R"))
    
    
def get_residue_charge(seq):
    """
    Computes net charge - or be more accurate an estimate of the contributed
    residue charge in a peptide ignoring the termini.
    
    Parameters:
    ------------------------
    seq: str,
                Peptide Sequence
    """
    return(seq.count("D")+seq.count("E") - seq.count("K") - seq.count("R"))
    

def get_aa_count(pepseq, residue, pos=-1, direction="N"):
    """
    """
    
    if pos == -1:
        return(pepseq.count(residue))
    else:
        if direction == "N":
            return(pepseq[pos:pos+1].count(residue))
        else:
            return(pepseq[-pos-1:][0].count(residue))
    
# =============================================================================
#     helper functions
# =============================================================================
    
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
        print ("ERROR! Option {} not supported.".format(opt))
    
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
    
def extract_nterm_mods(sequences):
    """
    """
    #matches all nterminal mods, e.g. glD or acA
    nterm_pattern = re.compile(r'\b([a-z]+)([A-Z])')
    mods = []
    #test each sequence for non-AA letters
    for ii, seqi in enumerate(sequences):
        nterm_match = re.findall(nterm_pattern, seqi)
        #nterminal acetylation
        if len(nterm_match) == 0:
            pass
        else:
            mods.append([nterm_match[0][0]])
    return(mods)

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
    
    
    
def get_patches(seq, aa_set1=["D", "E"], aa_set2=None, counts_only=True):
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
        
    aminoacids: list,
                The following features were intended:
                    - D, E (acidic)
                    - K, R (basic)
                    - W, F, Y (aromatics)
                    - K, R, D, E (mixed)
                    
    counts_only: bool,
                if True DE and ED are added in a single column "acidic_patterns",
                if False, DE, ED are counts are added in separate columns.
                Same is true for the combinations of KRH and WYF.
  
    
    aa_set1 = ["K", "R"]
    aa_set2 = ["D", "E"]
    seq = "XXEEXXDEXXKRRD"
    re.findall(pattern, seq)    

    """
    #this representation is used to be easily also used if not only
    #the counts but also the patterns are requested.
    if aa_set2 == None:
        ac_combs = ["".join(i) for i in itertools.combinations_with_replacement(aa_set1, 2)]
        pattern = re.compile("[" + "|".join(ac_combs) +"]{2,}")
    else:
        ac_combs = ["".join(i) for i in list(itertools.product(aa_set1, aa_set2))]
        ac_combs = ac_combs + ["".join(reversed(i)) for i in list(itertools.product(aa_set1, aa_set2))]
        p1 = "|".join(aa_set1)
        p2 = "|".join(aa_set2)
        pattern = re.compile("([{}]+[{}]+)|[{}]+[{}]+".format(p1, p2, p2,p1))
        
    #just count the patterns (DD, DDD) and do not distinguish between
    #different patterns of the same type
    if counts_only:
        return(len(re.findall(pattern, seq)))
    else:
        res = {}
        for pattern in ac_combs:
            res[pattern] = str(seq).count(pattern)
        return(res)

def get_sandwich(seq, aa="FYW", single_value=True):
    """
    Adds sandwich counts based on aromatics ()
    
    Parameters:
    ----------------------

    aa: str,
         amino acids to check fo rsandwiches. Def:FYW
    """
    #count sandwich patterns between all aromatic aminocds and do not
    #distinguish between WxY and WxW.
    pattern = re.compile(r"(?=(["+aa+"][^"+aa+"]["+aa+"]))")
    return(len(re.findall(pattern, seq)))


def get_structure_perc(seq, structure="helix"):
    """
    """
    bio_seq = ProteinAnalysis(seq)
    helix, turn, sheets = bio_seq.secondary_structure_fraction()
    
    if structure == "helix":
        return(helix)
        
    elif structure == "turn":
        return(turn)
        
    else:
        return(sheets)
        
def get_gravy(seq):
    """
    Gets the gravy of the sequence.
    """
    bio_seq = ProteinAnalysis(seq)
    return(bio_seq.gravy())
  
def get_aromaticity(seq):
    """
    Gets the gravy of the sequence.
    """
    bio_seq = ProteinAnalysis(seq)
    return(bio_seq.aromaticity())
    
def get_pi(seq):
    """
    Gets the gravy of the sequence.
    """
    bio_seq = ProteinAnalysis(seq)
    return(bio_seq.isoelectric_point())
    
    
def get_turn_indicator(seq, verbose=1):
    """
    Computes the average number of amino acids between Proline residues
    
    Example:
    -----------------------
    myseq = "ASDPASDL"
    myseq2= "ASDPASDP"

    """
    starts = [i.start() for i in re.finditer("P", seq)]
    #no prolines
    if len(starts) == 0:
        return(0.0)
        
    #one proline
    if len(starts) == 1:
        return(starts[0] / (len(seq)*1.))
        
    else:
        return(np.mean(np.diff(starts)) / (len(seq)*1.))
        
def get_instabillity(seq):
    """
    Gets instability (Biopython API).
    """
    bio_seq = ProteinAnalysis(seq)
    return(bio_seq.instability_index())
    
    
def get_weight(seq):
    """
    Get weight ignoreing modifications.
    """
    bio_seq = ProteinAnalysis(seq)
    return(bio_seq.molecular_weight())
    
    
def get_flexibility(seq):
    """
    Get weight ignoreing modifications.
    """
    bio_seq = ProteinAnalysis(seq)
    return(bio_seq.flexibility())
    
    
def get_AA_matrix(sequences, pos_specific=False, ntermini=5, lcp=1, mods=0,
                  correct=False, residues=parser.std_amino_acids):
    """
    Counts the amino acid in a peptide sequence. Counting uses the
    pyteomics amino_acid composition. Modified residues of the pattern
    "modA" are already supported.

    If the modifications should not be considered another sequence column 
    can be used. As read on the pyteomics doc an "lcp" factor can substantially 
    increase the prediction accuracy.
    
    Parameters:
    -----------------------------------
    df: ar, with sequences
    
    seq_column: string,
                sequence column that is used to generate the features
                
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
    df = pd.DataFrame()
    df["Sequence"] = sequences.copy()
    #create dataframe with counts
    aa_counts = [parser.amino_acid_composition(i) for i in df["Sequence"]]
    aa_count_df = pd.DataFrame(aa_counts).replace(np.nan, 0)
    #only count without position index
    if pos_specific:
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
        #todo: fixme
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
                    count_df.at[ii, nterm] = count_df.loc[ii][nterm]+1
                    
                if seq[-i-1] in residues_hash:
                    cterm = "C"+seq[-i-1]+str(i)
                    #sinec the last amino acid is usually K/R don't add unnecessary
                    #features here
                    if i != 0:
                        count_df.at[ii, cterm] = count_df.loc[ii][cterm]+1
        
        #correct other counts
        #by substracting the sequence specific counts
        new_df = aa_count_df.join(count_df)
        #iterate over columns
        for res in residues:
            tmp_df = new_df.filter(regex="(N|C){}\d".format(res))
            sums = tmp_df.sum(axis=1)
            #correct the internal counts
            new_df[res] = new_df[res] - sums
    else:
        return(aa_count_df)
        
    #multiply each raw value by a correction term, see pyteomics docu
    #for details ("lcp")    
    if correct:
        cfactor = 1. + lcp * np.log(df["Sequence"].apply(len))
        new_df = new_df.mul(cfactor, axis=0)
    
    new_df = new_df.replace(np.nan, 0)
    return(new_df)

def simply_alphabet(seq):
    """
    Some sequences are encoded with 'U', arbitrarly choose C as
    residue to replace any U (Selenocystein)
    """
    return(seq.replace("U", "C"))
    