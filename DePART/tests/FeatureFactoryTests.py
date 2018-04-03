# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:44:54 2018

@author: hanjo
"""
import unittest
import pandas as pd
from DePART.preprocessing import FeatureFactory as FF
from DePART.preprocessing import PrepFactory as PF
from pyteomics import parser

def to_df(exp_counts, cmp_counts, sequence):
    """
    """
    res_df = pd.DataFrame([exp_counts, cmp_counts]).transpose()
    res_df.columns = ["Exp", "Found"]
    res_df["Sequence"]  = sequence
    res_df["equal"] = [True if i == j else False for i,j in zip(exp_counts, cmp_counts)]
    return(res_df)
    
    
class EvaluationTest(unittest.TestCase):

    def test_acidic(self):
        df = pd.read_csv("data/TestPatterns.csv")
        
        exp_counts  = df["acidic"]
        cmp_counts = df["Sequence"].apply(FF.get_patches, args=[["D","E"]])
        assert((exp_counts == cmp_counts).all())


    def test_basic(self):
        df = pd.read_csv("data/TestPatterns.csv")
        
        exp_counts  = df["basic"]
        cmp_counts = df["Sequence"].apply(FF.get_patches, args=[["K","R"]])
        res_df = to_df(exp_counts, cmp_counts, df["Sequence"])

        if (exp_counts == cmp_counts).all():
            pass
        else:
            print (res_df)
            
        assert((exp_counts == cmp_counts).all())


    def test_aromatic(self):
        df = pd.read_csv("data/TestPatterns.csv")
        
        exp_counts  = df["aromatic"]
        cmp_counts = df["Sequence"].apply(FF.get_patches, args=[["F","Y","W"]])
        res_df = to_df(exp_counts, cmp_counts, df["Sequence"])

        if (exp_counts == cmp_counts).all():
            pass
        else:
            print (res_df)
            
        assert((exp_counts == cmp_counts).all())
        
        
    def test_mixed(self):
        df = pd.read_csv("data/TestPatterns.csv")
        
        exp_counts  = df["mixed"]
        cmp_counts = df["Sequence"].apply(FF.get_patches, args=(["K","R"],
                                                                ["D","E"]))
        res_df = to_df(exp_counts, cmp_counts, df["Sequence"])

        if (exp_counts == cmp_counts).all():
            pass
        else:
            print (res_df)
            
        assert((exp_counts == cmp_counts).all())
        
        
    def test_sandwich(self):
        df = pd.read_csv("data/TestSandwich.csv")
        
        exp_counts = df["sandwich"]
        cmp_counts = df["Sequence"].apply(FF.get_sandwich)
        res_df = to_df(exp_counts, cmp_counts, df["Sequence"])

        if (exp_counts == cmp_counts).all():
            pass
        else:
            print (res_df)
            
        assert((exp_counts == cmp_counts).all())
        
        
    def test_poscounts(self):
        #%%
        df = pd.read_csv("data/TestPosCounts.csv")
        #df["PyteomicsSequence"] = df["Sequence"].apply(FF.get_pyteomics_sequence)
        df_aa = FF.get_AA_matrix(df, True)
        
        length = df["Sequence"].apply(len)
        sum1 = df_aa[[i for i in df_aa.columns if len(i) < 2]].sum(axis=1)
        sum2 = df_aa.filter(regex="\w{2,}").sum(axis=1)
        sum3 = sum1 + sum2
        length == sum3
        
        T0 = (df_aa["NK0"] == df["NK0"]).all()
        T1 = (df_aa["NK1"] == df["NK1"]).all()
        T2 = (df_aa["NK2"] == df["NK2"]).all()
        T3 = (df_aa["NK3"] == df["NK3"]).all()
        T4 = (df_aa["NK4"] == df["NK4"]).all()
        T = (T0 and T1 and T2 and T3 and T4)
        
        R0 = (df_aa["K"] == df["K"]).all()
        R1 = (df_aa["CK1"] == df["CK1"]).all()
        R2 = (df_aa["CK2"] == df["CK2"]).all()
        R3 = (df_aa["CK3"] == df["CK3"]).all()
        R4 = (df_aa["CK4"] == df["CK4"]).all()
        R = (R0 and R1 and R2 and R3 and R4)
        
        res = T and R
        assert (True == res)
        #%%
        #assert((length == sum2).any())
        
    def test_modparser(self):
        """
        """
        df = pd.read_csv("data/TestModParser.csv")
        
        df["Sequence"] = df["Sequence"].apply(PF.remove_brackets)
        df["Sequence"] = df["Sequence"].apply(PF.replace_numbers)
        
        mod_dic, mods_seq = PF.extract_modifications(df["Sequence"], True)
        df["detected"] = [i[0] for i in mods_seq]
        df["Test"] = [True if i == j else False for i,j in zip(df["Mods"], 
                                                             df["detected"])]
        
        #%%
        mods = [i[1]+i[0] for i in mod_dic.items()]
        for seqi in df["NewSeqs"]:
            print (seqi)
            print (parser.amino_acid_composition(seqi,
                                                 labels=parser.std_labels +
                                                 mods))
        #%%
        assert (True == df["Test"].all())
        
if __name__ == '__main__':
    unittest.main()
