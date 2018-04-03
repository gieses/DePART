# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:42:04 2018

@author: hanjo

thx to https://stackoverflow.com/a/30493366 for provding this to the world.
"""


import argparse
#in_train = "C:\\Users\\Hanjo\\Spyder_DePART\\DePART\\data\\evidence_SAX_Trost.txt"
#in_valid = "C:\\Users\\Hanjo\\Spyder_DePART\\DePART\\data\\evidence_SAX_Trost.txt"

#if __name__ == "__main__":
#    #%%
#    """
#    """
    
description = """\
'DePART - Deep Learning for Predicting Anion Exchange Chromatography Retention Times.'

This script trains the neural network on the given training data. If specifid,
the model is used to make predictions on the new data.
"""

# Instantiate the parser
parser = argparse.ArgumentParser(description=description)

#    # Required positional argument
#    parser.add_argument('train', type=int,
#                        help='Required loca')
#    # Optional positional argument
#    parser.add_argument('opt_pos_arg', type=int, nargs='?',
#                        help='An optional integer positional argument')
# Optional argument
parser.add_argument('-t', '--train', type=str,
                    help='File location of the training data.', required=True)

parser.add_argument('-p','--pred', type=str,
                    help='File location of the prediction data.', required=True)
#arguments with default
parser.add_argument('-r', '--reader', nargs='?', const=1, type=str, default="MaxQuant", help="Choose the reader that is going to be used. If the results are in MaxQuant format (evidence file) choose MaxQuant. If the data are in a CSV format with the two columns Sequence & Fraction, choose CSV")
parser.add_argument('-o','--min_obs', nargs='?', const=1, type=str, default=1, help="Minimum obsercations per fraction required for the training. Recommended to leave as 1.")
parser.add_argument('-s', '--store_features', nargs='?', const=1, type=bool, default=False, help="If enabled the feature tables from the input data are writte to the disk.")
parser.add_argument('-c', '--chrom', nargs='?', const=1, type=str, default="hsax", help="If 'hsax' is given as option the predictions are generated from a neural networ classifier. If 'RP' is choosen the results are from a regression classifier.")
parser.add_argument('-e', '--CV', nargs='?', const=1, type=bool, default=False, help="Enable cross-validation on the training data to estimate performance.")
parser.add_argument('-n', '--nfold', nargs='?', const=1, type=int, default=5, help="If cross-validation (CV) is enabled performs n-fold cross-validation.")
parser.add_argument('-m', '--mods', nargs='?', const=1, type=bool, default=False, help="If True peptide modifications are processed as if a new amino acid is introduced to the alphabet, e.g. Mox would introduce M and Mox as amino acids. If false, all modified peptides are removed from the data")
parser.add_argument('-x', '--epochs', nargs='?', const=1, type=int, default=100, help="The number of epochs the neural network is trained.")
parser.add_argument('-b', '--batchsize', nargs='?', const=1, type=int, default=512, help="The number of epochs the neural network is trained.")
args = parser.parse_args()

#summarize options
print("Settings:")
print("______________________________________")
print("Train:", args.train)
print("Predict:", args.pred)
print("Reader:", args.reader)
print("Min Obs.:", args.min_obs)
print("Store Features:", args.store_features)
print("LC:", args.chrom)
print("CV:", args.CV)
print("nfold:", args.nfold)
print("mods:", args.mods)
print("epochs:", args.epochs)
print("batchsize:", args.batchsize)
print("______________________________________")

#do some tests on the parameters
if args.nfold < 5:
    parser.error("You should perform at least CV-folds for a good error estimate.")


from DePART.wrapper import  wDePART
#call depart
wDePART.depart(train_loc=args.train,
               pred_loc=args.pred,
               reader=args.reader,
               min_obs=args.min_obs,
               store_features=args.store_features,
               cv=args.CV,
               nfolds=args.nfold,
               use_mods=args.mods,
               chrom=args.chrom)