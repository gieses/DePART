# DePART - Deep Learning for Predicting Anion Exchange Chromatography Retention Times.

DePART is a high-level python package for retention time prediction in hydrophilic strong anion exchange chromatography (hSAX). DePART uses several packages for the pre-processing of (MaxQuant) peptide tables and the final machine learning prediction (custom format peptide tables are also supported).There are convenience functions implemented to use the same preprocessing as used in the manuscript.
The core function of DePART is the training & prediction of the 'elution time' of peptides from hSAX prepfractionation. Elution time is in our workflow exchangeable with the fraction from an prefractionation experiment, e.g. a discrete number.



Getting Started
---

DePART requires the installation of the following packages and ideally python 3.5.

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
Manual installation required.
 

 CLI interface
 ----
 DePART comes with a command line script... [description tbd]
 
 Notes
 ----
 The current master & dev branch also supports the prediction of RP-LC RT times.
 Changing the architecture and the parameters of neural network can be done
 by editing the RTMLlib. A few parameters can be changed by the function call,
 e.g. the activation function (swish is also supported).
 

Authors
----
* Sven Giese

License
----
This project is licensed under the MIT License - see the LICENSE.md file for details
