# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:28:55 2018

@author: hanjo
"""
from DePART.learning import processing as LP
from DePART.preprocessing import FeatureFactory as FF

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model
#
# not really necessary and probably more confusing since the
# sklearn API would be broken. Instead provide an example.
#class DePART_Regressor():
#    """
#    Convenience Wrapper for using DePART. Features are generated automatically
#    from the input data X.
#    """
#    
#    def __init__(self, model):
#        """
#        """
#        self.model = model
#        
#    def fit(self, X, epochs=100, batch_size=512):
#        """
#        Fits the model
#        """
#        #regression
#        X_train_full, Seqs_train_full, y_train_full = LP.process_df(X)
#        self.train_data_x = X_train_full
#        self.train_sequences = Seqs_train_full
#        self.train_data_y = y_train_full
#        self.history = self.model.fit(X_train_full, y_train_full, 
#                                      epochs=epochs,
#                                      batch_size=batch_size)
#        
#    def predict(self, X):
#        """
#        Predict the outcomes.
#        """
#        X_train_full, Seqs_train_full, y_train_full = LP.process_df(X)
#        self.pred_data_x = X_train_full
#        self.pred_sequences = Seqs_train_full
#        self.pred_data_y = y_train_full        
#        return(self.model.predict(X_train_full))        
#           
#        
#    def __process_df__(self):
#        #todo
#        pass
#        
#    
#class DePART_Classifier(DePART_Regressor):
#    """
#    Convenience Wrapper for using DePART. Features are generated automatically
#    from the input data X. X needs to have the following columns: Sequence, Fraction
#    """
#    
#    def __init__(self, model=None, label_encoder=LabelEncoder()):
#        """
#        """
#        self.model = model
#        self.label_encoder = label_encoder
#        
#        
#    def fit(self, df, epochs=100, batch_size=512):
#        """
#        Fits the model after feature generation
#        """
#        
#        #generate features
#        Generator = FF.FeatureGenerator()
#        ff_df = Generator.create_all_features(df)
#        #regression
#        X_train_full, Seqs_train_full, y_train_full = LP.process_df(X)
#        self.train_data_x = X_train_full
#        self.train_sequences = Seqs_train_full
#        self.train_data_y = y_train_full
#        
#        #label encoders
#        self.label_encoder.fit(y_train_full)
#    
#        self.history = self.model.fit(X_train_full, y_train_full, 
#                                      epochs=epochs,
#                                      batch_size=batch_size)
#        
#    def predict(self, X):
#        """
#        Predict the outcomes.
#        """
#        X_train_full, Seqs_train_full, y_train_full = LP.process_df(X)
#        self.pred_data_x = X_train_full
#        self.pred_sequences = Seqs_train_full
#        self.pred_data_y = y_train_full        
#        return(self.model.predict(X_train_full).argmax(axis=1) + 1)  
#    
#    def predict_proba(self, X):
#        """
#        Predicts the probability for each class.
#        """
#        X_train_full, Seqs_train_full, y_train_full = LP.process_df(X)
#        self.pred_data_x = X_train_full
#        self.pred_sequences = Seqs_train_full
#        self.pred_data_y = y_train_full
#        
#        return(self.model.predict(np.array(X_train_full)))

def SAX_Model(ini_mode="normal", optimizer="adam", loss="categorical_crossentropy", 
              act=["relu", "tanh", "relu"], dr1=0.0, dr2=0.0, dr3=0.0, 
              input_dim=218, output_dim=29):
    """
    Returns the best-performing neural network model from the manuscript.
    """
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, kernel_initializer=ini_mode, 
                    activation=act[0]))
    model.add(Dropout(dr1)) 
    model.add(Dense(40, kernel_initializer='normal', activation=act[1]))
    model.add(Dropout(dr2)) 
    model.add(Dense(35, kernel_initializer='normal', activation=act[2]))
    model.add(Dropout(dr3)) 
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, 
                  metrics=['categorical_accuracy', 'accuracy'])
    return(model)


def FNN_Classifier(ini_mode="normal", optimizer="adam", 
                              loss="categorical_crossentropy", 
                              act=["relu", "tanh", "relu"],
                              dropout=[0.0, 0.0, 0.0], input_dim=218,
                              output_dim=29):
    """
    Returns the best-performing neural network model from the manuscript.
    """
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, kernel_initializer=ini_mode, 
                    activation=act[0]))
    model.add(Dropout(dropout[0])) 
    model.add(Dense(40, kernel_initializer='normal', activation=act[1]))
    model.add(Dropout(dropout[1])) 
    model.add(Dense(35, kernel_initializer='normal', activation=act[2]))
    model.add(Dropout(dropout[2])) 
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, 
                  metrics=['categorical_accuracy', 'accuracy'])
    return(model)
    
    
def FNN_Regressor(ini_mode="normal", optimizer="adam", 
                  loss="mse", 
                  act=["relu", "tanh", "relu"],
                  dropout=[0.0, 0.0, 0.0], input_dim=218):
    """
    Returns the best-performing neural network model from the manuscript.
    """
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, kernel_initializer=ini_mode, 
                    activation=act[0]))
    model.add(Dropout(dropout[0])) 
    model.add(Dense(40, kernel_initializer='normal', activation=act[1]))
    model.add(Dropout(dropout[1])) 
    model.add(Dense(35, kernel_initializer='normal', activation=act[2]))
    model.add(Dropout(dropout[2])) 
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])
    return(model)
    
    