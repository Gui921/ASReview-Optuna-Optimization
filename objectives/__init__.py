from .objective_nb import NaiveBayes
from .objective_2_layer_NN import NN_2_Layer
from .objective_xgboost import XGBoost
from .objective_dynamic_NN import DynamicNN
from .objective_logistic import Logistic
from .objective_RF import RandomForest
from .objective_svm import SVM
from.objective_adaboost import AdaBoost

from asreview.models.feature_extraction import Tfidf
from asreviewcontrib.models.feature_extraction import SBERT, Doc2Vec, LaBSE, MXBAI

OBJECTIVES = {
    'NaiveBayes': NaiveBayes,
    'NN_2_Layer': NN_2_Layer,
    'XGBoost': XGBoost,
    'DynamicNN': DynamicNN,
    'Logistic' : Logistic,
    'RandomForest': RandomForest,
    'SVM' : SVM,
    'AdaBoost': AdaBoost
}

FEATURE_EXTRACTORS = {
    'Tfidf' : Tfidf,
    'SBERT' : SBERT,
    'Doc2Vec' : Doc2Vec,
    'LaBSE' : LaBSE,
    'MXBAI' : MXBAI
}

__all__ = ['NaiveBayes', 'NN_2_Layer','XGBoost','DynamicNN','Logistic','RandomForest','SVM','AdaBoost']