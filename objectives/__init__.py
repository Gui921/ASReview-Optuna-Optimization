from .objective_nb import NaiveBayes
from .objective_2_layer_NN import NN_2_Layer
from .objective_xgboost import XGBoost
from .objective_dynamic_NN import DynamicNN

OBJECTIVES = {
    'NaiveBayes': NaiveBayes,
    'NN_2_Layer': NN_2_Layer,
    'XGBoost': XGBoost,
    'DynamicNN': DynamicNN
}

__all__ = ['NaiveBayes', 'NN_2_Layer','XGBoost','DynamicNN']