from asreviewcontrib.models.classifiers import XGBoost as xgboost
from .utils import optimization_loop

class XGBoost:
    """
    Objective class for optimizing the XGBoost classifier using Optuna.
    
    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()
    
    def __call__(self, trial):
        
        max_depth_value = trial.suggest_int('max_depth', 1, 15, step=1)
        learning_rate_value = trial.suggest_float('learning_rate', 0.0001, 2, log=True)
        n_estimators_value = trial.suggest_int('n_estimators',50,150,step=1)
        gamma_value = trial.suggest_float('gamma', 0, 1, step = 0.001)
        reg_lambda_value = trial.suggest_float('reg_lambda', 0.1, 3, log=True)

        model = xgboost(max_depth=max_depth_value, learning_rate=learning_rate_value,
                        n_estimators=n_estimators_value,gamma=gamma_value,
                        reg_lambda=reg_lambda_value)

        return optimization_loop(model, self.feature_extractor)