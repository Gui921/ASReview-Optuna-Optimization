from asreviewcontrib.models.classifiers import AdaBoost as adaboost
from .utils import optimization_loop

class AdaBoost:

    """
    Objective class for optimizing the AdaBoost classifier using Optuna.
    
    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()

    def __call__(self, trial):
        
        n_estimators_value = trial.suggest_int('n_estimators',10,100, step=1)
        learning_rate_value = trial.suggest_float('learning_rate', 0.0001, 2, log=True)

        model = adaboost(n_estimators=n_estimators_value,learning_rate=learning_rate_value)
        
        return optimization_loop(model, self.feature_extractor)