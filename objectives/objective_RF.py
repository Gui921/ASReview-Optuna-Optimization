from asreview.models.classifiers import RandomForestClassifier
from .utils import optimization_loop

class RandomForest:
    """
    Objective class for optimizing the Random Forest classifier using Optuna.
    
    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()
    
    def __call__(self, trial):
        num_est_value = trial.suggest_int('n_estimators', 50, 200, step = 1)
        mx_feat_value = trial.suggest_int('max_features', 1, 20, step = 1)
        cls_weight_value = trial.suggest_float('class_weight', 0.1, 10, log= True)

        model = RandomForestClassifier(n_estimators=num_est_value,max_features=mx_feat_value
                                    ,class_weight=cls_weight_value)

        return optimization_loop(model, self.feature_extractor)