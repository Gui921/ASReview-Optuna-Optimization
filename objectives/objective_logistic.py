from asreview.models.classifiers import LogisticClassifier
from .utils import optimization_loop

class Logistic:
    """
    Objective class for optimizing the Logistic classifier using Optuna.
    
    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()

    def __call__(self, trial):
        c_value = trial.suggest_float('C',0.01, 5, log=True)
        cls_weight_value = trial.suggest_float('class_weight', 0.01, 10, log=True)

        model = LogisticClassifier(C=c_value,class_weight=cls_weight_value)

        return optimization_loop(model, self.feature_extractor)