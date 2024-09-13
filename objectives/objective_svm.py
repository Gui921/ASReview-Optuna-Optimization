from asreview.models.classifiers import SVMClassifier
from .utils import optimization_loop

class SVM:
    """
    Objective class for optimizing the SVM classifier using Optuna.
    
    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()

    def __call__(self, trial):

        gamma_value = trial.suggest_categorical('gamma',['scale','auto'])
        cls_weight_value = trial.suggest_float('class_weight', 0.001, 2, log=True)
        c_value = trial.suggest_float('C', 0.1, 30, log=True)

        model = SVMClassifier(gamma=gamma_value, class_weight=cls_weight_value
                            ,C=c_value)

        return optimization_loop(model, self.feature_extractor)