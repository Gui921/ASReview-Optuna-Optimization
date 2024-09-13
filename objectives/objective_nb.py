from asreview.models.classifiers import NaiveBayesClassifier
from .utils import optimization_loop

class NaiveBayes:
    """
    Objective class for optimizing the Naive Bayes classifier using Optuna.

    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()

    def __call__(self, trial):

        alpha_value = trial.suggest_float('alpha', 0.1, 10.0, log = True)
        model = NaiveBayesClassifier(alpha=alpha_value)

        return optimization_loop(model, self.feature_extractor)