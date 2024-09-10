from asreview.models.classifiers import NaiveBayesClassifier
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def NaiveBayes(trial):
    """
    Objective function for optimizing the Naive Bayes classifier using Optuna.
    
    Parameters:
        trial: An Optuna trial object that suggests hyperparameters.
    
    Returns:
        final_loss: A weighted loss from all datasets.
    """

    alpha_value = trial.suggest_float('alpha', 0.1, 10.0, log = True)
    
    model = NaiveBayesClassifier(alpha=alpha_value)
    feature_extractor= Tfidf()

    return optimization_loop(model, feature_extractor)