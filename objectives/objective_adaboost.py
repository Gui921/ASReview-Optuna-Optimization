from asreviewcontrib.models.classifiers import AdaBoost as adaboost
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def AdaBoost(trial):
    n_estimators_value = trial.suggest_int('n_estimators',10,100, step=1)
    learning_rate_value = trial.suggest_float('learning_rate', 0.0001, 2, log=True)

    model = adaboost(n_estimators=n_estimators_value,learning_rate=learning_rate_value)
    feature_extractor = Tfidf()
    
    return optimization_loop(model, feature_extractor)