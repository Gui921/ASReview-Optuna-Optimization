from asreview.models.classifiers import SVMClassifier
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def SVM(trial):
    gamma_value = trial.suggest_categorical('gamma',['scale','auto'])
    cls_weight_value = trial.suggest_float('class_weight', 0.001, 2, log=True)
    c_value = trial.suggest_float('C', 0.1, 30, log=True)

    model = SVMClassifier(gamma=gamma_value, class_weight=cls_weight_value
                          ,C=c_value)
    feature_extractor = Tfidf()

    return optimization_loop(model, feature_extractor)