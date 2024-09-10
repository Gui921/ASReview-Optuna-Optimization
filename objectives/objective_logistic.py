from asreview.models.classifiers import LogisticClassifier
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def Logistic(trial):
    c_value = trial.suggest_float('C',0.01, 5, log=True)
    cls_weight_value = trial.suggest_float('class_weight', 0.01, 10, log=True)

    model = LogisticClassifier(C=c_value,class_weight=cls_weight_value)
    feature_extractor = Tfidf()

    return optimization_loop(model, feature_extractor)