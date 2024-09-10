from asreview.models.classifiers import RandomForestClassifier
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def RandomForest(trial):
    num_est_value = trial.suggest_int('n_estimators', 50, 200, step = 1)
    mx_feat_value = trial.suggest_int('max_features', 1, 20, step = 1)
    cls_weight_value = trial.suggest_float('class_weight', 0.1, 10, log= True)

    model = RandomForestClassifier(n_estimators=num_est_value,max_features=mx_feat_value
                                   ,class_weight=cls_weight_value)
    
    feature_extractor = Tfidf()

    return optimization_loop(model, feature_extractor)