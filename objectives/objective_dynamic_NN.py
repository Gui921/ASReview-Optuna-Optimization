from asreviewcontrib.models.classifiers import DynamicNNClassifier
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def DynamicNN(trial):
    """
    Objective function for optimizing the Dynamic Neural Network classifier 
    using Optuna.
    
    Parameters:
        trial: An Optuna trial object that suggests hyperparameters.
    
    Returns:
        final_loss: A weighted loss from all datasets.
    """

    btch_size = trial.suggest_int('batch_size',8,64, step=4)
    shffl = trial.suggest_categorical('shuffle',[False,True])
    delta = trial.suggest_float('min_delta',0.0001,1, log=True)
    clss_weight = trial.suggest_float('class_weight',1,50, step=1)

    model = DynamicNNClassifier(batch_size=btch_size,shuffle=shffl,
                                min_delta=delta,class_weight=clss_weight)
    feature_extractor= Tfidf()

    return optimization_loop(model, feature_extractor)