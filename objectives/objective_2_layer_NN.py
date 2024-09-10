from asreviewcontrib.models.classifiers import NN2LayerClassifier
from asreview.models.feature_extraction import Tfidf
from .utils import optimization_loop

def NN_2_Layer(trial):
    """
    Objective function for optimizing the 2 Layer Neural Network classifier 
    using Optuna.
    
    Parameters:
        trial: An Optuna trial object that suggests hyperparameters.
    
    Returns:
        final_loss: A weighted loss from all datasets.
    """

    density = trial.suggest_int('dense_width',64, 256, step=1)
    optim = trial.suggest_categorical('optimizer', ['rmsprop','adam'])
    lr = trial.suggest_float('learn_rate', 0.001,1, step=0.01)
    reg = trial.suggest_float('regularization',0.00001,0.002, step=0.0001)
    btch_size = trial.suggest_int('batch_size',8,64, step=4)
    shffl = trial.suggest_categorical('shuffle',[False,True])
    clss_weight = trial.suggest_float('class_weight',1,50, step=1)

    model = NN2LayerClassifier(dense_width=density,optimizer=optim,learn_rate=lr,
                               regularization=reg, batch_size= btch_size,
                               shuffle=shffl, class_weight=clss_weight)
    feature_extractor= Tfidf()
    
    return optimization_loop(model, feature_extractor)