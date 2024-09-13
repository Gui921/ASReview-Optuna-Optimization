from asreviewcontrib.models.classifiers import NN2LayerClassifier
from .utils import optimization_loop

class NN_2_Layer:
    """
    Objective class for optimizing the 2 Layer Neural Network classifier 
    using Optuna.

    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation

    Returns:
        final_loss: A weighted loss from all datasets.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()
    
    def __call__(self, trial):
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
        
        return optimization_loop(model, self.feature_extractor)