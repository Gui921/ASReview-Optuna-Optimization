from asreviewcontrib.models.classifiers import DynamicNNClassifier
from .utils import optimization_loop

class DynamicNN():
    """
    Objective class for optimizing the Dynamic Neural Network classifier 
    using Optuna.
    
    Arguments:
        feature_extractor: Feature Extracture model to be used in the simulation
        
    Returns:
        final_loss: A weighted loss from all datasets.
    """

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor()
    
    def __call(self, trial):

        btch_size = trial.suggest_int('batch_size',8,64, step=4)
        shffl = trial.suggest_categorical('shuffle',[False,True])
        delta = trial.suggest_float('min_delta',0.0001,1, log=True)
        clss_weight = trial.suggest_float('class_weight',1,50, step=1)

        model = DynamicNNClassifier(batch_size=btch_size,shuffle=shffl,
                                    min_delta=delta,class_weight=clss_weight)

        return optimization_loop(model, self.feature_extractor)