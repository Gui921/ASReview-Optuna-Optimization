import shutil
import tempfile

from asreview import open_state
from asreview import ASReviewData
from asreview import ASReviewProject
from asreview.models.balance import DoubleBalance
from asreview.models.query import MaxQuery
from asreview.review import ReviewSimulate
from asreviewcontrib.models.classifiers import XGBoost as xgboost
from asreview.models.feature_extraction import Tfidf
from asreviewcontrib.insights.metrics import loss
from synergy_dataset import iter_datasets
from .utils import weighted_sum

def XGBoost(trial):
    """
    Objective function for optimizing the XGBoost classifier using Optuna.
    
    Parameters:
        trial: An Optuna trial object that suggests hyperparameters.
    
    Returns:
        final_loss: A weighted loss from all datasets.
    """
    accumulative_score = []
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", 
                                                         ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", 
                                                         ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", 
                                                            ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    
    model = xgboost(**param)
    feature_extractor= Tfidf()

    for d in iter_datasets():
        with tempfile.TemporaryDirectory() as tmpdir:

            project_temp_path = tmpdir + '/api_simulation/'

            project = ASReviewProject.create(
                project_path=project_temp_path,
                project_id="api_example",
                project_mode="simulate",
                project_name="api_example",
            )

            project_data_path = project_temp_path + 'data/' + d.name + 'csv'

            shutil.copy(f'data/{d.name}.csv' , project_data_path ) 

            project.add_dataset(project_data_path)

            reviewer = ReviewSimulate(
                as_data=ASReviewData.from_file(project_data_path),
                model=model,
                query_model=MaxQuery(),
                balance_model=DoubleBalance(),
                feature_model=feature_extractor,
                init_seed=535,
                n_instances=10,
                project=project,
                n_prior_included=1,
                n_prior_excluded=1,
            )

            project.update_review(status="review")
            reviewer.review()
            project.export(project_temp_path / 'output.asreview')
    
        loss_value = None
        with open_state(project_temp_path / 'output.asreview') as s:
            loss_value = loss(s)

        accumulative_score.append(loss_value)

    return weighted_sum(accumulative_score)