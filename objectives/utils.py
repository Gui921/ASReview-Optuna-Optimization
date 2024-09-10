import shutil
import tempfile

from asreview import open_state
from asreview import ASReviewData
from asreview import ASReviewProject
from asreview.models.balance import DoubleBalance
from asreview.models.query import MaxQuery
from asreview.review import ReviewSimulate
from asreviewcontrib.insights.metrics import loss
from synergy_dataset import iter_datasets

def weighted_sum(losses):
    weights = [1 / loss_value for loss_value in losses]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    weighted_loss = sum(w * l for w, l in zip(normalized_weights, losses)) # noqa E501
    return weighted_loss

def optimization_loop(model, feature_extractor):
    accumulative_score = []
    for d in iter_datasets():

        with tempfile.TemporaryDirectory() as tmpdir:

            project_temp_path = tmpdir + '/api_simulation/'
            project = ASReviewProject.create(
                project_path=project_temp_path,
                project_id="api_example",
                project_mode="simulate",
                project_name="api_example",
            )

            project_data_path = project_temp_path + 'data/' + d.name + '.csv'

            shutil.copy(f'data/{d.name}.csv' , project_data_path ) 

            project.add_dataset(project_data_path)

            reviewer = ReviewSimulate(
                as_data=ASReviewData.from_file(project_data_path),
                model=model,
                query_model=MaxQuery(),
                balance_model=DoubleBalance(),
                feature_model=feature_extractor,
                init_seed=535,
                n_instances=5,
                project=project,
                n_prior_included=1,
                n_prior_excluded=1,
                stop_if='min'
            )

            project.update_review(status="review")
            reviewer.review()
            project.export(project_temp_path + 'output.asreview')
            
            loss_value = None
            with open_state(project_temp_path + 'output.asreview') as s:
                loss_value = loss(s)

            accumulative_score.append(loss_value)
    return weighted_sum(accumulative_score)