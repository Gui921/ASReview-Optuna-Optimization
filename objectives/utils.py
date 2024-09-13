import shutil
import tempfile
import os

from scipy.sparse import save_npz, load_npz

from asreview import open_state
from asreview import ASReviewData
from asreview import ASReviewProject
from asreview.models.balance import DoubleBalance
from asreview.models.query import MaxQuery
from asreview.review import ReviewSimulate
from asreviewcontrib.insights.metrics import loss
from synergy_dataset import iter_datasets

def weighted_sum(losses):
    '''
    Function that calculates the final loss that is to be optimized by penalizing big losses and
    favouring small losses.

    Params:
        losses: List of all the losses
    
    Returns:
        weighted_loss: Weighed Loss from all the losses
    '''
    weights = [1 / loss_value for loss_value in losses]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    weighted_loss = sum(w * l for w, l in zip(normalized_weights, losses)) # noqa E501
    return weighted_loss

def optimization_loop(model, feature_extractor):
    '''
    Main loop of the optimization process where it loops through all the datasets and 
    computes the final loss

    Params:
        model: Model to be optimized
        feature_extractor: Feature Extractor to be used in the simulation
    Returns:
        Final loss
    '''
    accumulative_score = []
    for d in iter_datasets():

        with tempfile.TemporaryDirectory() as tmpdir:

            project_temp_path = tmpdir + '/api_simulation/'
            feature_matrix_path = f'data/{d.name}/{feature_extractor.name}_feature_matrix.npz'
            project = ASReviewProject.create(
                project_path=project_temp_path,
                project_id="api_example",
                project_mode="simulate",
                project_name="api_example",
            )
            if os.path.isdir(feature_matrix_path):
                feature_matrix = load_npz(feature_matrix_path)
                project.add_feature_matrix(feature_matrix, feature_extractor.name)
            project_data_path = project_temp_path + 'data/' + d.name + '.csv'

            shutil.copy(f'data/{d.name}/{d.name}.csv' , project_data_path ) 

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
            if not os.path.isdir(feature_matrix_path):
                feature_matrix = project.get_feature_matrix(feature_extractor.name)
                save_npz(feature_matrix_path,feature_matrix)
            project.export(project_temp_path + 'output.asreview')
            
            loss_value = None
            with open_state(project_temp_path + 'output.asreview') as s:
                loss_value = loss(s)

            accumulative_score.append(loss_value)
    delete_tmp_files()
    return weighted_sum(accumulative_score)

def delete_tmp_files():
    ''''
    Function to delete all the temporary files
    '''
    tmp_path = tempfile.gettempdir()

    for folder in os.listdir(tmp_path):
        folder_path = os.path.join(tmp_path, folder)

        if folder.startswith('tmp'):
            shutil.rmtree(folder_path)