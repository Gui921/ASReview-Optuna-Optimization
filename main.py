import sys
import os
import multiprocessing
import json
import xlsxwriter

from concurrent.futures import ProcessPoolExecutor
import optuna
import logging
from optuna.samplers import TPESampler
import fire
from progress.bar import Bar

from synergy_dataset import iter_datasets
from objectives import OBJECTIVES, FEATURE_EXTRACTORS

def build_dataset():
    '''
        Build and save Synergy datasets to the 'data' directory.

        Returns
        -------
        CSV files for each dataset are saved in the 'data' directory.
    '''

    if not(os.path.isdir('data')):
        os.mkdir('data')

    dataset_count = sum(1 for _ in iter_datasets())

    with Bar('Building dataset...', max=dataset_count) as bar:
        for d in iter_datasets():
            dataset_name = d.name
            if not(os.path.isdir(f'data/{dataset_name}')):
                os.mkdir(f'data/{dataset_name}')
            d = d.to_frame()
            dataset_path = f'data/{dataset_name}/{dataset_name}.csv'
            d.to_csv(dataset_path)
            bar.next()
    print("Synergy dataset is ready to be used.")

def plot(study_name):
    '''
        Create and save plots for the given study.
        
        Creates:
            - Slice Plot
            - Optimization History Plot
            - Parallel Coordinate Plot
            - Param Importances Plot

        Parameters
        ----------
        study_name: String
            Name of the study
        
        Returns
        -------
        A plots folder with the .png files from each plot
    '''

    storage = 'sqlite:///' + study_name + '/db.sqlite3'
    study_names = optuna.study.get_all_study_names(
        storage=storage)

    if not(os.path.isdir(study_name + '/plots')):
        os.mkdir(study_name + '/plots')
    with Bar('Creating plots...',max=len(study_names)*4) as bar:
        for name in study_names:
            study = optuna.study.load_study(study_name=name,
                                            storage=storage)

            fig = optuna.visualization.plot_slice(study)
            fig.update_layout(title_text=f'Slice Plot for study: {name}')
            fig.write_image(f"{study_name}/plots/{name}_slice_plot.png")
            bar.next()

            fig = optuna.visualization.plot_optimization_history(study)
            fig.update_layout(title_text=f'Optimization History Plot for study: {name}')
            fig.write_image(f'{study_name}/plots/{name}_optimization_history.png')
            bar.next()

            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.update_layout(title_text=f'Parallel Coordinate Plot for study: {name}')
            fig.write_image(f'{study_name}/plots/{name}_plot_parallel_coordinate.png')
            bar.next()

            fig = optuna.visualization.plot_param_importances(study)
            fig.update_layout(title_text=f'Param Importances Plot for study: {name}')
            fig.write_image(f'{study_name}/plots/{name}_plot_param_importances.png')
            bar.next()
    print(f"Plots were save in: {study_name}/plots")

def combine(folder_name):
    '''
    Function that combines all the info from the config files into an excel file.

    Parameters:
        folder_name: Folder with all the studies
    '''
    workbook = xlsxwriter.Workbook(f'{folder_name}/Combined Studies.xlsx')
    worksheet = workbook.add_worksheet()

    header_format = workbook.add_format({'bold': True, 'bg_color': 'cyan', 'align': 'center'})

    headers = ['Study Name', 'Model', 'Feature Extractor', 'Value', 'Hyperparameters']
    worksheet.set_column(0, 4, 20)

    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header, header_format)

    row = 1 

    for folder in os.listdir(folder_name):
        config_path = os.path.join(folder_name, folder, 'study_config.json')

        with open(config_path) as json_file:
            data = json.load(json_file)

            worksheet.write_string(row, 0, folder)
            worksheet.write_string(row, 1, data['Model'])
            worksheet.write_string(row, 2, data['Feature Extractor'])
            worksheet.write_number(row, 3, data['Value'])
            worksheet.write_string(row, 4, json.dumps(data['Best Hyperparameters']))
            row += 1

    table_range = f'A1:E{row}'

    worksheet.add_table(table_range, {
        'columns': [{'header': 'Study Name'},
                    {'header': 'Model'},
                    {'header': 'Feature Extractor'},
                    {'header': 'Value'},
                    {'header': 'Hyperparameters'}],
        'autofilter': True,  # Add autofilter to the table
    })

    workbook.close()

def optimize(model_name, feature_extractor_name, n_trials = 10, study_name = "custom_study", 
              cpu = 1):
    '''
    Function that optimizes a model given the set of parameters defined.

    Parameters
    ----------
    model_name : String
        Name of the model to optimize
    feature_extractor_name: String
        Name of the feature extractor to be used in the simulations
    n_trials : int, optional
        Number of trials of the optimization process
        Default: 10
    study_name : String, optional
        Name given to the study
        Default: "custom_study"
    cpu : int, optional
        Number of CPUs allocated for the task
        Default: 1
    '''

    objective = OBJECTIVES.get(model_name, None)
    feature_Extractor = FEATURE_EXTRACTORS.get(feature_extractor_name, None)

    if not objective:
        print(f"Model '{model_name}' is not supported. Available models: {', '.join(OBJECTIVES.keys())}.")# noqa E501
        sys.exit(1)
    if not feature_Extractor:
        print(f"Feature Extractor: '{feature_Extractor}' is not supported. Available models: {', '.join(FEATURE_EXTRACTORS.keys())}.")# noqa E501
        sys.exit(1)

    if cpu == -1:
        number_of_cpu = multiprocessing.cpu_count()
    else:
        number_of_cpu = cpu

    if not(os.path.isdir(study_name)):
        os.mkdir(study_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = 'sqlite:///' + study_name +'/db.sqlite3'

    study = optuna.create_study(direction='minimize',
                                study_name=study_name,
                                storage=storage_name, sampler=TPESampler(), 
                                load_if_exists=True)
    
    if number_of_cpu != 1 and n_trials>= number_of_cpu:
        with ProcessPoolExecutor(max_workers=number_of_cpu) as pool:
            for _ in range(number_of_cpu):
                pool.submit(study.optimize, objective(feature_Extractor), 
                        n_trials = n_trials // number_of_cpu)
                
    elif number_of_cpu != 1 and n_trials < number_of_cpu:
        with ProcessPoolExecutor(max_workers=n_trials) as pool:
            for _ in range(n_trials):
                pool.submit(study.optimize, objective(feature_Extractor), 
                        n_trials = 1)
    else:
        study.optimize(objective(feature_Extractor), n_trials=n_trials)

    best_trial = study.best_trial

    file_output = {
        'Model': model_name,
        'Feature Extractor': feature_extractor_name,
        'Value': best_trial.value,
        'Best Hyperparameters': best_hyp(best_trial)
    }
    json_object = json.dumps(file_output, indent=4)

    with open(study_name +'/study_config.json', 'w') as outfile:
        outfile.write(json_object)

def best_hyp(trial):
    dic = {}
    for key, value in trial.params.items():
        dic[key] = str(value)
    
    return dic

if __name__ == '__main__':
    fire.Fire({
        'optimize': optimize,
        'build_dataset': build_dataset,
        'plot': plot,
        'combine': combine
    })
    