import sys
import os
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
import optuna
import logging
from optuna.samplers import TPESampler
import fire
from progress.bar import Bar

from synergy_dataset import iter_datasets
from objectives import OBJECTIVES

def build_dataset():
    '''
        Build and save Synergy datasets to the 'data' directory.

        Returns
        -------
        CSV files for each dataset are saved in the 'data' directory.
    '''
    if not(os.path.isdir('data')):
        os.mkdir('data')

    with Bar('Building dataset...', max=26) as bar:
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

def optimize(model_name, n_trials = 10, study_name = "custom_study", 
              cpu = 1):
    '''
    Function that optimizes a model given the set of parameters defined.

    Parameters
    ----------
    model_name : String
        Name of the model to optimize
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

    if not objective:
        print(f"Model '{model_name}' is not supported. Available models: {', '.join(OBJECTIVES.keys())}.")# noqa E501
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
    if number_of_cpu != 1:
        with ProcessPoolExecutor(max_workers=number_of_cpu) as pool:
            for _ in range(number_of_cpu):
                pool.submit(study.optimize, objective, 
                            n_trials = n_trials // number_of_cpu)
    else:
        study.optimize(objective, n_trials=n_trials)



if __name__ == '__main__':
    fire.Fire({
        'optimize': optimize,
        'build_dataset': build_dataset,
        'plot': plot
    })
    