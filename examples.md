# OPTIONS:

python main.py build_dataset

python main.py optimize

python main.py plot

python main.py --help **OR** python main.py *COMMAND* --help

# PLOT
    main.py plot

## SYNOPSIS
    main.py plot *DB_STORAGE*

In the *DB_STORAGE* do not forget to add the full path without the file extension! (ex.: study_folder/db_file)
REQUIRES PLOTLY AND KALEIDO
## POSITIONAL ARGUMENTS
DB_STORAGE

# OPTIMIZE
    main.py optimize

## SYNOPSIS
    main.py optimize MODEL_NAME <flags>  

## POSITIONAL ARGUMENTS
    MODEL_NAME

## FLAGS
    -n, --n_trials=N_TRIALS
        Default: 10
    --study_name=STUDY_NAME
        Default: 'custom_study'
    -p, --parallel=PARALLEL
        Default: False

# EXAMPLES:

### Build dataset:
    python main.py build_dataset

### Optimize Naive Bayes with a different name and with parallelization:
    python main.py optimize NaiveBayes --study_name=example -p=True

### Optimize Naive Bayes with a 100 trials:
    python main.py optimize NaiveBayes -n=100
OR

    python main.py optimize NaiveBayes --n_trials=100

### Create plots:
    python main.py plot study_folder/dbfile
