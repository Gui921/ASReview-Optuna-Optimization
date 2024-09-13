# Available Cases:
**To build a dataset:**

    python main.py build_dataset
**To optimize:**

    python main.py optimize
**To create the plots:**

    python main.py plot
**For help:**

    python main.py --help 
OR 

    python main.py *COMMAND* --help

---

## PLOT
    main.py plot

### SYNOPSIS
    main.py plot STUDY_NAME

Requires the *plotly* and *kaleido* libraries
### POSITIONAL ARGUMENTS
DB_STORAGE

---

## OPTIMIZE
    main.py optimize

### SYNOPSIS
    main.py optimize MODEL_NAME FEATURE_EXTRACTOR_NAME <flags>  

### POSITIONAL ARGUMENTS
    MODEL_NAME
        Name of the model to optimize
    FEATURE_EXTRACTOR_NAME
        Name of the feature extractor to be used in the simulations

### FLAGS
    -n, --n_trials=N_TRIALS
        Default: 10
        Number of trials of the optimization process
    -s, --study_name=STUDY_NAME
        Default: 'custom_study'
        Name given to the study
    -c, --cpu=CPU
        Default: 1
        Number of CPUs allocated for the task. 
        If --cpu = -1, then the maximum CPU availability will be used.

---
## Available Models and Feature Extractors
### Models:
* NaiveBayes
* NN_2_Layer
* XGBoost
* DynamicNN
* Logistic
* RandomForest
* SVM
* AdaBoost

### Feature Extractors:
* Tfidf
* SBERT
* Doc2Vec
* LaBSE
* MXBAI

---

# EXAMPLES:

### Build dataset:
    python main.py build_dataset

### Optimize Naive Bayes with a different name and with parallelization:
    python main.py optimize NaiveBayes Tfidf -s=example -c=-1
OR

    python main.py optimize NaiveBayes Tfidf --study_name=example --cpu=-1

### Optimize XGBoost with a 100 trials:
    python main.py optimize XGBoost Doc2vec-n=100
OR

    python main.py optimize XGBoost Doc2vec --n_trials=100

### Create plots:
    python main.py plot study_name
