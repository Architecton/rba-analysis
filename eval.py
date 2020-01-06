import numpy as np
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import scipy.io as sio
import itertools
import os

# Import RBAS.
from skrelief.relief import Relief
from skrelief.relieff import Relieff
from skrelief.surf import SURF
from skrelief.surfstar import SURFStar
from skrelief.multisurf import MultiSURF
from skrelief.multisurfstar import MultiSURFStar
from skrelief.reliefseq import ReliefSeq
from skrelief.reliefmss import ReliefMSS
from skrelief.swrfstar import SWRFStar
from skrelief.boostedsurf import BoostedSURF
from skrelief.turf import TuRF
from skrelief.vlsrelief import VLSRelief

# Import mass based dissimilarity distance metric function.
from skrelief.mbd.mbd import MBD

from select_by_rank import SelectByRank

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def get_ranks_matrix(data, target, rba, rba_param_grid, use_mbd, mbd_num_itrees_range):
    """Get matrix of feature rankings for each hyperparameter combination.
    
    Args:
        data (numpy.ndarray): matrix of data samples.
        target (numpy.ndarray): array of class values for samples.
        rba (object): rba being evluated.
        rba_param_grid (dict): paramater grid for rba being evaluated. 
        use_mbd (bool): parameter specifying whether mass based similarity is used.
        mbd_num_itrees_range (numpy.ndarray): Parameter values for the parameter specifying the number
        of itrees used for the mass-based dissimilarity.

    Returns:
        None: function returns None on successful completion.
    """

    # If using mass-based dissimilarity, add num_itree parameter search parameter to rba_param_grid.
    if use_mbd:
        rba_param_grid['num_itrees'] = mbd_num_itrees_range

    # Count number of cartesian product elements of parameter values.
    n_param_vals = 1
    for key in rba_param_grid.keys():
        n_param_vals *= len(rba_param_grid[key])
    
    # Initialize index for matrix of ranks.
    mat_idx = 0
    
    # Allocate matrix for storing ranks for each value of each parameter.
    rnk_mat = np.empty((n_param_vals, data.shape[1]), dtype=np.int)
    
    # Get names of parameters.
    param_names = list(rba_param_grid.keys())
    
    num = len(list(itertools.product(*rba_param_grid.values())))
    i = 0

    # Go over parameter combinations in grid.
    for param_vals in itertools.product(*rba_param_grid.values()):

        print("{0}/{1}".format(i, num))
        i += 1

        # Set parameter values of rba.
        for param_idx in np.arange(len(rba_param_grid)):
            if param_names[param_idx] == 'num_itrees':
                mbd = MBD(num_itrees=param_vals[param_idx])
                rba.dist_func = mbd.get_dist_func(data)
            else:
                setattr(rba, param_names[param_idx], param_vals[param_idx])
        
        # Get rank for next combination of parameters and save to ranks matrix.
        rnk_nxt = rba.fit(data, target).rank
        rnk_mat[mat_idx, :] = rnk_nxt
        mat_idx += 1
   
    # Return matrix of rankings for each parameter value.
    return rnk_mat



def run_eval(rba_name, dataset_name, rba, param_grid, param_grid_rba, data, target, use_mbd, mbd_num_itrees_range, results_folder_path):
    '''
    Run evaluation pipeline for single dataset.

    Args:
        rba_name (str): name of RBA being evaluated (used for naming results file).
        dataset_name (str): name of dataset on which the algorithm is evaluated (used for naming results file).
        rba (object): RBA instance.
        param_grid (dict): parameter grid used for hyperparameter optimization.
        data (numpy.ndarray): matrix of data samples.
        target (numpy.ndarray): array of class values for samples.
        use_mbd (bool): parameter specifying whether to use mass-based dissimilarity fistance metric.
        mbd_num_itrees_range (numpy.ndarray): array of values for the number of i-trees value parameter to check using grid search.
        results_folder_path (str): path for folder in which to save the results matrices.

    Returns:
        None: function returns None on successful completion.
    '''

    print("Evaluating {0} on dataset '{1}'.".format(rba_name, dataset_name))

    ### CONSTANTS ###

    # Define number of folds and runs in cross validation process.
    NUM_SPLITS = 10
    NUM_REPEATS = 10

    #################


    ### CLASSIFIER AND CV INITIALIZATIONS ###

    # Initialize classifier (KNN).
    clf = KNeighborsClassifier(n_neighbors=5)

    # Allocate vector for storing accuracies for each fold and fold counter that indexes it.
    cv_results = np.empty(NUM_SPLITS * NUM_REPEATS, dtype=np.float)
    idx_fold = 0

    # Initialize CV iterator.
    kf = RepeatedStratifiedKFold(n_splits=NUM_SPLITS, n_repeats=NUM_REPEATS, random_state=1)

    # Initialize cross validation strategy for hyperparameter tuning.
    cv_strategy = KFold(n_splits=3, random_state=1)

    #########################################


    # Perform cross-validation.
    for train_idx, test_idx in kf.split(data, target):
        
        # Split data into training set, validation set and test set
        data_train = data[train_idx]
        target_train = target[train_idx]
        data_test = data[test_idx]
        target_test = target[test_idx]
        data_train, data_val, target_train, target_val = train_test_split(data_train, target_train, test_size=0.3, random_state=1)
        
        # NOTE:
        # training set -- data_train, target_train
        # validation set -- data_val, target_val
        # test set -- data_test, target_test


        ### HYPERPARAMETER OPTIMIZATION ###
        
        # Fit learner to validation dataset and initialize dummy learner
        # for n_features_to_select parameter optimization.
        rnk_rba = get_ranks_matrix(data_val, target_val, rba, param_grid_rba, use_mbd, mbd_num_itrees_range)

        sbr = SelectByRank(rank=rnk_rba)

        # Perform hyperparameter optimization on validation set.
        clf_pipeline = Pipeline([('scaling', StandardScaler()), ('sbr', sbr), ('clf', clf)])
        gs = GridSearchCV(clf_pipeline, param_grid=param_grid, cv=cv_strategy, verbose=True, iid=False, n_jobs=-1)
        gs.fit(data_val, target_val)

        ###################################


        # Train model on training set using optimized hyperparameter values.
        trained_model = gs.best_estimator_.fit(data_train, target_train)


        ### ACCURACY TESTING ###

        # Compute classification accuracy on test set and store in results vector.
        res = trained_model.predict(data_test)
        fold_score = accuracy_score(target_test, res)
        cv_results[idx_fold] = fold_score
        idx_fold += 1

        ########################

        print("finished fold {0}/{1}".format(idx_fold-1, NUM_SPLITS*NUM_REPEATS))
    
    # Save vector of cv results to file.
    if use_mbd:
        if rba_name == 'VLSRelief' or rba_name == 'TuRF':
            sio.savemat(results_folder_path + dataset_name + '_' + rba_name + '-' + rba._rba.name + '-mbd.mat', {'res' : cv_results})
        else:
            sio.savemat(results_folder_path + dataset_name + '_' + rba_name + '-mbd.mat', {'res' : cv_results})
    else:
        if rba_name == 'VLSRelief' or rba_name == 'TuRF':
            sio.savemat(results_folder_path + dataset_name + '_' + rba_name + '-' + rba._rba.name + '.mat', {'res' : cv_results})
        else:
            sio.savemat(results_folder_path + dataset_name + '_' + rba_name + '.mat', {'res' : cv_results})


if __name__ == '__main__':

    # Set path to datasets folder.
    DATASETS_FOLDER_PATH = './datasets/eval-set/'
    RESULTS_FOLDER_PATH = './results-matrices/'
    
    # Parse dataset.
    datasets = dict()
    dataset_names = list(os.listdir(DATASETS_FOLDER_PATH))
    for dataset_name in dataset_names:
        data = sio.loadmat(DATASETS_FOLDER_PATH + dataset_name + '/data.mat')['data']
        target = np.ravel(sio.loadmat(DATASETS_FOLDER_PATH + dataset_name + '/target.mat')['target'])
        datasets[dataset_name] = (data, target) 
    
    # set of algorithms to evaluate -- PUT NAMES OF ALGORITHMS TO COMPARE HERE.
    eval_set = {"SWRFStar"}
    
    # Algorithm wrapped by TuRF and VLSRelief
    wrapped_rba = SWRFStar()
    wrapped_rba.name = "SWRFStar"

    # Parameters for MBD.
    use_mbd = True
    mbd_num_itrees_range = range(30, 31)
    
    # Go over datasets and compute results.
    for dataset_name in datasets.keys():

        # Get data samples and target variable values of next dataset.
        data, target = datasets[dataset_name]

        # Get array of values for the parameter controlling the number of features to select
        # parameter for grid search.
        n_features_to_select = np.arange(1, min(data.shape[1], 600)+1)

        # Define dictionary that rba instances and their parameter grids.
        rba_data = {
                'Relief' : {'rba' : Relief(), 'param_grid' : {
                        'sbr__n_features_to_select': np.arange(1, 600), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'ReliefF_k_nearest' : {'rba' : Relieff(k=10, mode="k_nearest"), 'param_grid' : {
                    'sbr__n_features_to_select': n_features_to_select, 'sbr__row': np.arange(1), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'k' : range(10, 11)
                        }},
                'ReliefF_diff' : {'rba' : Relieff(k=10, mode="diff"), 'param_grid' : {
                    'sbr__n_features_to_select': n_features_to_select, 'sbr__row': np.arange(1), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'k' : range(10, 11)
                        }},
                'ReliefF_exp_rank' : {'rba' : Relieff(k=10, mode="exp_rank"), 'param_grid' : {
                    'sbr__n_features_to_select': n_features_to_select, 'sbr__row': np.arange(1), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'k' : range(10, 11)
                        }},
                'ReliefSeq' : {'rba' : ReliefSeq(k_min=5, k_max=15), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'SURF' : {'rba' : SURF(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'SURFStar' : {'rba' : SURFStar(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'MultiSURFStar' : {'rba' : MultiSURFStar(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'MultiSURF' : {'rba' : MultiSURF(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'ReliefMSS' : {'rba' : ReliefMSS(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'sbr__row': np.arange(1), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'k' : range(10, 11)
                        }},
                'SWRFStar' : {'rba' : SWRFStar(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {}},
                'BoostedSURF' : {'rba' : BoostedSURF(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'sbr__row': np.arange(2), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'phi' : range(13, 15+1)
                        }},
                'TuRF' : {'rba' : TuRF(), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'sbr__row': np.arange(20), 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'num_it' : range(10, 100)
                        }},
                'VLSRelief' : {'rba' : VLSRelief(wrapped_rba), 'param_grid' : {
                        'sbr__n_features_to_select': n_features_to_select, 'clf__n_neighbors' : np.arange(1, 10)
                        }, 'param_grid_rba' : {
                            'num_partitions_to_select' : range(10, 15), 'num_subsets' : range(50, 55), 'partition_size' : range(10, 15)
                        }},

                }
        
        # Go over RBAs and evaluate on dataset.
        for rba_name in rba_data.keys():

            # If next RBA in set of RBAs to be evaluated.
            if rba_name in eval_set:

                # Get RBA evaluation data and perform evaluation.
                data_nxt = rba_data[rba_name]
                run_eval(rba_name, dataset_name, data_nxt['rba'], data_nxt['param_grid'], data_nxt['param_grid_rba'], data, target, use_mbd, mbd_num_itrees_range, RESULTS_FOLDER_PATH)

