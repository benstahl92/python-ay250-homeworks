# imports

# basics
import pickle as pkl
import pandas as pd
import numpy as np
from astropy.io import ascii

# system interaction
import os

# status bar
from tqdm import tqdm

# machine learning imports
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# custom classes and functions
from Spectrum import Spectrum
from ML_prep import ML_prep
import SNDB
from SNDB import Spectra, Objects

# login credentials of MySQL database
import db_params as dbp

def main(query = None, n_min = 50, n_bins = 1024, n_regions = 16, tet = (0.8, 0.2), norm = True, base_dir = dbp.base_dir, rs = 100):
    '''
    provides top level execution of final project
        retrieves spectral metadata (either from database query or from saved database query results)
        performs pre-processing steps on all spectra (or retrieves saved pre-processed data)
        featurizes pre-processed data for ingestion by ML models
        trains ML classifiers, computes success metrics, saves results to file

    Examples and doctests can be found in the docstrings of all functions and classes used by this function

    Parameters
    ----------
    query (optional, sqlalchemy query instance) : sqlalchemy query instance through which data is pulled from SNDB
                                                  (NB: if None, query results should be stored in checkpoints/query_results.pkl)
    n_min (optional, int) : minimum number of examples of a given subtype for it be included in classifier training
    n_bins (optional, int) : number of bins in new wavelength scale
    n_regions (optional, int) : number of regions to break each spectrum into for integrating
                                (NB: dividing this into n_bins from the Spectrum class should result in an integer)
    tet (optional, tuple) : 2 (or 3) element tuple containing the proportions to select for training, testing(, validation) 
    norm (optional, bool) : selects whether to normalize data based on training set
    base_dir (optional, str) : base path that all spectra filepaths are relative to
    rs (optional, int) : seed for random state

    Outputs
    -------
    'checkpoints/query_results.pkl' : pkl file containing query results (no need to duplicate db interactions if query doesn't change)
    'checkpoints/query.txt' : plain text SQL query corresponding to contents of 'checkpoints/query_results.pkl'
    'checkpoints/proc.npz' : file containing two arrays (preprocessed spectra and labels), so no need to re-preprocess on the same data
    'checkpoints/feat.npz' : file containing four arrays (X_train, y_train, X_test, y_test)
    'best_mod.pkl' : pkl file containing a dictionary with ML results (best models, baseline, X_scaler)
    '''

    # global filenames for storage
    query_res_fl = 'checkpoints/query_results.pkl'
    query_fl = 'checkpoints/query.txt'
    proc_fl = 'checkpoints/proc.npz'
    feat_fl = 'checkpoints/feat.npz'
    best_mod_fl = 'best_mod.pkl'

    print('\nWelcome to the Supernova Type Classifier Builder!\n')

    ######################################### data acquisition #########################################

    # if no query has been passed and a results file exists, read from that
    if (query is None) and os.path.isfile(query_res_fl):
        print('reading from query results file...')
        with open(query_res_fl, 'rb') as f:
            results = pkl.load(f)

    # otherwise, execute query against database, retrieve results, and write to disk for later use
    else:
        print('querying database...')
        results = query.all()
        with open(query_fl, 'w') as f:
            f.write(str(query))
        with open(query_res_fl, 'wb') as f:
            pkl.dump(results, f)
        print('done --- results written to {}'.format(query_res_fl))

    ######################################### data preprocessing #########################################

    # if no query has been passed and processed file exists, read from that
    if (query is None) and os.path.isfile(proc_fl):
        print('\nreading preprocessed spectra and labels from file...')
        data = np.load(proc_fl)
        pr_spectra = data['arr_0']
        labels = data['arr_1']

    # otherwise, load, preprocess, and store all spectra and labels from results
    else:
        print('\nloading and preprocessing {} spectra and labels...'.format(len(results)))
        pr_spectra = np.zeros((len(results), n_bins))
        labels = np.zeros(len(results), dtype=object)
        keep_ind = [] # to hold indices where process succeeds
        for idx, row in enumerate(tqdm(results)):
            st = row.Spectra
            ot = row.Objects
            try:
                s = Spectrum(ot.ObjName, st.SNID_Subtype, base_dir + st.Filepath + '/' + st.Filename, ot.Redshift_Gal)
            except ascii.InconsistentTableError:
                pass
            try:
                pr_spectra[idx, :] = s.preprocess(n_bins = n_bins)[:,1]
                labels[idx] = s.type
                keep_ind.append(idx)
            except ValueError:
                pass
        pr_spectra = pr_spectra[keep_ind, :]
        labels = labels[keep_ind]
        np.savez(proc_fl, pr_spectra, labels)
        print('done --- results written to {}'.format(proc_fl))

    # display summary statistics of sample
    s = pd.Series(labels)
    print('\ndistribution of types in selected sample:')
    print(s.value_counts().sort_index())

    # optionally remove classes with few than n_min examples
    if n_min is not None:
        uniqs, cnts = np.unique(labels, return_counts = True)
        to_remove = uniqs[cnts < n_min]
        if len(to_remove) > 0:
            print('\nremoving classes with fewer than {} occurences'.format(n_min))
            for rem in to_remove:
                rem_indices = np.where(labels == rem)
                pr_spectra = np.delete(pr_spectra, rem_indices, 0)
                labels = np.delete(labels, rem_indices, 0)

    ######################################### machine learning #########################################

    # extract features and split into test, evaluation, and training sets
    print('\nfeaturizing data and extracting training, validation, and testing sets with oversampling...')
    mlp = ML_prep(pr_spectra, labels, n_regions = n_regions)
    X_train, y_train, X_test, y_test = mlp.train_test_val_split(tet = tet)
    np.savez(feat_fl, X_train, y_train, X_test, y_test)
    print('done --- results written to {}'.format(feat_fl))

    # normalize based on training data (optionally)
    if norm:
        X_train = mlp.X_scaler.transform(X_train)
        X_test = mlp.X_scaler.transform(X_test)

    # compute baseline accuracy (assuming 'Ia-norm' is the most frequent)
    print('\ncomputing baseline accuracy for {} classes'.format(len(np.unique(y_train))))
    dc = DummyClassifier(strategy = 'constant', constant = 'Ia-norm')
    dc.fit(X_train, y_train)
    baseline = dc.score(X_test, y_test)
    print('baseline accuracy: {:.3f}'.format(baseline))
    
    # do a grid search with a k nearest neighbors algorithm and k fold cross-validation to identify the best hyper parameters
    est = KNeighborsClassifier()
    cv = KFold(n_splits = 6, random_state = rs)
    param_grid = {'n_neighbors': [3, 9, 15, 21], 'weights': ['uniform', 'distance'], 'leaf_size': [10, 15, 20, 25, 30]}
    print('\ncommencing KNN grid search over the following parameter grid:')
    print(param_grid)
    gs_knn = GridSearchCV(est, param_grid, n_jobs = -1, cv = cv)
    gs_knn.fit(X_train, y_train)
    print('done --- best parameters (score: {:.3f}):'.format(gs_knn.best_score_))
    print(gs_knn.best_params_)
    print('accuracy: {:.3f}'.format(gs_knn.score(X_test, y_test)))

    # do a grid search with a random forest algorithm and k fold cross-validation to identify the best hyper parameters
    est = RandomForestClassifier()
    param_grid = {'n_estimators': np.arange(25, 150, 25), 'max_depth': np.arange(2, 12, 2),
                  'min_samples_split': np.arange(2, 12, 2), 'max_features': np.arange(2, 32, 6), 'min_samples_leaf': np.arange(2, 12, 2)}
    print('\ncommencing Random Forest grid search over the following parameter grid:')
    print(param_grid)
    gs_rf = GridSearchCV(est, param_grid, n_jobs = -1, cv = cv)
    gs_rf.fit(X_train, y_train)
    print('done --- best parameters (score: {:.3f}):'.format(gs_rf.best_score_))
    print(gs_rf.best_params_)
    print('accuracy: {:.3f}'.format(gs_rf.score(X_test, y_test)))

    # save best model, X_scaler, and baseline for future reference
    best_mod = {'knn': gs_knn.best_estimator_, 'rf': gs_rf.best_estimator_, 'X_scaler': mlp.X_scaler, 'baseline': baseline}
    with open(best_mod_fl, 'wb') as f:
        pkl.dump(best_mod, f)
    print('\nbest model written to file: {}'.format(best_mod_fl))

# set query and run
if __name__ == "__main__":

    s = SNDB.get_session(dbp.usr, dbp.pswd, dbp.host, dbp.db)
    query = s.query(Spectra, Objects).filter(Spectra.ObjID == Objects.ObjID).filter(Objects.Redshift_Gal >= 0).filter(
          Spectra.SNID_Subtype != 'NULL').filter(Spectra.Min < 4500).filter(Spectra.Max > 7000).filter(
          ~Spectra.SNID_Subtype.like('%,%')).filter(Spectra.SNID_Subtype.like('I%'))
    main(query = query)
