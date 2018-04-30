# imports

# sql interaction
#import pymysql as sql

# basics
import pickle as pkl
import pandas as pd
import numpy as np

# system interaction
import os

# status bar
from tqdm import tqdm

# machine learning imports
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
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

def main(query = None, n_bins = 1024, n_regions = 16, tet = (0.8, 0.2), norm = True, base_dir = dbp.base_dir, rs = 100):
    '''
    Parameters
    ----------
    query : valid MySQL query
    query_res_fl : pkl file containing a dictionary of results yielded by a mysql_query function call

    Returns
    -------

    '''

    # global filenames for storage
    query_res_fl = 'query_results.pkl'
    query_fl = 'query.txt'
    proc_fl = 'proc.npz'
    feat_fl = 'feat.npz'
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
        #results = mysql_query(dbp.usr, dbp.pswd, dbp.db, query)
        with open(query_fl, 'w') as f:
            f.write(query)
        with open(query_res_fl, 'wb') as f:
            pkl.dump(results, f)
        print('done --- results written to {}'.format(query_res_fl))

    # display summary statistics of sample
    df = pd.DataFrame(results)
    print('\nDistribution of types in selected sample:')
    print(df['SNID_Subtype'].value_counts().sort_index())

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
        for idx, row in enumerate(tqdm(results)):
            st = row.Spectra
            ot = row.Objects
            s = Spectrum(ot.ObjName, st.SNID_subtype, base_dir + st.Filepath + '/' + st.Filename, ot.Redshift_Gal)
            #s = Spectrum(row['ObjName'], row['SNID_Subtype'], base_dir + row['Filepath'] + '/' + row['Filename'], row['Redshift_Gal'])
            pr_spectra[idx, :] = s.preprocess(n_bins = n_bins)[:,1]
            labels[idx] = s.type
        np.savez(proc_fl, pr_spectra, labels)
        print('done --- results written to {}'.format(proc_fl))

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
    print('\ncommencing grid search over the following parameter grid:')
    print(param_grid)
    gs = GridSearchCV(est, param_grid, n_jobs = -1, cv = cv)
    gs.fit(X_train, y_train)
    print('done --- best parameters (score: {:.3f}):'.format(gs.best_score_))
    print(gs.best_params_)
    print('accuracy: {:.3f}'.format(gs.score(X_test, y_test)))

    # save best model, X_scaler, and baseline for future reference
    best_mod = {'model': gs.best_estimator_, 'X_scaler': mlp.X_scaler, 'baseline': baseline}
    with open(best_mod_fl, 'wb') as f:
        pkl.dump(best_mod, f)
    print('\nbest model written to file: {}'.format(best_mod_fl))

# set query and run
if __name__ == "__main__":

    s = SNDB.get_session(dbp.usr, dbp.pswd, dbp.host, dbp.db)
    query = s.query(Spectra, Objects).filter(Spectra.ObjID == Objects.ObjID).filter(Objects.Redshift_Gal >= 0).filter(
        Spectra.SNID_Subtype != 'NULL').filter(Spectra.Min < 4500).filter(Spectra.Max > 7000).filter(~Spectra.SNID_Subtype.like('%,%')).limit(10)
    main(query = query)
