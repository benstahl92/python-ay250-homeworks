# imports

# sql interaction
import pymysql as sql

# basics
import pickle as pkl
import pandas as pd
import numpy as np

# system interaction
import os

# status bar
from tqdm import tqdm

# machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# custom classes and functions
from Spectrum import Spectrum
from ML_prep import ML_prep

# login credentials of MySQL database
import db_params as dbp

def mysql_query(usr, pswd, db, query):
    '''
    execute a mysql query on a given database and return the result as a list containing all retrieved rows as dictionaries

    Parameters
    ----------
    usr : user name for database access (typically accessed from a param file, see example)
    pswd : password for given user name (typically accessed from a param file, see example)
    db : database name (typically accessed from a param file, see example)
    query : valid MySQL query

    Returns
    -------
    results : list of all retrieved results (each as a dictionary)

    Doctests/Examples
    -----------------
    # NB: these tests will fail unless usr, pswd, and db are the valid credentials for my research group's database (supplied by the import, but must be run on the appropriate computer)
    >>> query = "SELECT t1.ObjName FROM objects as t1, spectra as t2 WHERE (t1.ObjID = t2.ObjID) AND (t2.UT_Date > 19850101) AND (t2.UT_Date < 20180101) AND (t2.Min < 4500) and (t2.Max > 7000) AND (t2.Filename NOT LIKE '%gal%');"
    >>> result = mysql_query(dbp.usr, dbp.pswd, dbp.db, query)
    >>> len(result)
    6979
    >>> type(result)
    <class 'list'>
    >>> type(result[0])
    <class 'dict'>

    >>> query2 = "SELECT ObjName FROM objects LIMIT 10;"
    >>> result2 = mysql_query(dbp.usr, dbp.pswd, dbp.db, query2)
    >>> len(result2)
    10
    >>> result2[1]['ObjName']
    'SN 1954A'

    # one more test on a realistic query here: make sure that keys are what is expected, etc.
    '''

    # connect to db
    connection = sql.connect(user = usr, password = pswd, db = db, cursorclass = sql.cursors.DictCursor)

    # issue query and get results
    with connection.cursor() as cursor:
        cursor.execute(query)
        results = cursor

    # return results
    return list(results)

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
        results = mysql_query(dbp.usr, dbp.pswd, dbp.db, query)
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
            s = Spectrum(row['ObjName'], row['SNID_Subtype'], base_dir + row['Filepath'] + '/' + row['Filename'], row['Redshift_Gal'])
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

"""
example query

SELECT t1.ObjName, t2.Filename, t2.Filepath, t1.Redshift_Gal, t2.SNID_Subtype FROM objects as t1, spectra as t2 WHERE (t1.ObjID = t2.ObjID) AND (t1.Redshift_Gal != 'NULL') AND (t2.SNID_Subtype LIKE 'Ia%') AND (t2.UT_Date > 20090101) AND (t2.UT_Date < 20180101) AND (t2.Min < 4500) and (t2.Max > 7000) AND (t2.Filename NOT LIKE '%gal%') AND (t1.DiscDate > DATE('2008-01-01'));
"""

# do tests
#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()
