# standard imports
import pymysql as sql
import pickle as pkl
import pandas as pd
import numpy as np
import os
from astropy.io import ascii
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt
from scipy.integrate import simps
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

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

class Spectrum:
    '''
    base class for spectra that provides a self-contained pathway from metadata to various data products
        class contains methods for direct use by class instances and functions that can be exposed outside of a specific instance
        for examples/doctests see docstrings for the methods and functions of the class

    General Concepts
    ----------------
    spec : 2d array of (column-wise) wavelength, flux [spec should be assumed to be in this form unless explicitly noted as otherwise]

    Methods
    -------
    __init__ : instantiation instructions
    preprocess : performs preprocessing procedure on spectrum

    Functions
    ---------
    read_file : reads (and returns) spectrum from provided file (with relative path)
    dez : de-redshifts (and returns) given spectrum
    sp_medfilt : wrapper around scipy medfilt function
    log_bin : bins spectrum to logarithmic scale and returns re-binned spectrum
    cont_subtr : fits cubic spline to subset of points in given spectrum and subtracts it to remove a pseudo continuum
    apodize : tapers edges of spectrum using a Hanning window
    flux_norm : normalizes flux (f) according to the algorithm: (f - f_min) / (f_max - f_min) - mean(qty on left)
    '''

    def __init__(self, sn_name, sn_subtype, spec_filename, redshift, spec = None):
        '''
        instantiation instructions

        Parameters
        ----------
        (object instance)
        sn_name : name of supernova
        sn_subtype : subtype of supernova (i.e. Ia-norm)
        spec_filename : filename (with relative path) of SN spectrum (file should have two columns: wavelength flux)
        redshift : redshift of SN
        spec (optional, 2d array) : 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> s.name
        'SN 1997y'
        >>> s.z
        0.01587
        >>> s.spec.shape
        (3361, 2)
        '''

        # require redshift to be positive
        assert redshift >= 0

        self.name = sn_name
        self.type = sn_subtype
        self.file = spec_filename
        self.z = redshift

        # attempt to make sure spec is in correct format, and read in otherwise
        if type(spec) != np.ndarray:
            self.spec = Spectrum.read_file(spec_filename)
        elif spec.shape[1] != 2:
            self.spec = Spectrum.read_file(spec_filename)
        else:
            self.spec = spec

    def read_file(file):
        '''
        reads (and returns) spectrum from provided file (with relative path)

        Parameters
        ----------
        file : filename (with relative path) of supernova spectrum (file should have two columns: wavelength flux)

        Returns
        -------
        spec : raw spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> Spectrum.read_file('sn1997y-19970209-uohp.flm').shape
        (3361, 2)
        '''
        
        tmp = ascii.read(file)
        cols = tmp.colnames
        return np.array([tmp[cols[0]], tmp[cols[1]]]).T

    def dez(spec, z):
        '''
        de-redshifts (and returns) given spectrum

        Parameters
        ----------
        spec : spectrum as 2d array of (column-wise) wavelength, flux
        z : redshift of SN

        Returns
        -------
        spec : de-redshifted spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> dz = Spectrum.dez(s.spec, s.z)
        >>> dz[:,0].max() <= s.spec[:,0].max()
        True
        >>> (dz[:,1] == s.spec[:,1]).all()
        True
        '''

        # require redshift to be positive
        assert z >= 0

        spec = spec
        wav = spec[:,0] / (1 + z)
        return np.array([wav, spec[:,1]]).T

    def sp_medfilt(spec, ksize = 15):
        '''
        wrapper around scipy medfilt function
            applies a median filter (and returns) spectrum

        Parameters
        ----------
        spec : spectrum as 2d array of (column-wise) wavelength, flux
        ksize (optional, odd int) : size of median filter window

        Returns
        -------
        spec : median-filtered spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> mf = Spectrum.sp_medfilt(s.spec, ksize=45)
        >>> mf[:,1].std() < s.spec[:,1].std()
        True
        >>> (mf[:,0] == s.spec[:,0]).all()
        True
        '''

        flux = medfilt(spec[:,1], ksize)
        return np.array([spec[:,0], flux]).T

    def log_bin(spec, wav_min = 3100, wav_max = 9600, n_bins = 1024):
        '''
        bins spectrum to logarithmic scale and returns re-binned spectrum
            flux re-binning utilizes a cubic spline interpolation

        Parameters
        ----------
        spec : spectrum as 2d array of (column-wise) wavelength, flux
        wav_min (optional, int) : minimum of new wavelength scale
        wav_max (optional, int) : maximum of new wavelength scale
        n_bins (optional, int) : number of bins in new wavelength scale

        Returns
        -------
        spec : re-binned spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> lb = Spectrum.log_bin(s.spec, n_bins = 1024)
        >>> lb.shape
        (1024, 2)
        '''

        # compute new wavelength array
        dl = np.log(wav_max / wav_min) / n_bins
        n = np.arange(0.5, n_bins + 0.5)
        log_wav = wav_min * np.exp(n * dl)

        # find flux at new wavelength points (set flux to zero for wavelengths outside original range)
        wav = spec[:,0]
        spl = CubicSpline(wav, spec[:,1])
        flux = spl(log_wav)
        flux[np.logical_or(log_wav < wav.min(), log_wav > wav.max())] = 0
        return np.array([log_wav, flux]).T

    def cont_subtr(spec, n_spline_pts = 13, ret_cont = False):
        '''
        fits cubic spline to subset of points in given spectrum and subtracts it to remove a pseudo continuum

        Parameters
        ----------
        spec : spectrum as 2d array of (column-wise) wavelength, flux
        n_spline_pts (optional, int) : number of (evenly spaced) points to use for spline fit to spectrum
        ret_cont (optional, bool) : returns fitted spline with corrected spectrum if True

        Returns
        -------
        spec : continuum-subtracted spectrum as 2d array of (column-wise) wavelength, flux
        spl (if ret_cont is True) : fitted spline

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> Spectrum.cont_subtr(s.spec).shape[1]
        2
        '''

        # compute indices corresponding to points to fit spline
        spl_indices = np.linspace(0, spec.shape[0] - 1, n_spline_pts, dtype=int)
        
        # fit spline and continuum subtract
        spl = UnivariateSpline(spec[spl_indices,0], spec[spl_indices,1])
        wav = spec[:,0]
        flux = spec[:,1] - spl(wav)
        if ret_cont is True:
            return np.array([wav, flux]).T, spl
        else:
            return np.array([wav, flux]).T

    def apodize(spec, end_pct = 0.05):
        '''
        tapers edges of spectrum using a Hanning window

        Parameters
        ----------
        spec : spectrum as 2d array of (column-wise) wavelength, flux
        end_pct (optional, float between 0 and 1) : percentage of each end of spectrum to taper

        Returns
        -------
        spec : apodized spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> a = Spectrum.apodize(s.spec, end_pct = 0.05)
        >>> np.abs(a[0,1]) < np.abs(s.spec[0,1])
        True
        >>> np.abs(a[-1,1]) < np.abs(s.spec[-1,1])
        True
        >>> a[int(a.shape[0]/2),1] == s.spec[int(a.shape[0]/2), 1]
        True
        '''

        size = int(end_pct * spec.shape[0])
        h = np.hanning(2 * size)

        # form full window by wrapping tapered edges around un-tapered body
        window = np.concatenate((h[:size], np.ones(spec.shape[0] - 2 * size), h[-size:]))

        flux = spec[:,1] * window
        return np.array([spec[:,0], flux]).T

    def flux_norm(spec):
        '''
        normalizes flux (f) according to the algorithm: (f - f_min) / (f_max - f_min) - mean(qty on left)

        Parameters
        ----------
        spec : spectrum as 2d array of (column-wise) wavelength, flux

        Returns
        -------
        spec : normalized spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> fn = Spectrum.flux_norm(s.spec)
        >>> np.abs(fn[:,1].mean()) < 0.001 # tolerance on being close enough to zero
        True
        >>> fn[:,1].max() < 1
        True
        >>> fn[:,1].min() > -1
        True
        '''

        f = spec[:,1]
        nflux = (f - f.min()) / (f.max() - f.min())
        return np.array([spec[:,0], nflux - nflux.mean()]).T

    def preprocess(self, ksize = 15, n_spline_pts = 13, wav_min = 3100, wav_max = 9600, n_bins = 1024, end_pct = 0.05):
        '''
        performs preprocessing procedure on instance's spectrum:
            de-redshifts, median filters, removes pseudo continuum, normalizes, apodizes edges, bins to logarithmic wavelength bins

        Parameters
        ----------
        (object instance)
        ksize (optional, odd int) : size of median filter window
        n_spline_pts (optional, int) : number of (evenly spaced) points to use for spline fit to spectrum
        wav_min (optional, int) : minimum of new wavelength scale
        wav_max (optional, int) : maximum of new wavelength scale
        n_bins (optional, int) : number of bins in new wavelength scale
        end_pct (optional, float between 0 and 1) : percentage of each end of spectrum to taper

        Returns
        -------
        spec : fully preprocessed spectrum as 2d array of (column-wise) wavelength, flux

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> p = s.preprocess(n_bins = 1024)
        >>> p.shape
        (1024, 2)
        >>> p[:,1].max() < 1
        True
        >>> p[:,1].min() > -1
        True
        '''

        # de-redshift
        spec = Spectrum.dez(self.spec, self.z)

        # do median filtering
        spec = Spectrum.sp_medfilt(spec, ksize = ksize)

        # remove pseudo continuum
        spec = Spectrum.cont_subtr(spec, n_spline_pts = n_spline_pts)

        # normalize
        spec = Spectrum.flux_norm(spec)

        # apodize
        spec = Spectrum.apodize(spec, end_pct = end_pct)

        # bin to logarithmic wavelength bins
        spec = Spectrum.log_bin(spec, wav_min = wav_min, wav_max = wav_max, n_bins = n_bins)

        return spec

class ML_prep:
    '''
    base class for organizing and preparing data for machine learning processes
        class contains methods for direct use by class instances and functions that can be exposed outside of a specific instance
        for examples/doctests see docstrings for the methods and functions of the class

    General Concepts
    ----------------
    takes spectra that have been preprocessed by the Spectrum class and prepares them for ingestion by a ML model
    features are the integrated areas of different regions of each spectrum

    Methods
    -------
    __init__ : instantiation instructions
    featurize : featurizes spectra for ingestion by ML models
    proc_labels : do any needed processing on labels before ingestion by ML model (currently trivial)
    train_val_test_split : splits featurized data and labels into training, evaluation, and testing sets

    Functions
    ---------
    integ_reg_area : breaks each spectrum into n_regions and calculates (and returns) the integrated area of each region
    os_balance : makes number of occurrences of each label (and associated features) the same by oversampling
    '''

    def __init__(self, spectra, labels, n_regions = 16, rs = 100):
        '''
        instantiation instructions

        Parameters
        ----------
        (object instance)
        spectra : 2d array where each row corresponds to the flux of a given spectrum
        labels : array of subtypes corresponding to the SNe in the rows of spectra
        n_regions (optional, int) : number of regions to break each spectrum into for integrating
                                    (NB: dividing this into n_bins from the Spectrum class should result in an integer)
        rs : random state

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> mlp = ML_prep(s.preprocess()[:,1].reshape(1,-1), np.array([s.type]), n_regions = 16)
        >>> len(mlp.osi) >= len(mlp.labels)
        True
        >>> mlp.X.shape
        (1, 16)
        '''

        # full (un-prepared) sets
        self.spectra = spectra
        self.labels = labels

        # full (featurized but not oversampled) sets
        self.X = self.featurize(n_regions = n_regions)
        self.y = self.proc_labels()

        # X scaler to be fitted to training data
        self.X_scaler = StandardScaler()

        np.random.seed(rs)

    def integ_reg_area(spectra, n_regions = 16):
        '''
        breaks each spectrum into n_regions and calculates (and returns) the integrated area of each region

        Parameters
        ----------
        spectra : 2d array where each row corresponds to the flux of a given spectrum
        n_regions (optional, int) : number of regions to break each spectrum into for integrating
                                    (NB: dividing this into n_bins from the Spectrum class should result in an integer)

        Returns
        -------
        2d array where each row corresponds to the integrated area of each region of a given spectrum
        '''

        # split spectra into n_regions then integrate each of those regions
        regions = np.split(spectra, n_regions, axis = 1)
        return simps(regions, axis = 2).T

    def featurize(self, n_regions = 16):
        '''
        featurizes spectra for ingestion by ML models
            features are integrated areas of regions of each spectrum

        Parameters
        ----------
        (object instance)
        n_regions (optional, int) : number of regions to break each spectrum into for integrating
                                    (NB: dividing this into n_bins from the Spectrum class should result in an integer)

        Returns
        -------
        features : 2d array where each row corresponds features of a given spectrum
        '''

        # create container for features and then populate
        features = np.zeros((self.spectra.shape[0], n_regions))
        features[:, :n_regions] = ML_prep.integ_reg_area(self.spectra, n_regions = n_regions)
        return features

    def proc_labels(self):
        '''
        do any needed processing on labels before ingestion by ML model (currently trivial)

        Parameters
        ----------
        (object instance)

        Returns
        -------
        unmodified labels (trivial, but in the future only this function would need to be modified to adjust this)
        '''

        return self.labels

    def os_balance(labels):
        '''
        makes number of occurrences of each label (and associated features) the same by oversampling

        Parameters
        ----------
        labels : array of subtypes corresponding to the SNe in the rows of spectra

        Returns
        -------
        osi : (shuffled) array of indices corresponding to labels that have been oversampled to match the most common label
        '''

        # get unique labels and their counts
        uniques, counts = np.unique(labels, return_counts = True)

        # get index of max count and max count
        mci = np.argmax(counts)
        mc = counts[mci]

        # create array to hold oversample indices
        osi = np.zeros(mc * len(uniques), dtype = int)

        # iterate through labels, oversampling to match the label with the max count
        li = np.arange(len(labels)) # label index array
        for idx, cnt in enumerate(counts):
            if cnt == mc:
                osi[(idx * mc):((idx + 1) * mc)] = li[labels == uniques[idx]]
            elif cnt < mc:
                sample_indices = li[labels == uniques[idx]]
                osi[(idx * mc):((idx + 1) * mc)] = np.random.choice(sample_indices, size = mc) # with replacement
            elif cnt > mc:
                raise ValueError('label {} has max count ({}) greater than label with calculated max count ({})'.format(
                                  uniques[idx], counts[idx], mc))

        # return (un-shuffled) oversample indices
        return osi

    def train_test_val_split(self, tet = (0.6, 0.2, 0.2), os_train = True):
        '''
        splits featurized data and labels into shuffled training, testing(, validation) sets and fits (but does not apply) normalization to test set
            normalization gets stored in self.X_scaler
            training data are oversampled to uniformity but testing(, validation) sets are not (if os_train is True, otherwise nothing done)

        Parameters
        ----------
        (object instance)
        tet (optional, tuple) : 2 (or 3) element tuple containing the proportions to select for training, testing(, validation) 

        Returns
        -------
        splitting : list of length 4 (or 6) containing the splits of training, testing(, validation) data (each split is X, y)
        '''

        assert np.sum(tet) == 1

        # get indices to shuffle data
        dlen = len(self.y)
        shuff_ind = np.random.choice(np.arange(dlen, dtype = int), size = dlen, replace = False)

        # construct training data (with oversampling if requested)
        y_train = self.y[shuff_ind[:int(tet[0]*dlen)]]
        if os_train:
            train_ind = ML_prep.os_balance(y_train)
            X_train = self.X[train_ind]
            y_train = self.y[train_ind]
        elif not os_train:
            X_train = self.X[suff_ind[:int(tet[0]*dlen)],:]
        self.X_scaler.fit(X_train)

        # construct testing(, validation) sets
        X_test = self.X[shuff_ind[int(tet[0]*dlen):int((tet[0]+tet[1])*dlen)],:]
        y_test = self.y[shuff_ind[int(tet[0]*dlen):int((tet[0]+tet[1])*dlen)]]
        if len(tet) == 3:
            X_val = self.X[shuf_ind[-int(tet[2]*dlen):],:]
            y_val = self.y[shuf_ind[-int(tet[2]*dlen):]]
            return X_train, y_train, X_test, y_test, X_val, y_val
        elif len(tet) == 2:
            return X_train, y_train, X_test, y_test
        else:
            raise ValueError('tet is not length 2 or 3')

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

    print('Welcome to the Supernova Type Classifier Builder!\n')

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

    ######################################### data preprocessing #########################################

    # could add skip for the below if the file exists...

    # load, preprocess, and store all spectra and labels from results
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

    # compute baseline accuracy (due to oversampling, expect this to be close to 1/(number of classes))
    print('\ncomputing baseline accuracy for {} classes'.format(len(np.unique(y_train))))
    dc = DummyClassifier(strategy = 'prior')
    dc.fit(X_train, y_train)
    baseline = dc.score(X_test, y_test)
    print('baseline accuracy: {:.3f}'.format(baseline))

    # do a grid search with a k nearest neighbors algorithm and k fold cross-validation to identify the best hyper parameters
    est = KNeighborsClassifier()
    cv = KFold(n_splits = 6, random_state = rs)
    param_grid = {'n_neighbors': [3, 5, 10, 15], 'weights': ['uniform', 'distance'], 'leaf_size': [20, 30, 40]}
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
