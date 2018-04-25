# standard imports
import pymysql as sql
import pickle as pkl
import pandas as pd
import numpy as np
from astropy.io import ascii
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt

# login credentials of MySQL database
#import db_params as dbp



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

def sp_medfilt(spec, ksize = 15):
    '''
    wrapper around scipy medfilt function
        applies a median filter (and returns) spectrum
    '''

    flux = medfilt(spec[:,1], ksize)
    return np.array([spec[:,0], flux]).T

def log_bin(spec, wav_min = 3100, wav_max = 9600, n_bins = 1024):
    '''
    bins (and returns) spectrum to logarithmic scale between wav_min and wav_max with n_bins points
    '''

    dl = np.log(wav_max / wav_min) / n_bins
    n = np.arange(0.5, n_bins + 0.5)
    wav = spec[:,0]
    log_wav = wav_min * np.exp(n * dl)
    spl = CubicSpline(wav, spec[:,1])
    flux = spl(log_wav)
    flux[np.logical_or(log_wav < wav.min(), log_wav > wav.max())] = 0
    return np.array([log_wav, flux]).T

def cont_subtr(spec, n_spline_pts = 13):
    '''
    fits a cubic spline to a subset of points in the spectrum and subtracts it out to remove a pseudo continuum
    '''

    wav = spec[:,0]
    loc = int(len(wav)/(2*n_spline_pts))
    spl = CubicSpline(wav[loc:-loc:2*loc], spec[loc:-loc:2*loc,1])
    flux = spec[:,1] - spl(wav)
    return np.array([wav, flux]).T

def apodize(spec, end_pct = 0.05):
    '''
    tapers the edges of the spectrum using a hanning window
    '''

    size = int(end_pct * spec.shape[0])
    h = np.hanning(2 * size)
    window = np.concatenate((h[:size], np.ones(spec.shape[0] - 2 * size), h[-size:]))
    flux = spec[:,1] * window
    return np.array([spec[:,0], flux]).T

def flux_norm(spec):
    '''
    normalizes the flux (f) according to the algorithm: (f - f_min) / (f_max - f_min)
    '''

    f = spec[:,1]
    nflux = (f - f.min()) / (f.max() - f.min())
    return np.array([spec[:,0], nflux]).T

class Spectrum:
    '''
    base class for each spectrum to be analyzed that provides a self-contained pathway from metadata to processed spectra

    General Concepts
    ----------------
    spec : 2d array of (column-wise) wavelength, flux [wherever spec is used, it should be assumed to be in this form unless explicitly noted as otherwise]

    '''

    def __init__(self, sn_name, sn_subtype, spec_filename, redshift, SNR):
        '''
        initialize object with given arguments
        '''

        self.name = sn_name
        self.type = sn_subtype
        self.file = spec_filename
        self.z = redshift
        self.SNR = SNR
        self.spec = self.read_file()
        self.spec_flag = 'raw'

    def read_file(self):
        '''
        read spectrum from the given filename and return a 2d array of (column-wise) wavelength, flux
        '''
        
        tmp = ascii.read(self.file)
        cols = tmp.colnames
        return np.array([tmp[cols[0]], tmp[cols[1]]]).T

    def dez(self):
        '''
        de-redshifts (and returns) spectrum
        '''

        spec = self.spec
        wav = spec[:,0] / (1 + self.z)
        return np.array([wav, spec[:,1]]).T

    def preprocess(self, ksize = 15, n_spline_pts = 13, wav_min = 3100, wav_max = 9600, n_bins = 1024):
        '''
        perform preprocessing procedure on spectrum and return a 2d array of (column-wise) wavelength, flux
        '''

        # de-redshift
        spec = self.dez()

        # do median filtering
        spec = sp_medfilt(spec, ksize = ksize)

        # bin to logarithmic wavelength bins
        spec = log_bin(spec, wav_min = wav_min, wav_max = wav_max, n_bins = n_bins)

        # remove pseudo continuum
        spec = cont_subtr(spec, n_spline_pts = n_spline_pts)

        # apodize spectrum
        spec = apodize(spec)

        # normalize
        spec = flux_norm(spec)

        return spec

def main(query = None, query_res_fl = 'query_results.pkl'):
    '''
    Parameters
    ----------
    query : valid MySQL query
    query_res_fl : pkl file containing a dictionary of results yielded by a mysql_query function call

    Returns
    -------

    '''

    ###################### data acquisition ######################

    # if no query has been passed and a results file exists, read from that
    if (query is None) and os.path.isfile(query_res_fl):
        with open(query_res_fl, 'rb') as f:
            results = pkl.load(f)

    # otherwise, execute query against database, retrieve results, and write to disk for later use
    else:
        results = mysql_query(dbp.usr, dbp.pswd, dbp.db, query)
        with open('query.txt', 'w') as f:
            f.write(query)
        with open(query_res_fl, 'wb') as f:
            pkl.dump(results, f)








# do tests
if __name__ == "__main__":
    import doctest
    doctest.testmod()