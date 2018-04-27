# imports
import numpy as np
from scipy.integrate import simps

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
        li = np.arange(len(labels), dtype = int) # label index array
        for idx, cnt in enumerate(counts):
            if cnt == mc:
                osi[(idx * mc):((idx + 1) * mc)] = li[labels == uniques[idx]]
            elif cnt < mc:
                sample_indices = li[labels == uniques[idx]]
                osi[(idx * mc):((idx + 1) * mc)] = np.random.choice(sample_indices, size = mc) # with replacement
            elif cnt > mc:
                raise ValueError('label {} has max count ({}) greater than label with calculated max count ({})'.format(
                                  uniques[idx], counts[idx], mc))

        # return (shuffled) oversample indices
        return np.random.choice(osi, size = len(osi), replace = False)

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
        X_train = self.X[shuff_ind[:int(tet[0]*dlen)],:]
        y_train = self.y[shuff_ind[:int(tet[0]*dlen)]]
        if os_train:
            train_ind = ML_prep.os_balance(y_train)
            X_train = X_train[train_ind]
            y_train = y_train[train_ind]
            c, u = np.unique(y_train, return_counts=True)
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