# imports
import numpy as np
from Spectrum import Spectrum
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simps
from scipy.special import factorial

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
    integ_reg_area : breaks each spectrum into regions and calculates (and returns) the integrated area of each region
    os_balance : makes number of occurrences of each label (and associated features) the same by oversampling
    '''

    def __init__(self, spectra, labels, regions = 20, r_regions = 5, rs = 100):
        '''
        instantiation instructions

        Parameters
        ----------
        (object instance)
        spectra : 2d array where each row corresponds to the flux of a given spectrum
        labels : array of subtypes corresponding to the SNe in the rows of spectra
        regions (optional, int) : number of regions to break each spectrum into for integrating
                                    (NB: dividing this into n_bins from the Spectrum class should result in an integer)
        r_regions (optional, int) : number of regions to break each spectrum into for computing ratios of integrated areas
                                    (NB: dividing this into n_bins from the Spectrum class should result in an integer)
        rs : random state

        Doctests/Examples
        -----------------
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test_files/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> mlp = ML_prep(s.preprocess()[:,1].reshape(1,-1), np.array([s.type]), regions = 16, r_regions = 5)
        >>> mlp.X.shape
        (1, 80)
        >>> data = np.load('test_files/proc_test.npz')
        >>> mlp = ML_prep(data['arr_0'], data['arr_1'], regions = 16, r_regions = 5)
        >>> mlp.spectra.shape[0] == len(mlp.labels)
        True
        >>> mlp.X.shape
        (10, 80)
        '''

        # full (un-prepared) sets
        self.spectra = spectra
        self.labels = labels

        # full (featurized but not oversampled) sets
        self.X = self.featurize(regions = regions, r_regions = r_regions)
        self.y = self.proc_labels()

        # X scaler to be fitted to training data
        self.X_scaler = StandardScaler()

        np.random.seed(rs)

    def integ_reg_area(spectra, regions = 20):
        '''
        breaks each spectrum into regions and calculates (and returns) the integrated area of each region

        Parameters
        ----------
        spectra : 2d array where each row corresponds to the flux of a given spectrum
        regions (optional, int) : number of regions to break each spectrum into for integrating
                                  (NB: dividing this into n_bins from the Spectrum class should result in an integer)

        Returns
        -------
        2d array where each row corresponds to the integrated area of each region of a given spectrum

        Doctests/Examples
        -----------------
        >>> data = np.load('test_files/proc_test.npz')
        >>> mlp = ML_prep(data['arr_0'], data['arr_1'], regions = 16)
        >>> ML_prep.integ_reg_area(mlp.spectra, regions = 16).shape
        (10, 16)
        '''

        # split spectra into regions then integrate each of those regions
        sections = np.split(spectra, regions, axis = 1)
        return simps(sections, axis = 2).T

    def featurize(self, regions = 20, r_regions = 5):
        '''
        featurizes spectra for ingestion by ML models
            features are integrated areas of regions of each spectrum, the flux at midpoint of each region
            additional features are extracted by breaking each spectrum into r_regions and computing all 
                permutations of the ratios of integrated areas

        Parameters
        ----------
        (object instance)
        regions (optional, int) : number of regions to break each spectrum into for integrating
                                  (NB: dividing this into n_bins from the Spectrum class should result in an integer)
        r_regions (optional, int) : number of regions to break each spectrum into for computing ratios of integrated areas
                                    (NB: dividing this into n_bins from the Spectrum class should result in an integer)

        Returns
        -------
        features : 2d array where each row corresponds features of a given spectrum

        Doctests/Examples
        -----------------
        >>> data = np.load('test_files/proc_test.npz')
        >>> mlp = ML_prep(data['arr_0'], data['arr_1'], regions = 16, r_regions = 5)
        >>> mlp.featurize(regions = 16, r_regions = 5).shape
        (10, 80)
        '''

        # create container for features and then populate
        r_region_size = int(r_regions * (r_regions - 1) / 2)
        features = np.zeros((self.spectra.shape[0], 2 * regions + r_region_size))

        # midpoint of each region
        features[:, :regions] = self.spectra[:, int(self.spectra.shape[1]/(2*regions))::int(self.spectra.shape[1]/regions)]
        
        # integrated area of each region
        features[:, regions:(2*regions)] = ML_prep.integ_reg_area(self.spectra, regions = regions)

        # compute all ratio permutations
        areas = ML_prep.integ_reg_area(self.spectra, regions = r_regions)
        a_ratios = []
        for i in range(r_regions - 1):
            # set divisions by zero to zero
            a_ratios.append(np.divide(areas[:, (i+1):],  areas[:, :-(i+1)], where = areas[:, :-(i+1)] != 0))
        features[:, -r_region_size:] = np.concatenate(a_ratios, axis = 1)

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

        Doctests/Examples
        -----------------
        >>> data = np.load('test_files/proc_test.npz')
        >>> mlp = ML_prep(data['arr_0'], data['arr_1'], regions = 16, r_regions = 5)
        >>> len(mlp.labels) == 10
        True
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

        Doctests/Examples
        -----------------
        >>> data = np.load('test_files/proc_test.npz')
        >>> mlp = ML_prep(data['arr_0'], data['arr_1'], regions = 16, r_regions = 5)
        >>> len(ML_prep.os_balance(mlp.labels))
        16
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
        os_train (optional, bool) : bool that selects whether training data should be over sampled until class proportions are equal

        Returns
        -------
        splitting : list of length 4 (or 6) containing the splits of training, testing(, validation) data (each split is X, y)

        >>> data = np.load('test_files/proc_test.npz')
        >>> mlp = ML_prep(data['arr_0'], data['arr_1'], regions = 16, r_regions = 5)
        >>> X_train, y_train, X_test, y_test = mlp.train_test_val_split(tet = (0.625, 0.375))
        >>> X_train, y_train, X_test, y_test = mlp.train_test_val_split(tet = (0.8, 0.2), os_train = False)
        >>> X_train.shape 
        (8, 80)
        >>> X_test.shape
        (2, 80)
        >>> len(y_train)
        8
        >>> len(y_test)
        2
        >>> X_train, y_train, X_test, y_test, X_val, y_val = mlp.train_test_val_split(tet = (0.625, 0.250, 0.125))
        >>> X_train, y_train, X_test, y_test, X_val, y_val = mlp.train_test_val_split(tet = (0.6, 0.2, 0.2), os_train = False)
        >>> X_train.shape
        (6, 80)
        >>> X_test.shape
        (2, 80)
        >>> X_val.shape
        (2, 80)
        >>> len(y_train)
        6
        >>> len(y_test)
        2
        >>> len(y_val)
        2
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
            X_train = self.X[shuff_ind[:int(tet[0]*dlen)],:]
        self.X_scaler.fit(X_train)

        # construct testing(, validation) sets
        X_test = self.X[shuff_ind[int(tet[0]*dlen):int((tet[0]+tet[1])*dlen)],:]
        y_test = self.y[shuff_ind[int(tet[0]*dlen):int((tet[0]+tet[1])*dlen)]]
        if len(tet) == 3:
            X_val = self.X[shuff_ind[-int(tet[2]*dlen):],:]
            y_val = self.y[shuff_ind[-int(tet[2]*dlen):]]
            return X_train, y_train, X_test, y_test, X_val, y_val
        elif len(tet) == 2:
            return X_train, y_train, X_test, y_test
        else:
            raise ValueError('tet is not length 2 or 3')
