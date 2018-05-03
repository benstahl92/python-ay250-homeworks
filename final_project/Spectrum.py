# imports
import numpy as np
from astropy.io import ascii
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt

class Spectrum:
    '''
    base class for spectra that provides a self-contained pathway from metadata to various data products
        includes methods for direct use by instances and functions that can be exposed outside of a specific instance

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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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
        >>> Spectrum.read_file('test/sn1997y-19970209-uohp.flm').shape
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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

    def sp_medfilt(spec, ksize = 13):
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
        >>> spec1 = Spectrum.cont_subtr(s.spec)
        >>> spec1.shape[1]
        2
        >>> spec2, spl = Spectrum.cont_subtr(s.spec, ret_cont = True)
        >>> spec2.shape == spec1.shape
        True
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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
        >>> s = Spectrum('SN 1997y', 'Ia-norm', 'test/sn1997y-19970209-uohp.flm', 0.01587, 44.2219)
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
