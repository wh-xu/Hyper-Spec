import logging, time, os, glob
from typing import  Optional

import tqdm
import numpy as np
import numba as nb
import pandas as pd

import bottleneck as bn
from joblib import Parallel, delayed

from config import Config
from utils import load_mgf_file

@nb.njit(fastmath=True, cache=True)
def _precursor_to_interval(
    mz: float,
    charge: int,
    interval_width: int
    ) -> int:
    """
    Convert the precursor m/z to the neutral mass and get the interval index.
    Parameters
    ----------
    mz : float
        The precursor m/z.
    charge : int
        The precursor charge.
    interval_width : int
        The width of each m/z interval.
    Returns
    -------
    int
        The index of the interval to which a spectrum with the given m/z and
        charge belongs.
    """
    hydrogen_mass, cluster_width = 1.00794, 1.0005079
    neutral_mass = (mz - hydrogen_mass) * max(abs(charge), 1)
    return round(neutral_mass / cluster_width) // interval_width


@nb.njit(cache=True)
def _get_mz_mask(
    mz: np.array,
    min_mz: float, 
    max_mz: float
):
    mask = np.ones(mz.size, dtype=np.bool_)
    for i in range(mz.size):
        if(mz[i]<min_mz or mz[i]>max_mz):
            mask[i] = False

    return mask

def _set_mz_range(
    spectrum: list,
    min_mz: Optional[float] = None,
    max_mz: Optional[float] = None
):
    """
    Restrict the mass-to-charge ratios of the fragment peaks to the
    given range.

    Parameters
    ----------
    min_mz : Optional[float], optional
        Minimum m/z (inclusive). If not set no minimal m/z restriction will
        occur.
    max_mz : Optional[float], optional
        Maximum m/z (inclusive). If not set no maximal m/z restriction will
        occur.

    Returns
    -------
    self : `MsmsSpectrum`
    """
    if min_mz is None and max_mz is None:
        return spectrum
    else:
        if min_mz is None:
            min_mz = spectrum[6][0]
        if max_mz is None:
            max_mz = spectrum[6][-1]

    mask = _get_mz_mask(spectrum[6], min_mz, max_mz)

    spectrum[6] = spectrum[6][mask]
    spectrum[7] = spectrum[7][mask]
    return spectrum


@nb.njit(fastmath=True, cache=True)
def _check_spectrum_valid(
    spectrum_mz: list,\
    min_peaks: int,\
    min_mz_range: float
    ) -> bool:
    """
    Check whether a cluster is of good enough quality to be used.
    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the cluster whose quality is checked.
    min_peaks : int
        Minimum number of peaks the cluster has to contain.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover.
    Returns
    -------
    bool
        True if the cluster has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)



@nb.njit(fastmath=True, cache=True)
def mass_diff(mz1, mz2, mode_is_da):
    """
    Calculate the mass difference(s).
    Parameters
    ----------
    mz1
        First m/z value(s).
    mz2
        Second m/z value(s).
    mode_is_da : bool
        Mass difference in Dalton (True) or in ppm (False).
    Returns
    -------
        The mass difference(s) between the given m/z values.
    """
    return mz1 - mz2 if mode_is_da else (mz1 - mz2) / mz2 * 10 ** 6


@nb.njit(cache=True)
def mass_diff_mask(mz, remove_mz, tol, mode_is_da):
    """
    Calculate the mass difference(s).
    Parameters
    ----------
    mz1
        First m/z value(s).
    mz2
        Second m/z value(s).
    mode_is_da : bool
        Mass difference in Dalton (True) or in ppm (False).
    Returns
    -------
        The mass difference(s) between the given m/z values.
    """
    mask = np.ones(mz.size, np.bool_)

    for i in range(len(remove_mz)):
        mask_i = np.abs(remove_mz[i] - mz)>tol \
            if mode_is_da else np.abs((remove_mz[i] - mz)/remove_mz[i]*10**6)>tol
        mask = np.logical_and(mask, mask_i)
    return mask

def _remove_precursor_peak(
        spectrum: list,
        fragment_tol_mass: float,
        fragment_tol_mode: str,
        isotope: int = 0,
    ):
    # TODO: This assumes [M+H]x charged ions.
    adduct_mass = 1.007825
    neutral_mass = (spectrum[2] - adduct_mass) * spectrum[1]
    c_mass_diff = 1.003355
    remove_mz = [
        (neutral_mass + iso * c_mass_diff) / charge + adduct_mass
        for charge in range(spectrum[1], 0, -1)
        for iso in range(isotope + 1)
    ]
    remove_mz = np.array(remove_mz, dtype=np.float32)

    # vectorized
    mask = mass_diff_mask(
        mz=spectrum[6], remove_mz=remove_mz, 
        tol=fragment_tol_mass, mode_is_da=fragment_tol_mode == "Da")

    # Remove the masked mz and peaks
    spectrum[6] = spectrum[6][mask]
    spectrum[7] = spectrum[7][mask]
    return spectrum


def get_intensity_mask(
    intensity: np.array,
    min_intensity: float,
    max_num_peaks: int
):
    top_intensity_idx = bn.argpartition(-intensity, max_num_peaks)[:max_num_peaks]\
        if len(intensity) > max_num_peaks else np.arange(len(intensity))
    
    min_intensity *= intensity[top_intensity_idx].max()

    # # Only retain at most the `max_num_peaks` most intense peaks.
    mask = np.zeros(intensity.size, np.bool_)
    idx = intensity[top_intensity_idx]>min_intensity
    mask[top_intensity_idx[idx]] = True
    return mask


def _filter_intensity(
        spectrum: list,
        min_intensity: float = 0.0,
        max_num_peaks: Optional[int] = None
    ):
        if max_num_peaks is None:
            max_num_peaks = len(spectrum[7])

        mask = get_intensity_mask(
            intensity=spectrum[7], 
            min_intensity=min_intensity, 
            max_num_peaks=max_num_peaks)

        # return update_spectrum_by_mask(spectrum, mask)
        spectrum[6] = spectrum[6][mask]
        spectrum[7] = spectrum[7][mask]
        return spectrum


def _scale_intensity(
        spectrum_intensity,
        scaling: Optional[str] = None,
        max_intensity: Optional[float] = None,
        degree: int = 2,
        base: int = 2,
        max_rank: Optional[int] = None,
    ):
        if scaling == "root":
            spectrum_intensity = np.power(
                spectrum_intensity, 1 / degree
            ).astype(np.float32)
        elif scaling == "log":
            spectrum_intensity = (
                np.log1p(spectrum_intensity) / np.log(base)
            ).astype(np.float32)
        elif scaling == "rank":
            if max_rank is None:
                max_rank = len(spectrum_intensity)
            if max_rank < len(spectrum_intensity):
                raise ValueError(
                    "`max_rank` should be greater than or equal to the number "
                    "of peaks in the spectrum. See `filter_intensity` to "
                    "reduce the number of peaks in the spectrum."
                )
            spectrum_intensity = (
                max_rank - np.argsort(np.argsort(spectrum_intensity)[::-1])
            ).astype(np.float32)
        if max_intensity is not None:
            spectrum_intensity = (
                spectrum_intensity* max_intensity / spectrum_intensity.max()
            ).astype(np.float32)
        return spectrum_intensity

@nb.njit(cache=True)
def _norm_intensity(
    spectrum_intensity
):
    """
    Normalize cluster peak intensities by their vector norm.
    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The cluster peak intensities to be normalized.
    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def preprocess_read_spectra_list(
        spectra_list: list, 
        min_peaks: int = 5, min_mz_range: float = 250.0,
        mz_interval: int = 1,
        mz_min: Optional[float] = 101.0,
        mz_max: Optional[float] = 1500.,
        remove_precursor_tolerance: Optional[float] = 1.50,
        min_intensity: Optional[float] = 0.01,
        max_peaks_used: Optional[int] = 50,
        scaling: Optional[str] = 'off'
    ):
    """
    Process a cluster.
    Processing steps include:
    - Restrict the m/z range to a minimum and maximum m/z.
    - Remove peak(s) around the precursor m/z value.
    - Remove peaks below a percentage of the base peak intensity.
    - Retain only the top most intense peaks.
    - Scale and normalize peak intensities.
    Parameters
    ----------
    spectrum : MsmsSpectrum
        The cluster to be processed.
    min_peaks : int
        Minimum number of peaks the cluster has to contain to be valid.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover to be valid.
    mz_min : Optional[float], optional
        Minimum m/z (inclusive). If not set no minimal m/z restriction will
        occur.
    mz_max : Optional[float], optional
        Maximum m/z (inclusive). If not set no maximal m/z restriction will
        occur.
    remove_precursor_tolerance : Optional[float], optional
        Fragment mass tolerance (in Dalton) around the precursor mass to remove
        the precursor peak. If not set, the precursor peak will not be removed.
    min_intensity : Optional[float], optional
        Remove peaks whose intensity is below `min_intensity` percentage
        of the base peak intensity. If None, no minimum intensity filter will
        be applied.
    max_peaks_used : Optional[int], optional
        Only retain the `max_peaks_used` most intense peaks. If None, all peaks
        are retained.
    scaling : {'root', 'log', 'rank'}, optional
        Method to scale the peak intensities. Potential transformation options
        are:
        - 'root': Square root-transform the peak intensities.
        - 'log':  Log2-transform (after summing the intensities with 1 to avoid
          negative values after the transformation) the peak intensities.
        - 'rank': Rank-transform the peak intensities with maximum rank
          `max_peaks_used`.
        - None: No scaling is performed.
    Returns
    -------
    List
        The processed cluster.
    """
    invalid_spec_list = []
    for i in range(len(spectra_list)):
        spectra_list[i] = _set_mz_range(spectra_list[i], mz_min, mz_max)

        # Check if spectrum is valid
        if not _check_spectrum_valid(spectra_list[i][6], min_peaks, min_mz_range):
            invalid_spec_list.append(i)
            continue

        if remove_precursor_tolerance is not None:
            spectra_list[i] = _remove_precursor_peak(spectra_list[i], remove_precursor_tolerance, 'Da', 0)
            if not _check_spectrum_valid(spectra_list[i][6], min_peaks, min_mz_range):
                invalid_spec_list.append(i)
                continue
        
        if min_intensity is not None or max_peaks_used is not None:
            min_intensity = 0. if min_intensity is None else min_intensity
            spectra_list[i] = _filter_intensity(spectra_list[i], min_intensity, max_peaks_used)
            if not _check_spectrum_valid(spectra_list[i][6], min_peaks, min_mz_range):
                invalid_spec_list.append(i)
                continue

        spectra_list[i][7] = _scale_intensity(spectra_list[i][7], scaling, max_rank=max_peaks_used)

        spectra_list[i][7] = _norm_intensity(spectra_list[i][7])

        # Add bucket
        interval_i = _precursor_to_interval(
            mz=spectra_list[i][2], charge=spectra_list[i][1], interval_width=mz_interval)
        spectra_list[i][0] = interval_i

    # Delete invalid spectrum
    for i in invalid_spec_list:
        spectra_list[i] = -1
    spectra_list = [item for item in spectra_list if item!=-1]

    return spectra_list


def fast_mgf_parse(filename):
    read_spectra_list = load_mgf_file(filename)
    return read_spectra_list


def load_process_single(
    file: str,
    min_peaks: int = 5, min_mz_range: float = 250.0,
    mz_interval: int = 1,
    mz_min: Optional[float] = 101.0,
    mz_max: Optional[float] = 1500.,
    remove_precursor_tolerance: Optional[float] = 1.50,
    min_intensity: Optional[float] = 0.01,
    max_peaks_used: Optional[int] = 50,
    scaling: Optional[str] = 'off'
):
    spec_list = fast_mgf_parse(file)
    spec_list = preprocess_read_spectra_list(
        spectra_list = spec_list,
        min_peaks = min_peaks, min_mz_range = min_mz_range,
        mz_interval = mz_interval,
        mz_min = mz_min, mz_max = mz_max,
        remove_precursor_tolerance = remove_precursor_tolerance,
        min_intensity = min_intensity,
        max_peaks_used = max_peaks_used,
        scaling = scaling)

    return spec_list


def load_process_spectra_parallel(
    path:str, 
    file_type: str, 
    config: Config,
    logger:logging
):
    # 1. Load and preprocess spectra data from MGF or mzXML files
    input_files = glob.glob(os.path.join(path, '*.'+file_type))
    files_with_size = [(file_i, os.stat(file_i).st_size/1e9) for file_i in input_files]
    
    logger.info('Starting processing {} {} files with {:.3f}GB size on {} cores'.format(len(input_files), file_type, sum([s[1] for s in files_with_size]), config.cpu_core_preprocess))
    
    start = time.time()
    with Parallel(n_jobs=config.cpu_core_preprocess) as parallel_pool:
        read_spectra_list = parallel_pool(
            delayed(load_process_single)(
                file = f_i, 
                min_peaks = config.min_peaks, min_mz_range = config.min_mz_range,
                mz_interval = config.mz_interval,
                mz_min = config.min_mz, mz_max = config.max_mz,
                remove_precursor_tolerance = config.remove_precursor_tol,
                min_intensity = config.min_intensity,
                max_peaks_used = config.max_peaks_used,
                scaling = config.scaling) \
                for f_i in tqdm.tqdm(input_files))

    read_spectra_list = [j for i in read_spectra_list for j in i]
    total_spec_num = len(read_spectra_list)

    parse_time = time.time() - start
    logger.info("Load and process {} spectra in {:.4f}s".format(total_spec_num, parse_time))

    read_spectra_list = pd.DataFrame(read_spectra_list,\
        columns=['bucket', 'precursor_charge', 'precursor_mz', 'identifier',
        'scan', 'retention_time', 'mz', 'intensity'])
    
    # Add exception for scan missing
    for c in read_spectra_list.columns:
        if c in ['precursor_charge']:
            read_spectra_list[c] = read_spectra_list[c].astype(np.int8)

        if c in ['scan', 'bucket']:
            read_spectra_list[c] = read_spectra_list[c].astype(np.int32)

        if c in ['identifier']:
            read_spectra_list[c] = read_spectra_list[c].astype('category')

        if c in ['retention_time', 'precursor_mz']:
            read_spectra_list[c] = read_spectra_list[c].astype(np.float32)

    read_spectra_list = read_spectra_list.sort_values(by=['precursor_charge', 'bucket'], ascending=True)
    
    return read_spectra_list.reset_index(drop=True)
