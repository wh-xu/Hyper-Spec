import sys, gc, logging
gc.enable()

from typing import Union, List
from config import * 

import tqdm
import pandas as pd

import hd_preprocess, hd_cluster
# from memory_profiler import profile

logger = logging.getLogger('HyperSpec')

# @profile
def main(args: Union[str, List[str]] = None) -> int:
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    
    # Disable dependency non-critical log messages.
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('cupy').setLevel(logging.WARNING)
    logging.getLogger('joblib').setLevel(logging.WARNING)

    # Load the configuration.
    config.parse(args)
    logger.debug('input_filepath= %s', config.input_filepath)
    # logger.debug('work_dir = %s', config.work_dir)
    # logger.debug('overwrite = %s', config.overwrite)
    # logger.debug('export_representatives = %s', config.export_representatives)
    logger.debug('cpu_core_preprocess = %s', config.cpu_core_preprocess)
    logger.debug('cpu_core_cluster = %s', config.cpu_core_cluster)
    logger.debug('batch_size = %d', config.batch_size)
    logger.debug('use_gpu_cluster = %s', config.use_gpu_cluster)

    logger.debug('min_peaks = %d', config.min_peaks)
    logger.debug('min_mz_range = %.2f', config.min_mz_range)
    logger.debug('min_mz = %.2f', config.min_mz)
    logger.debug('max_mz = %.2f', config.max_mz)
    logger.debug('remove_precursor_tol = %.2f', config.remove_precursor_tol)
    logger.debug('min_intensity = %.2f', config.min_intensity)
    logger.debug('max_peaks_used = %d', config.max_peaks_used)
    logger.debug('scaling = %s', config.scaling)

    logger.debug('hd_dim = %d', config.hd_dim)
    logger.debug('hd_Q = %d', config.hd_Q)
    logger.debug('hd_id_flip_factor = %.1f', config.hd_id_flip_factor)
    logger.debug('cluster_charges = %s', config.cluster_charges)

    logger.debug('precursor_tol = %.2f %s', *config.precursor_tol)
    logger.debug('rt_tol = %s', config.rt_tol)
    logger.debug('fragment_tol = %.2f', config.fragment_tol)
    logger.debug('eps = %.3f', config.eps)
    logger.debug('refine = %s', config.refine)

    bin_len, min_mz, max_mz = hd_cluster.get_dim(config.min_mz, config.max_mz, config.fragment_tol)
    
    # Generate LV-ID hypervectors
    lv_hvs, id_hvs = hd_cluster.gen_lv_id_hvs(config.hd_dim, config.hd_Q, bin_len, config.hd_id_flip_factor)

    
    ###################### 1. Load and parse spectra files
    processed_spectra_df = hd_preprocess.load_process_spectra_parallel(
        path=config.input_filepath, 
        file_type=config.file_type,
        config=config,
        logger=logger)

    processed_spectra_df.drop(
        processed_spectra_df.loc[~processed_spectra_df['precursor_charge'].isin(config.cluster_charges)].index, 
        inplace=True)
    logger.info("Preserve {} spectra for cluster charges: {}".format(len(processed_spectra_df), config.cluster_charges))


    ###################### 2. Cluster for each charge
    cluster_df = pd.DataFrame()
    for prec_charge_i in tqdm.tqdm(config.cluster_charges):
        # Select spectra with cluster charge
        idx = processed_spectra_df['precursor_charge']==prec_charge_i
        spec_df_by_charge = processed_spectra_df[idx]

        # Delete processed_spectra_df
        processed_spectra_df = processed_spectra_df[~idx]

        ###################### 2.1 Encoding and clustering
        logger.info(
            "Start clustering Charge {} with {} spectra".format(
                prec_charge_i, len(spec_df_by_charge)))
        bucket_cluster_labels = hd_cluster.encode_cluster_spectra(
            spectra_by_charge_df=spec_df_by_charge,
            config=config, logger=logger,
            bin_len=bin_len,
            lv_hvs=lv_hvs, id_hvs=id_hvs)

        logger.info("Post processing clustering labels for Charge {}".format(prec_charge_i))
        charge_cluster_labels = hd_cluster.post_process_cluster(
            spec_df_by_charge, 
            bucket_cluster_labels, 
            if_refine=config.refine, 
            precursor_tol_mass=config.precursor_tol[0], 
            precursor_tol_mode=config.precursor_tol[1], 
            rt_tol=config.rt_tol, 
            min_samples=2,
            logger=logger)

        spec_df_by_charge['cluster'] = list(charge_cluster_labels)
        cluster_df = pd.concat([cluster_df, spec_df_by_charge])

    logger.info("Exporting clustering labels to {}".format(config.output_filename))
    cluster_df.to_csv(config.output_filename, index=False)


if __name__ == "__main__":
    main()

