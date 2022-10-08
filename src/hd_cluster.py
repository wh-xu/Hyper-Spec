from distutils.log import error
import time, logging
import math
from typing import Callable, Iterator, List, Optional, Tuple

from tqdm import tqdm

import numpy as np
np.random.seed(0)

import numba as nb
from numba import cuda
from numba.typed import List

import cupy as cp
import cuml
import rmm
rmm.reinitialize(pool_allocator=False, managed_memory=True)

import pandas as pd

import scipy.sparse as ss
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN

import config
from joblib import Parallel, delayed


def gen_lvs(D: int, Q: int):
    base = np.ones(D)
    base[:D//2] = -1.0
    l0 = np.random.permutation(base)
    levels = list()
    for i in range(Q+1):
        flip = int(int(i/float(Q) * D) / 2)
        li = np.copy(l0)
        li[:flip] = l0[:flip] * -1
        levels.append(list(li))
    return cp.array(levels, dtype=cp.float32).ravel()


def gen_idhvs(D: int, totalFeatures: int, flip_factor: float):
    nFlip = int(D//flip_factor)

    mu = 0
    sigma = 1
    bases = np.random.normal(mu, sigma, D)

    import copy
    generated_hvs = [copy.copy(bases)]

    for _ in range(totalFeatures-1):        
        idx_to_flip = np.random.randint(0, D, size=nFlip)
        bases[idx_to_flip] *= (-1)
        generated_hvs.append(copy.copy(bases))

    return cp.array(generated_hvs, dtype=cp.float32).ravel()


def gen_lv_id_hvs(
    D: int,
    Q: int,
    bin_len: int,
    id_flip_factor: float
):
    lv_hvs = gen_lvs(D, Q)
    lv_hvs = cuda_bit_packing(lv_hvs, Q+1, D)
    id_hvs = gen_idhvs(D, bin_len, id_flip_factor)
    id_hvs = cuda_bit_packing(id_hvs, bin_len, D)
    return lv_hvs, id_hvs


def cuda_bit_packing(orig_vecs, N, D):
    pack_len = (D+32-1)//32
    packed_vecs = cp.zeros(N * pack_len, dtype=cp.uint32)
    packing_cuda_kernel = cp.RawKernel(r'''
                    extern "C" __global__
                    void packing(unsigned int* output, float* arr, int origLength, int packLength, int numVec) {
                        int i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= origLength)
                            return;
                        for (int sample_idx = blockIdx.y; sample_idx < numVec; sample_idx += blockDim.y * gridDim.y) 
                        {
                            int tid = threadIdx.x;
                            int lane = tid % warpSize;
                            int bitPattern=0;
                            if (i < origLength)
                                bitPattern = __brev(__ballot_sync(0xFFFFFFFF, arr[sample_idx*origLength+i] > 0));
                            if (lane == 0) {
                                output[sample_idx*packLength+ (i / warpSize)] = bitPattern;
                            }
                        }
                    }
                    ''', 'packing')
    threads = 1024
    packing_cuda_kernel(((D + threads - 1) // threads, N), (threads,), (packed_vecs, orig_vecs, D, pack_len, N))

    return packed_vecs.reshape(N, pack_len)


def hd_encode_spectra_packed(csr_spectra, id_hvs_packed, lv_hvs_packed, N, D, Q, output_type):
    packed_dim = (D + 32 - 1) // 32
    encoded_spectra = cp.zeros(N * packed_dim, dtype=cp.uint32)
    
    spectra_data = cp.array(csr_spectra.data, dtype=cp.float32).ravel()
    spectra_indices = cp.array(csr_spectra.indices, dtype=cp.int32).ravel()
    spectra_indptr = cp.array(csr_spectra.indptr, dtype=cp.int32).ravel()
    
    hd_enc_lvid_packed_cuda_kernel = cp.RawKernel(r'''
                __device__ float* get2df(float* p, const int x, int y, const int stride) {
                    return (float*)((char*)p + x*stride) + y;
                }
                __device__ char get2d_bin(unsigned int* p, const int i, const int DIM, const int d) {
                    unsigned int v = ((*(p + i * ((DIM + 32-1)/32) + d/32)) >> ((32-1) - d % 32)) & 0x01;
                    if (v == 0) {
                        return -1;
                    } else {
                        return 1;
                    }
                }
                extern "C" __global__
                void hd_enc_lvid_packed_cuda(unsigned int* __restrict__ id_hvs_packed, unsigned int* __restrict__ level_hvs_packed, 
                                            int* __restrict__ feature_indices, float* __restrict__ feature_values, 
                                            int* __restrict__ csr_info, unsigned int* hv_matrix,
                                            int N, int Q, int D, int packLength) {
                    const int d = threadIdx.x + blockIdx.x * blockDim.x;
                    if (d >= D)
                        return;
                    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
                    {
                        // we traverse [start, end-1]
                        float encoded_hv_e = 0.0;
                        unsigned int start_range = csr_info[sample_idx];
                        unsigned int end_range = csr_info[sample_idx + 1];
                        #pragma unroll 1
                        for (int f = start_range; f < end_range; ++f) {
                            // encoded_hv_e += level_hvs[((int)(feature_values[f] * Q))*D+d] * id_hvs[feature_indices[f]*D+d];
                            // encoded_hv_e += level_hvs[(int)(info.Intensity * Q) * D + d] * id_hvs[info.Idx * D + d];
                            encoded_hv_e += get2d_bin(level_hvs_packed, (int)(feature_values[f] * Q), D, d) * \
                                            get2d_bin(id_hvs_packed, feature_indices[f], D, d);
                        }
                        
                        // hv_matrix[sample_idx*D+d] = (encoded_hv_e > 0)? 1 : -1;
                        int tid = threadIdx.x;
                        int lane = tid % warpSize;
                        int bitPattern=0;
                        if (d < D)
                            bitPattern = __ballot_sync(0xFFFFFFFF, encoded_hv_e > 0);
                        if (lane == 0) {
                            hv_matrix[sample_idx * packLength + (d / warpSize)] = bitPattern;
                        }
                    }
                }
                ''', 'hd_enc_lvid_packed_cuda')
                
    threads = 1024
    max_block = cp.cuda.runtime.getDeviceProperties(0)['maxGridSize'][1]
    hd_enc_lvid_packed_cuda_kernel(((D + threads - 1) // threads, min(N, max_block)), (threads,), (id_hvs_packed, lv_hvs_packed, spectra_indices, spectra_data, spectra_indptr, encoded_spectra, N, Q, D, packed_dim))

    if output_type=='numpy':
        return encoded_spectra.reshape(N, packed_dim).get()
    elif output_type=='cupy':
        return encoded_spectra.reshape(N, packed_dim)


@cuda.jit('float32(uint32, uint32)', device=True, inline=True)
def fast_hamming_op(a, b):
    return nb.float32(cuda.libdevice.popc(a^b))

TPB = 32
TPB1 = 33

@cuda.jit('void(uint32[:,:], float32[:,:], float32[:], float32, int32, int32)')
def fast_pw_dist_cosine_mask_packed(A, D, prec_mz, prec_tol, N, pack_len):
    """
        Pair-wise cosine distance
    """
    sA = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)
    sB = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)

    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx = cuda.blockIdx.x

    tmp = nb.float32(.0)
    for i in range((pack_len+TPB-1) // TPB):
        if y < N and (i*TPB+tx) < pack_len:
            sA[ty, tx] = A[y, i*TPB+tx]
        else:
            sA[ty, tx] = .0

        if (TPB*bx+ty) < N and (i*TPB+tx) < pack_len:
            sB[ty, tx] = A[TPB*bx+ty, i*TPB+tx]
        else:
            sB[ty, tx] = .0  
        cuda.syncthreads()

        for j in range(TPB):
            tmp += fast_hamming_op(sA[ty, j], sB[tx, j])

        cuda.syncthreads()

    if x<N and y<N and y>x:
        if cuda.libdevice.fabsf((prec_mz[x]-prec_mz[y])/prec_mz[y])>=prec_tol:
            D[x,y] = 1.0
            D[y,x] = 1.0
        else:
            tmp/=(16*pack_len)
            D[x,y] = tmp
            D[y,x] = tmp


def fast_nb_cosine_dist_mask(hvs, prec_mz, prec_tol, output_type, stream=None):
    N, pack_len = hvs.shape

    # start = time.time()

    hvs_d = cp.array(hvs)
    prec_mz_d = cp.array(prec_mz.ravel())
    prec_tol_d = nb.float32(prec_tol/1e6)
    dist_d = cp.zeros((N,N), dtype=cp.float32)
    # print("Data loading time: ", time.time()-start)

    TPB = 32
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # start = time.time()
    fast_pw_dist_cosine_mask_packed[blockspergrid, threadsperblock]\
        (hvs_d, dist_d, prec_mz_d, prec_tol_d, N, pack_len)
    cuda.synchronize()
    # print("CUDA computing time: ", time.time()-start)

    # start = time.time()
    if output_type=='cupy':
        dist = dist_d
    else:
        dist = dist_d.get()
    del dist_d
    # print("Data fetching time: ", time.time()-start)

    return dist


def get_dim(min_mz: float, max_mz: float, bin_size: float) \
        -> Tuple[int, float, float]:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum mass in the mass range (inclusive).
    max_mz : float
        The maximum mass in the mass range (inclusive).
    bin_size : float
        The bin size (in Da).

    Returns
    -------
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    # print(start_dim, end_dim, min_mz, max_mz, bin_size, math.ceil((end_dim - start_dim) / bin_size))
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim


# @nb.njit(cache=True)
def _to_csr_vector(
    spectra: pd.DataFrame, 
    min_mz: float, 
    bin_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz = spectra['mz'].to_numpy()
    intensity = spectra['intensity'].to_numpy()

    indptr = np.zeros(len(mz)+1, np.int32)
    indptr[1:] = np.array([len(spec) for spec in mz], np.int32)
    indptr = np.cumsum(indptr).ravel()

    indices = np.floor((np.hstack(mz).ravel()-min_mz)/bin_size)
    data = np.hstack(intensity).ravel()
    return data, indices, indptr


from multiprocessing import shared_memory
class SharedMem:
    """A simple example class"""

    def __init__(self, name: str, data: np.ndarray=None):

        if data is None:
            self.shm_data = shared_memory.SharedMemory(name=name)
        else:
            self.name = name
            self.nbytes = data.nbytes
            self.dtype = data.dtype
            self.shape = data.shape

            try:
                self.shm_data = shared_memory.SharedMemory(name=self.name, create=True, size=self.nbytes)
                self.put(data)
            except FileExistsError:
                self.shm_data = shared_memory.SharedMemory(name=self.name, create=False, size=self.nbytes)
                self.put(data)
            except Exception as shm_e:
                raise error(shm_e)

    def get_meta(self):
        return {'name': self.name, 'nbytes': self.nbytes, 'shape': self.shape, 'dtype': self.dtype}
        
    def put(self, data: np.ndarray):
        data_shm = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_data.buf)
        data_shm[:] = data[:]

    def gather(self, idx: list = None):
        if idx is None:
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_data.buf)
        else:
            col_size = 1 if len(self.shape)==1 else self.shape[1]
            shape = list(self.shape)
            shape[0] = idx[1]-idx[0]

            arr = np.ndarray(
                shape, dtype=self.dtype, 
                buffer=self.shm_data.buf[idx[0]*self.dtype.itemsize*col_size: idx[1]*self.dtype.itemsize*col_size])
        return arr

    def close(self):
        self.shm_data.close()
        self.shm_data.unlink()


def encode_func_shm(
    slice_idx: tuple,
    shm_dict: dict,
    D: int,
    Q: int,
    dim: int,
    output_type: str
) -> np.ndarray:
    lv_hvs = cp.array(shm_dict['lv_hvs'].gather())
    id_hvs = cp.array(shm_dict['id_hvs'].gather())

    indptr = shm_dict['indptr'].gather([slice_idx[0], slice_idx[1]+1])

    shm_idx = [indptr[0], indptr[-1]]
    data = shm_dict['intensity'].gather(shm_idx)
    indices = shm_dict['indices'].gather(shm_idx)    

    batch_size = len(indptr)-1
    csr_vec = ss.csr_matrix(
            (data, indices, indptr-indptr[0]), (batch_size, dim), np.float32, False)

    return hd_encode_spectra_packed(csr_vec, id_hvs, lv_hvs, batch_size, D, Q, output_type)


def encode_func(
    slice_idx: tuple,
    data_dict: dict,
    D: int,
    Q: int,
    dim: int,
    output_type: str
) -> np.ndarray:
    indptr = data_dict['indptr'][slice_idx[0]: slice_idx[1]+1]
    
    data, indices = data_dict['intensity'][indptr[0]: indptr[-1]], data_dict['indices'][indptr[0]: indptr[-1]]

    batch_size = len(indptr)-1
    csr_vec = ss.csr_matrix(
            (data, indices, indptr-indptr[0]), (batch_size, dim), np.float32, False)

    lv_hvs = cp.array(data_dict['lv_hvs'])
    id_hvs = cp.array(data_dict['id_hvs'])

    return hd_encode_spectra_packed(csr_vec, id_hvs, lv_hvs, batch_size, D, Q, output_type)



def encode_preprocessed_spectra(
    spectra_df: pd.DataFrame, 
    config: config,
    dim: int,
    lv_hvs_packed: cp.array,
    id_hvs_packed: cp.array,
    logger: logging,
    batch_size: int = 5000,
    output_type: str='numpy'
)-> List:
    # Create shared memory
    start = time.time()

    num_batch = len(spectra_df)//batch_size+1

    lv_hvs = cp.asnumpy(lv_hvs_packed).ravel()
    id_hvs = cp.asnumpy(id_hvs_packed).ravel()

    intensity, indices, indptr = _to_csr_vector(spectra_df, config.min_mz, config.fragment_tol)
    spectra_df.drop(columns=['mz', 'intensity'], inplace=True)

    data_dict = {'lv_hvs': lv_hvs, 'id_hvs': id_hvs, 'intensity': intensity, 'indices': indices, 'indptr': indptr}

    encoded_spectra = [ encode_func(
        [i*batch_size, min((i+1)*batch_size, len(spectra_df))], 
        data_dict, config.hd_dim, config.hd_Q, dim, output_type) for i in tqdm(range(num_batch)) ] 
                    
    encoded_spectra = np.concatenate(encoded_spectra, dtype=np.uint32)\
        if output_type=='numpy' else encoded_spectra

    logger.info("Encode {} spectra in {:.4f}s".format(len(encoded_spectra), time.time()-start))

    return encoded_spectra


def _get_bucket_idx_list(
    spectra_by_charge_df: pd.DataFrame,
    logger: logging
):
    # Get bucket list
    buckets = spectra_by_charge_df.bucket.unique()
    num_bucket = len(buckets)

    bucket_idx_arr = np.zeros((num_bucket ,2), dtype=np.int32)
    bucket_size_arr = np.zeros(num_bucket, dtype=np.int32)
    for i, b_i in enumerate(buckets):
        bucket_idx_i = (spectra_by_charge_df.bucket==b_i).to_numpy()
        bucket_idx_i = np.argwhere(bucket_idx_i==True).flatten()
        bucket_idx_arr[i, :] = [bucket_idx_i[0], bucket_idx_i[-1]]
        bucket_size_arr[i] = bucket_idx_i[-1]-bucket_idx_i[0]+1
    
    hist, bins = np.histogram(
        bucket_size_arr, bins=[0, 300, 1000, 5000, 10000, 20000, 30000], density=False)

    logger.info("There are {} buckets. Maximum bucket size = {}".format(num_bucket, max(bucket_size_arr)))
    logger.info("Bucket size distribution:")
    for i in range(len(bins)-1):
        logger.info("{:.2f}% of bucket size between {} and {}".format(hist[i]/num_bucket*100, bins[i], bins[i+1]))

    return bucket_idx_arr, bucket_size_arr


def schedule_bucket(
    spectra_by_charge_df: pd.DataFrame,
    logger: logging
):
    bucket_idx_arr, bucket_size_arr = _get_bucket_idx_list(spectra_by_charge_df, logger)

    # Sort the buckets based on their sizes
    sort_idx = np.argsort(-bucket_size_arr)
    sorted_bucket_idx_arr = bucket_idx_arr[sort_idx]

    reorder_idx = np.argsort(sort_idx)

    return {
        'sort_bucket_idx_arr': sorted_bucket_idx_arr, 
        'reorder_idx': reorder_idx}


def cluster_bucket(
    bucket_slice: tuple, 
    data_dict: dict, 
    prec_tol: float, 
    cluster_func: Callable,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return np.array([-1])
    else:
        bucket_slice[1] += 1
        bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]]
        bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]

        pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, prec_tol, output_type)
        cluster_func.fit(pw_dist) #
        L = cluster_func.labels_
        del pw_dist

        return L


def cluster_encoded_spectra(
    spectra_by_charge_df: pd.DataFrame,
    encoded_spectra_hv: np.array,
    config: config,
    logger: logging
):
    # Save data to shared memory
    start = time.time()
    prec_mz = np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32)
    data_dict = {'hv': encoded_spectra_hv, 'prec_mz': prec_mz}

    ## Start clustering in GPU or CPU ##
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)
    if config.use_gpu_cluster:
        # DBSCAN clustering on GPU
        dbscan_cluster_func = cuml.DBSCAN(
            eps=config.eps, min_samples=2, metric='precomputed',
            calc_core_sample_indices=False, output_type='numpy')
        cluster_device = 'GPU'
    else:
        # DBSCAN clustering on CPU
        dbscan_cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core_cluster)
        cluster_device = 'CPU'
   

    cluster_labels = [cluster_bucket(
        bucket_slice=b_slice_i, 
        data_dict=data_dict,
        prec_tol=config.precursor_tol[0], 
        cluster_func=dbscan_cluster_func,
        output_type='cupy' if config.use_gpu_cluster else 'numpy') 
        for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    cluster_labels = [cluster_labels[i] for i in bucket_idx_dict['reorder_idx']]

    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))

    return cluster_labels


def encode_cluster_spectra(
    spectra_by_charge_df: pd.DataFrame,
    config: config,
    logger: logging,
    bin_len: int,
    lv_hvs: cp.array,
    id_hvs: cp.array
):
    # Encode spectra
    logger.info("Start encoding")
    encoded_spectra_hv = encode_preprocessed_spectra(
            spectra_df=spectra_by_charge_df, 
            config=config, dim=bin_len, logger=logger,
            lv_hvs_packed=lv_hvs, id_hvs_packed=id_hvs,
            output_type='numpy')

    # Cluster encoded spectra
    logger.info("Start clustering")    
    cluster_labels = cluster_encoded_spectra(
        spectra_by_charge_df=spectra_by_charge_df,
        encoded_spectra_hv=encoded_spectra_hv,
        config=config, logger=logger)

    return cluster_labels


def encode_cluster_bucket_shm(
    bucket_slice: tuple, 
    shm_dict: dict,
    D: int,
    Q: int,
    prec_tol: float,
    bin_len: int,
    cluster_func: Callable,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return np.array([-1])
    else:
        bucket_slice[1] += 1
        bucket_hv = encode_func_shm(
            slice_idx=bucket_slice,
            shm_dict=shm_dict,
            D=D, Q=Q,
            dim=bin_len,
            output_type=output_type)

        bucket_prec_mz = shm_dict['shm_prec_mz'].gather(bucket_slice)

        # start = time.time()
        pw_dist = fast_nb_cosine_dist_mask(
            bucket_hv, bucket_prec_mz, prec_tol, output_type)

        cluster_func.fit(pw_dist) #
        L = cluster_func.labels_
        del pw_dist, bucket_hv
        
        return L


def post_process_cluster(
    spectra_df, 
    bucket_cluster_labels, 
    if_refine, 
    precursor_tol_mass, 
    precursor_tol_mode, 
    rt_tol, 
    min_samples,
    logger
    ):
    '''
        Re-order and assign unique cluster labels
    '''
    init_unique_cluster_num = sum([np.amax(l_i)+1 for l_i in bucket_cluster_labels])
    
    logger.info('Finetune %d initial unique non-singleton clusters to not'
                ' exceed %.2f %s precursor m/z tolerance%s',
                init_unique_cluster_num, precursor_tol_mass, precursor_tol_mode,
                f' and {rt_tol} retention time tolerance' if rt_tol is not None else '')

    reorder_labels = []
    label_base, cluster_idx_start = 0, 0
    ft_unique_cluster_num, ft_noise_cluster_num = 0, 0
    for idx_i, cluster_i in enumerate(bucket_cluster_labels):
        # print(cluster_i)
        cluster_i = cluster_i.flatten()

        # Cluster refinement step
        if if_refine:
            # Refine initial clusters to make sure spectra within a cluster don't
            # have an excessive precursor m/z difference.
            order = np.argsort(cluster_i)
            reverse_order = np.argsort(order)
            sorted_cluster_i = cluster_i[order]

            precursor_mzs_i, rts_i = np.hsplit(spectra_df.iloc[cluster_idx_start: cluster_idx_start+len(cluster_i)][['precursor_mz', 'retention_time']].to_numpy()[order, :], 2)
            precursor_mzs_i, rts_i = precursor_mzs_i.flatten(), rts_i.flatten()
            cluster_idx_start += len(cluster_i)

            if sorted_cluster_i[-1] == -1:     # Only noise samples.
                n_clusters, n_noise = 0, len(sorted_cluster_i)
            else:
                group_idx = nb.typed.List(_get_cluster_group_idx(sorted_cluster_i))
                n_clusters = nb.typed.List(
                    [_postprocess_cluster(
                        sorted_cluster_i[start_i:stop_i], 
                        precursor_mzs_i[start_i:stop_i], 
                        rts_i[start_i:stop_i], 
                        precursor_tol_mass, precursor_tol_mode, rt_tol, min_samples)
                        for start_i, stop_i in group_idx])

                _assign_unique_cluster_labels(sorted_cluster_i, group_idx, n_clusters, min_samples)

                cluster_i[:] = sorted_cluster_i[reverse_order]
                n_clusters, n_noise = np.amax(cluster_i) + 1, np.sum(cluster_i==-1)

            ft_unique_cluster_num += n_clusters
            ft_noise_cluster_num += n_noise
                
        # Re-order and assign unique cluster labels
        noise_idx = cluster_i == -1
        num_clusters, num_noises = np.amax(cluster_i) + 1, np.sum(noise_idx)

        cluster_i[noise_idx] = np.arange(num_clusters, num_clusters + num_noises)
        cluster_i += label_base
        label_base += (num_clusters+num_noises)

        reorder_labels.append(cluster_i)

    reorder_labels = np.hstack(reorder_labels)

    logger.info('%d unique non-singleton clusters after precursor m/z '
                'finetuning, %d total clusters',
                ft_unique_cluster_num, ft_unique_cluster_num + ft_noise_cluster_num)

    return reorder_labels


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.
    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).
    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while clusters[start_i] == -1 and start_i < clusters.shape[0]:
        start_i += 1
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _postprocess_cluster(
    cluster_labels: np.ndarray, 
    cluster_mzs: np.ndarray,
    cluster_rts: np.ndarray, 
    precursor_tol_mass: float,
    precursor_tol_mode: str, 
    rt_tol: float,
    min_samples: int
    ) -> int:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference.
    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    cluster_rts : np.ndarray
        Retention times of the samples in a single initial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    min_samples : int
        The minimum number of samples in a cluster.
    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    cluster_labels[:] = -1
    # No splitting needed if there are too few items in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if len(cluster_labels) < min_samples:
        n_clusters = 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        cluster_assignments = fcluster(
            _linkage(cluster_mzs, precursor_tol_mode),
            precursor_tol_mass, 'distance') - 1

        # Optionally restrict clusters by their retention time as well.
        if rt_tol is not None:
            cluster_assignments_rt = fcluster(
                _linkage(cluster_rts), rt_tol, 'distance') - 1
            # Merge cluster assignments based on precursor m/z and RT.
            # First prime factorization is used to get unique combined cluster
            # labels, after which consecutive labels are obtained.
            cluster_assignments = np.unique(
                cluster_assignments * 2 + cluster_assignments_rt * 3,
                return_inverse=True)[1]

        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels[:] = 0
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            n_clusters = 0
        else:
            unique_clusters, inverse, counts = np.unique(
                cluster_assignments, return_inverse=True, return_counts=True)
            non_noise_clusters = np.where(counts >= min_samples)[0]
            labels = -np.ones_like(unique_clusters)
            labels[non_noise_clusters] = np.unique(unique_clusters[non_noise_clusters], return_inverse=True)[1]
            cluster_labels[:] = labels[inverse]
            n_clusters = len(non_noise_clusters)
    return n_clusters


@nb.njit(cache=True, fastmath=True)
def _linkage(
    values: np.ndarray, 
    tol_mode: str = None
    ) -> np.ndarray:
    """
    Perform hierarchical clustering of a one-dimensional m/z or RT array.
    Because the data is one-dimensional, no pairwise distance matrix needs to
    be computed, but rather sorting can be used.
    For information on the linkage output format, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Parameters
    ----------
    values : np.ndarray
        The precursor m/z's or RTs for which pairwise distances are computed.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z, 'rt' for
        retention time).
    Returns
    -------
    np.ndarray
        The hierarchical clustering encoded as a linkage matrix.
    """
    linkage = np.zeros((values.shape[0] - 1, 4), np.double)
    # min, max, cluster index, number of cluster elements
    # noinspection PyUnresolvedReferences
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(values.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]  # Always positive.
            if tol_mode == 'ppm':
                dist = dist / clusters[i][0] * 10 ** 6
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [clusters[min_i][2], clusters[min_i + 1][2],
                          min_dist, n_points]
        clusters[min_i] = (clusters[min_i][0], clusters[min_i + 1][1],
                           values.shape[0] + it, n_points)
        del clusters[min_i + 1]

    return linkage


@nb.njit(cache=True)
def _assign_unique_cluster_labels(cluster_labels: np.ndarray,
                                  group_idx: nb.typed.List,
                                  n_clusters: nb.typed.List,
                                  min_samples: int) -> None:
    """
    Make sure all cluster labels are unique after potential splitting of
    clusters to avoid excessive precursor m/z differences.
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels per cluster grouping.
    group_idx : nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the cluster groupings.
    n_clusters: nb.typed.List[int]
        The number of clusters per cluster grouping.
    min_samples : int
        The minimum number of samples in a cluster.
    """
    current_label = 0
    for (start_i, stop_i), n_cluster in zip(group_idx, n_clusters):
        if n_cluster > 0 and stop_i - start_i >= min_samples:
            current_labels = cluster_labels[start_i:stop_i]
            current_labels[current_labels != -1] += current_label
            current_label += n_cluster
        else:
            cluster_labels[start_i:stop_i] = -1

