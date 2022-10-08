import numpy as np
import bottleneck as bn
import sys, time

cimport cython
from libc.stdio cimport FILE, fopen, scanf, getline, fclose, printf
from libc.stdlib cimport free, atof, atoi, strtof
from libc.string cimport strcmp, strlen

from libcpp.string cimport string
from libcpp.vector cimport vector

ctypedef float spec_data_fmt

cdef struct spec_peak:
    spec_data_fmt mz
    spec_data_fmt intensity

ctypedef vector[spec_data_fmt] peak_vector

cdef extern from "parse.hpp":
    cdef void test(float a)
    cdef size_t fast_parse(char* f, char* l, spec_data_fmt* mz, spec_data_fmt* intensity)
    cdef int fast_str_compare(const char *ptr0, const char *ptr1, int len)


@cython.boundscheck(False)
cpdef list_to_array(list inp):
    cdef float[:] arr = np.zeros(len(inp), dtype=np.float32)
    cdef Py_ssize_t idx
    for idx in range(len(inp)):
        arr[idx] = inp[idx]
    return np.asarray(arr)

@cython.boundscheck(False)
cpdef vector_to_array(peak_vector inp, Py_ssize_t N):
    cdef:
        float[:] arr = np.zeros(N, dtype=np.float32)
        Py_ssize_t idx

    for idx in range(N):
        arr[idx] = inp[idx]

    return np.asarray(arr)

@cython.boundscheck(False)
cdef inline bint is_begin(char *buf):
    # return fast_str_compare(buf, 'BEGIN', 5)==0
    return strcmp(buf[0:5], 'BEGIN')==0

@cython.boundscheck(False)
cdef inline bint is_end(char *buf):
    # return fast_str_compare(buf, 'END', 3)==0
    return strcmp(buf[0:3], 'END')==0

@cython.boundscheck(False)
cdef inline bint is_invalid(char *buf):
    return buf[0]=='#'

@cython.boundscheck(False)
cdef inline bint is_title(char *buf):
    # return fast_str_compare(buf, 'TITLE', 5)==0
    return strcmp(buf[0:5], 'TITLE')==0

@cython.boundscheck(False)
cdef inline bint is_scans(char *buf):
    # return fast_str_compare(buf, 'SCANS', 5)==0
    return strcmp(buf[0:5], 'SCANS')==0

@cython.boundscheck(False)
cdef inline bint is_rtins(char *buf):
    # return fast_str_compare(buf, 'RTINS', 5)==0
    return strcmp(buf[0:5], 'RTINS')==0

@cython.boundscheck(False)
cdef inline bint is_pepmass(char *buf):
    # return fast_str_compare(buf, 'PEPMA', 5)==0
    return strcmp(buf[0:5], 'PEPMA')==0

@cython.boundscheck(False)
cdef inline bint is_charge(char *buf):
    # return fast_str_compare(buf, 'CHARGE', 6)==0
    return strcmp(buf[0:6], 'CHARGE')==0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list load_mgf_file(filename):
    cdef:
        FILE *fp
        char *line = NULL
        char *end = NULL   

    cdef:
        size_t len, read_len, line_num
        Py_ssize_t peak_i = 0

    cdef:
        # bytes title
        spec_data_fmt rtinsecs, pepmass
        int charge, scans, spec_index
        spec_data_fmt mz_temp, intensity_temp

    cdef:
        peak_vector mz, intensity

    mz.reserve(2000)
    intensity.reserve(2000)

    fp = fopen(filename.encode(), "r")
    filename = filename[filename.rfind('/')+1: filename.rfind('.')]

    if(fp==NULL):
        print("File open failed!")
        sys.exit(0)

    # cdef:
        # char* test_str = "3.2 4.5"
    # peak_i = fast_parse(test_str, test_str+7, &mz_temp, &intensity_temp)
    # printf("parsing %d data\n", peak_i)

    cdef:
        double file_load_time = .0
        double meta_parse_time = .0
        double peak_parse_time = .0
        double list_gen_time = .0

    # total_start = time.time()

    # Start read and parse MGF files
    line_num = 0
    spec_index = 0
    read_spectra_list = []
    while True:
        # start = time.time()
        read_len = getline(&line, &len, fp)
        # file_load_time += time.time()-start

        # if(read_len == -1 or line_num>200):
        if(read_len == -1):
            break

        # printf("Retrieved line of length %zu:\n", strlen(line))
        # printf("%s\n", line[0: read_len-1])
        # start = time.time()

        line_num+=1
        if(is_invalid(line)):
            # printf("# line\n")
            continue
        elif(is_begin(line)):
            peak_i = 0
            charge = -1
            pepmass = -1
            # title = None
            scans = -1 
            rtinsecs = -1
            mz.clear()
            intensity.clear()
            continue
        # elif(is_title(line)):
            # title = line[6:]
            # printf("TITLE is: %s\n", title)
            # continue
        elif(is_scans(line)):
            scans = atoi(line[6:])
            # printf("SCANS is: %d\n", scans)
            continue
        elif(is_rtins(line)):
            rtinsecs = atof(line[12:])
            # printf("RTINS is: %f\n", rtinsecs)
            continue
        elif(is_pepmass(line)):
            pepmass = atof(line[8:read_len-1])
            # printf("PEPMASS is: %f\n", pepmass)
            continue
        elif(is_charge(line)):
            charge = atoi(line[7:read_len-2])
            # printf("CHARGE is: %d\n", charge)
            # continue

            # meta_parse_time += time.time()-start
            
            while True:
                # start = time.time()
                read_len = getline(&line, &len, fp)
                # file_load_time += time.time()-start

                if(is_end(line)):
                    # start = time.time()
                    # printf("collect %d peaks, actual %d peaks\n", peak_i, mz.size())
                    # printf("END OF LINE")

                    read_spectra_list.append([
                        -1, charge, pepmass, 
                        filename, scans, rtinsecs, 
                        vector_to_array(mz, peak_i),
                        vector_to_array(intensity, peak_i)
                        ])

                    spec_index +=1
                    # print(read_spectra_list[0])
                    # list_gen_time += time.time()-start

                    break
                else:
                    # peak_start = time.time()

                    # mz_temp = strtof(line, &end)
                    # intensity_temp = strtof(end, NULL)

                    fast_parse(line, line+read_len, &mz_temp, &intensity_temp)

                    mz.push_back(mz_temp)
                    intensity.push_back(intensity_temp)

                    peak_i+=1
                    # printf("mz=%f, peak=%f\n", mz_temp, intensity_temp)
        
                    # peak_parse_time += time.time()-peak_start

    free(line)
    fclose(fp)
    
    # print("load:{:.6f}\tmeta:{:.6f}\tpeak:{:.6f}\tlist:{:.6f}\ttotal:{:.6f}".format(
        # file_load_time, meta_parse_time, peak_parse_time, list_gen_time,
        # time.time()-total_start))

    return read_spectra_list
