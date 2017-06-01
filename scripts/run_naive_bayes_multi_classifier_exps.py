from concurrent.futures import ALL_COMPLETED
import concurrent.futures
import contextlib
import subprocess
import pathlib
import pickle
import numpy
import mmap
import os
import gc

from run_naive_bayes import run_multi_classifiers, data_dir, configure

csvfile = os.path.join(data_dir, 'treebased_phrases.csv')
procfile = os.path.join(data_dir, 'treebased_phrases_vocab.model')


if __name__ == '__main__':
    import sys 
    import joblib

    if len(sys.argv) > 1:
        n_rows_ = int(sys.argv[1])
        print('only %d records will be read in' % n_rows_)
    else:
        n_rows_ = -1
        print('all records will be read in')

    timeout_ = None
    max_workers_ = 2
    cvkws, gridkws = configure('multi')
    exp_levels = numpy.logspace(0, 2, 3, base=4, dtype=numpy.int)
    print(numpy.array2string(exp_levels))

    with open("multi_classifier_exps.pkl", "wb") as fp:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_) as executor:
            exec_res = [executor.submit(run_multi_classifiers, csvfile, 'ensemble', 'dumb', 
                    i, procfile , cvkws, gridkws, presort=False, n_rows=n_rows_, random_state=None, 
                    batch_mode=True, n_jobs=2) for i in exp_levels]
            numpy.savez(fp, *(future.result() for n_run, future in enumerate(
                concurrent.futures.as_completed(exec_res))))
            concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
        for future in exec_res:
            print("ensemble multi-classifier worker status:", future.done(), "test accruacy:", future.result()[-1])
        del exec_res
        gc.collect()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_) as executor:
            exec_res = [executor.submit(run_multi_classifiers, csvfile, 'boost', 'dumb', 
                    i, procfile, cvkws, gridkws, presort=False, n_rows=n_rows_, random_state=None, 
                    batch_mode=True, n_jobs=2) for i in exp_levels]
            numpy.savez(fp, *(future.result() for n_run, future in enumerate(
                concurrent.futures.as_completed(exec_res))))
            concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
        for future in exec_res:
            print("boost multi-classifier worker status:", future.done(), "test accruacy:", future.result()[-1])
        del exec_res
        gc.collect()