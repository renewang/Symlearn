from concurrent.futures import ALL_COMPLETED
import concurrent.futures
import subprocess
import pathlib
import numpy
import re
import os
import gc

from run_naive_bayes import run_multi_classifiers, data_dir, configure

csvfile = os.path.join(data_dir, 'treebased_phrases.csv')
procfile = os.path.join(data_dir, 'treebased_phrases_vocab.model')


if __name__ == '__main__':
    import sys 
    import joblib

    timeout_ = None
    max_workers_ = 2
    #n_rows_ = 10000
    cvkws, gridkws = configure('multi')

    exp_levels = numpy.logspace(0, 4, 5, base=2, dtype=numpy.int)
    print(numpy.array2string(exp_levels))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_) as executor:
        exec_res = [executor.submit(run_multi_classifiers, csvfile, 'ensemble', 'dumb', 
            i, procfile , cvkws, gridkws, presort=False, n_rows=n_rows_, random_state=None, 
            batch_mode=True, n_jobs=2) for i in exp_levels]
        exec_ens = []
        for future in concurrent.futures.as_completed(exec_res):
            print("ensemble multi-classifier worker status:", future.done(), "test accruacy:", future.result()[-1])
            exec_ens.append(future.result())
    concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
    del exec_res
    gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_) as executor:
        exec_res = [executor.submit(run_multi_classifiers, csvfile, 'boost', 'dumb', 
            i, procfile, cvkws, gridkws, presort=False, n_rows=n_rows_, random_state=None, 
            batch_mode=True, n_jobs=2) for i in exp_levels]
        exec_bts = []
        for future in concurrent.futures.as_completed(exec_res):
            print("ensemble multi-classifier worker status:", future.done(), "test accruacy:", future.result()[-1])
            exec_bts.append(future.result())
    concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
    del exec_res
    gc.collect()
    joblib.dump([exec_ens, exec_bts], os.path.join(data_dir, 'multi_classifier_exps.pkl'))