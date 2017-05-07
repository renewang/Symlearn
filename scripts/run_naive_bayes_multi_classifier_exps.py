from concurrent.futures import ALL_COMPLETED
import concurrent.futures
import subprocess
import pathlib
import numpy
import re
import os
import gc

from run_naive_bayes import run_multi_classifiers, data_dir

csvfile = os.path.join(data_dir, 'treebased_phrases.csv')
cvkws = {'n_splits': 10, 'random_state': None}
gridkws = {'scoring': 'accuracy', 'verbose': 0, 'n_jobs': 1}

if __name__ == '__main__':
    import sys 
    import joblib

    timeout_ = None
    exp_levels = numpy.logspace(0, 2, 3, base=4, dtype=numpy.int)
    print(numpy.array2string(exp_levels))
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        exec_res = [executor.submit(run_multi_classifiers, csvfile, cvkws, gridkws,
                    predefine=True, n_rows=-1, use_ensemble=False,
                    random_state=None, max_level=i, batch_mode=True, n_jobs=1) for i in exp_levels]
        exec_dec = []
        for future in concurrent.futures.as_completed(exec_dec):
            print("decision tree multi-classifier worker status:", future.done(), "test accruacy:", future.result()[-1])
            exec_dec.append(future.result())
    concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
    del exec_res
    gc.collect()
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        exec_res = [executor.submit(run_multi_classifiers, csvfile, cvkws, gridkws,
                    predefine=False, n_rows=-1, use_ensemble=True,
                    random_state=None, max_level=i, batch_mode=True, n_jobs=1) for i in exp_levels]
        exec_ens = []
        for future in concurrent.futures.as_completed(exec_res):
            print("ensemble multi-classifier worker status:", future.done(), "test accruacy:", future.result()[-1])
            exec_ens.append(future.result())
    concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
    joblib.dump((exec_dec, exec_ens), os.path.join(data_dir, 'multi_classifier_exps.pkl'))