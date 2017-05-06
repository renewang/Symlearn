from concurrent.futures import ALL_COMPLETED
import concurrent.futures
import subprocess
import pathlib
import numpy
import re
import os
import gc

from run_naive_bayes import run_single_classifier, data_dir
csvfile = os.path.join(data_dir, 'treebased_phrases.csv')
cvkws = {'n_splits': 10, 'random_state': None}
gridkws = {'scoring': 'accuracy', 'verbose': 0}

if __name__ == '__main__':
    import sys
    import joblib

    num_exps = int(sys.argv[1])
    timeout_ = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        exec_res = [executor.submit(run_single_classifier, csvfile, cvkws, gridkws,
                    predefine=True, n_rows=-1, use_ensemble=False,
                    random_state=os.getpid() % (i + 1), max_level=16, batch_mode=True) for i in range(num_exps)]
        exec_dec = []
        for future in concurrent.futures.as_completed(exec_res, timeout=timeout_):
            print("decision tree worker status:", future.done(), "test accruacy:", future.result()[-1])
            exec_dec.append(future.result())
    
    concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
    del exec_res
    gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        exec_res = [executor.submit(run_single_classifier, csvfile, cvkws, gridkws,
                    predefine=True, n_rows=-1, use_ensemble=True,
                    random_state=os.getpid() % (i + 1), max_level=16, batch_mode=True) for i in range(num_exps)]
        exec_ens = []
        for future in concurrent.futures.as_completed(exec_res):
            print("ensemble worker status:", future.done(), "test accruacy:", future.result()[-1])
            exec_ens.append(future.result())

    concurrent.futures.wait(exec_res, timeout=timeout_, return_when=ALL_COMPLETED)
    joblib.dump((exec_dec, exec_ens), os.path.join(data_dir, 'single_classifier_exps.pkl'))