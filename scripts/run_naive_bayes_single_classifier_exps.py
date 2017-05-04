import concurrent.futures
import subprocess
import pathlib
import numpy
import re
import os

from run_naive_bayes import run_single_classifier, data_dir
csvfile = os.path.join(data_dir, 'treebased_phrases.csv')
cvkws = {'n_splits': 10, 'random_state': None}
gridkws = {'scoring': 'accuracy', 'verbose': 0}

if __name__ == '__main__':
    import sys
    import joblib
    num_exps = int(sys.argv[1])
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        exec_dec = [executor.submit(run_single_classifier, csvfile, cvkws, gridkws,
                    predefine=True, n_rows=-1, use_ensemble=False,
                    random_state=None, max_level=16, batch_mode=True) for i in range(num_exps)]
        for future in concurrent.futures.as_completed(exec_dec):
            print("decision tree worker status:", future.done(), "test accruacy:", future.result()[-1])
            
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        concurrent.futures.wait(exec_dec)
        exec_ens = [executor.submit(run_single_classifier, csvfile, cvkws, gridkws,
                    predefine=True, n_rows=-1, use_ensemble=True,
                    random_state=None, max_level=16, batch_mode=True) for i in range(num_exps)]
        for future in concurrent.futures.as_completed(exec_ens):
            print("ensemble worker status:", future.done(), "test accruacy:", future.result()[-1])
    joblib.dump((exec_dec, exec_ens), os.path.join(data_dir, 'single_classifier_exps.pkl'))