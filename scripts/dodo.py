import shlex
import run_naive_bayes


def task_profile():
    """
    create task for profiling run_naive_bayes code
    """
    import cProfile

    def profile(targets):
        # need to pop off sys.argv
        import sys
        sys.argv[1:] = targets[0].split()

        cProfile.runctx('main()', run_naive_bayes.__dict__, 
            run_naive_bayes.__dict__, filename=targets[1])

    run_cmd = 'multi --classifier tree --predictor gnb -n 50000 -t 2'

    return {
            'actions': [profile],
            'targets': [run_cmd, 'run_naive_bayse.prof']
           }

def task_single():
    """
    create task for multiple single experiments to run
    """
    for cls in ['tree', 'ensemble', 'boost']:
            cmd = 'echo python run_naive_bayes.py single --presort --classifier %s --predictor dumb' % (cls)
            yield {
                    'name' : [cls],
                    'actions': [shlex.split(cmd)],
                    'verbosity': 2,
            }



if __name__ == '__main__':
	import doit
	doit.run(globals())
