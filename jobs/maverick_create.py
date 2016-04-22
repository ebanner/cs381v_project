import itertools

from collections import OrderedDict

from IPython import get_ipython

from support import my_range, item_generator


def args_generator(args, num_exps=32, grid_search=False):
    """Generates list of tuples corresponding to args hyperparam dict

    Parameters
    ----------
    args : dict of hyperparam values

        e.g. {p1: g1(), p2: g2(), ..., pn: gn()},

            where pi is the i'th parameter and gi is the generator which
            generates a random value for pi.

    """
    od = OrderedDict(args)

    if grid_search:
        keys = [param for param in od]

        for vals in itertools.product(*[value for param, value in od.items()]):
            yield zip(keys, [str(val) for val in vals])
    else:
        for _ in range(num_exps):
            args_setting = [(pname, str(next(pvalue))) for pname, pvalue in od.items()]

            yield args_setting

def make_exp(exp_group, args, exp_name, grid_search=False):
    """Perform setup work for a maverick experiment
    
    Parameters
    ----------
    exp_group : name of this experiment group
    args : list of (arg_str, arg_val) tuples

    1. Remove any output from a previous experiment with the same name
    2. Do the same thing for weights
    3. Create the maverick job file
    
    """
    equalized_args = ['='.join(tup) for tup in args]
    stripped_args = [arg.lstrip('-') for arg in equalized_args]
    # exp_name = '+'.join(stripped_args)
    
    unrolled_args = [arg for arg_tup in args for arg in arg_tup]
    arg_str = ' '.join(unrolled_args) + ' -exp-group ' + exp_group

    get_ipython().system(u'mkdir -p ../output/$exp_group/$exp_name')
    get_ipython().system(u'mkdir -p ../weights/$exp_group/')
    
    get_ipython().system(u"sed 's/ARGUMENTS/$arg_str/g' job_template > /tmp/tmp1")
    get_ipython().system(u"sed 's/EXP_GROUP/$exp_group/g' /tmp/tmp1 > /tmp/tmp2")
    get_ipython().system(u"sed 's/EXPERIMENT/$exp_name/g' /tmp/tmp2 > /tmp/tmp3")
    
    get_ipython().system(u'mkdir -p exps/$exp_group')
    get_ipython().system(u'cp /tmp/tmp3 exps/$exp_group/$exp_name')

    get_ipython().system(u"rm /tmp/tmp1 /tmp/tmp2 /tmp/tmp3")
    
def make_exps(exp_group, args, num_exps=32, grid_search=False):
    """Wrapper around make_exp()
    
    Call make_exp() with `num_exps` number of experiments.
    
    """
    args_list = list(args_generator(args, num_exps, grid_search))

    # Remove exp_group directories!
    get_ipython().system(u'rm -rf exps/$exp_group')
    get_ipython().system(u'rm -rf ../output/$exp_group')
    get_ipython().system(u'rm -rf ../weights/$exp_group')

    for i, args_setting in enumerate(args_list):
        make_exp(exp_group, args_setting, exp_name=i)


if __name__ == '__main__':
    # Example experiment group!

    from support import my_range, item_generator

    args = {'-exp-id': my_range(start=0, end=100),
            '-use-pretrained': item_generator([True, False]),
            '-nb-epoch': item_generator(range(50)),
    }

    make_exps(exp_group='test', args=args)
