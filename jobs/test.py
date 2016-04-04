import os

from maverick_create import make_exps
from support import my_range, item_generator


exp_group = os.path.basename(__file__).split('.')[-2] # name of current file

# Define hyperparameters!

args = {'-exp-id': my_range(start=0, end=100),
        '-use-pretrained': item_generator([True, False]),
        '-nb-epoch': item_generator(range(50)),
}

# Create maverick job files!

make_exps(exp_group, args, num_exps=2)
