import random
import os
import torch

import numpy as np

from utils.parse_args import create_arg_parser
from evaluation_scripts.metrics import get_metrics
from mail import send_mail
# from evaluation_scripts.single_plot import get_single_plot

# TODO: IMPLEMENT ALL EVAL LOGIC
if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.in_chans = 16
    args.out_chans = 16

    try:
        get_metrics(args)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(e)
        send_mail("TESTING FAILED", "See terminal for falure cause.")

