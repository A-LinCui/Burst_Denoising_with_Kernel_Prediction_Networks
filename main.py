import os
import yaml
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import extorch
import extorch.utils as utils

from kpn import KPN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type = int, default = 0, help = "used GPU id")
    parser.add_argument("--train-dir", type = str, default = None, help = "path to save the ckpt")
    parser.add_argument("--seed", type = int, default = None)
    parser.add_argument("--only-eval", action = "store_true", default = True)
    parser.add_argument("--load", type = str, default = None)
    args = parser.parse_args()

    LOGGER = utils.getLogger("Main")

    if args.train_dir and not args.only_eval:
        LOGGER.info("Save checkpoint at {}".format(os.path.abspath(args.train_dir)))
        utils.makedir(args.train_dir, remove = True)
        LOGGER.addFile(os.path.join(args.train_dir, "train.log"))
        writer = SummaryWriter(os.path.join(args.train_dir, "tensorboard"))
        with open(os.path.join(args.train_dir, "config.yaml"), "w") as wf:
            yaml.dump(cfg, wf)

    LOGGER.info("Load configuration from {}".format(os.path.abspath(args.cfg_file)))
    with open(args.cfg_file, "r") as rf:
        cfg = yaml.load(rf, Loader = yaml.FullLoader)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:{}".format(args.gpu))
        LOGGER.info("Used GPU: {}".format(args.gpu))
    else:
        DEVICE = torch.device("cpu")
        LOGGER.info("No GPU available. Use CPU instead.")

    if args.seed:
        utils.set_seed(args.seed)
        LOGGER.info("Set seed: {}".format(args.seed))

    model = KPN(**model_cfg)
    if args.load:
        LOGGER.info("Load checkpoint from {}".format(os.path.abspath(args.load)))
        ckpt = torch.load(args.load, map_location = torch.device("cpu"))
        model.load_state_dict(ckpt)
    model.to(DEVICE)

    # criterion = 
    
    if not args.only_eval:
        optimizer = optim.Adam(model.parameters(), lr = cfg["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, **cfg["scheduler_cfg"])