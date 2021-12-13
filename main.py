import os
import yaml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import extorch
import extorch.utils as utils

from kpn import KPN
from dataset import Adobe5K, KPNDataset
from loss import KPNLoss
from stats import cal_ssim, cal_psnr


def valid(net, trainloader, device, epoch, report_every, logger):
    psnrs = utils.AverageMeter()
    ssims = utils.AverageMeter()
    
    net.eval()
    for step, (burst, target, white_level) in enumerate(trainloader):
        burst = burst.to(device)
        target = target.to(device)
        white_level = white_level.to(device)

        mean_output = net(burst, white_level)

        n = burst.size(0)
        ssims.update(cal_ssim(mean_output, target), n)
        psnrs.update(cal_psnr(mean_output, target), n)

        if (step + 1) % report_every == 0:
            logger.info("Epoch {} valid {} / {} {:.3f}; {:.3f}".format(
                epoch, step + 1, len(trainloader), psnrs.avg, ssims.avg))
    
    return psnrs.avg, ssims.avg
 

def train_epoch(net, trainloader, device, optimizer, criterion, epoch, report_every, logger):
    objs = utils.AverageMeter()
    psnrs = utils.AverageMeter()
    ssims = utils.AverageMeter()
    
    net.train()
    for step, (burst, target, white_level) in enumerate(trainloader):
        burst = burst.to(device)
        target = target.to(device)
        white_level = white_level.to(device)

        optimizer.zero_grad()
        mean_output, output = net(burst, white_level)

        if isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
            loss = criterion(mean_output, target)
        else:
            loss = criterion(sRGBTransfer(output), sRGBTransfer(mean_output), sRGBTransfer(target), epoch)
        loss.backward()
        optimizer.step()

        n = burst.size(0)
        objs.update(loss.item(), n)
        ssims.update(cal_ssim(mean_output, target), n)
        psnrs.update(cal_psnr(mean_output, target), n)
        del loss

        if (step + 1) % report_every == 0:
            logger.info("Epoch {} train {} / {} {:.3f}; {:.3f}; {:.3f}".format(
                epoch, step + 1, len(trainloader), objs.avg, psnrs.avg, ssims.avg))
    
    return objs.avg, psnrs.avg, ssims.avg
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type = int, default = 0, help = "used GPU id")
    parser.add_argument("--data-dir", type = str, required = True, help = "path of the data")
    parser.add_argument("--train-dir", type = str, default = None, help = "path to save the ckpt")
    parser.add_argument("--seed", type = int, default = None)
    parser.add_argument("--report-every", type = int, default = 50)
    parser.add_argument("--save-every", type = int, default = 20)
    parser.add_argument("--only-eval", action = "store_true", default = False)
    parser.add_argument("--load", type = str, default = None)
    args = parser.parse_args()

    LOGGER = utils.getLogger("Main")

    LOGGER.info("Load configuration from {}".format(os.path.abspath(args.cfg_file)))
    with open(args.cfg_file, "r") as rf:
        cfg = yaml.load(rf, Loader = yaml.FullLoader)

    if args.train_dir and not args.only_eval:
        LOGGER.info("Save checkpoint at {}".format(os.path.abspath(args.train_dir)))
        utils.makedir(args.train_dir, remove = True)
        LOGGER.addFile(os.path.join(args.train_dir, "train.log"))
        writer = SummaryWriter(os.path.join(args.train_dir, "tensorboard"))
        with open(os.path.join(args.train_dir, "config.yaml"), "w") as wf:
            yaml.dump(cfg, wf)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:{}".format(args.gpu))
        LOGGER.info("Used GPU: {}".format(args.gpu))
    else:
        DEVICE = torch.device("cpu")
        LOGGER.info("No GPU available. Use CPU instead.")

    if args.seed:
        utils.set_seed(args.seed)
        LOGGER.info("Set seed: {}".format(args.seed))

    model = KPN(**cfg["model_cfg"])
    if args.load:
        LOGGER.info("Load checkpoint from {}".format(os.path.abspath(args.load)))
        ckpt = torch.load(args.load, map_location = torch.device("cpu"))
        model.load_state_dict(ckpt)
    model.to(DEVICE)

    datasets = KPNDataset(data_dir = args.data_dir, base_dataset_cls = Adobe5K, **cfg["dataset_cfg"])
    train_loader = data.DataLoader(dataset = datasets.splits["train"], \
            batch_size = cfg["batch_size"], num_workers = cfg["num_workers"], shuffle = True)
    val_loader = data.DataLoader(dataset = datasets.splits["test"], \
            batch_size = cfg["batch_size"], num_workers = cfg["num_workers"], shuffle = False)

    if cfg["criterion_type"] == "L1Loss":
        criterion = nn.L1Loss()
    elif cfg["criterion_type"] == "L2Loss":
        criterion = nn.MSELoss()
    elif cfg["criterion_type"] == "KPNLoss":
        criterion = KPNLoss(**cfg["criterion_cfg"])
   
    if args.only_eval:
        val_psnr, val_ssim = valid(
                model, val_loader, DEVICE, 0, args.report_every, LOGGER)
        LOGGER.info("Valid: PSNR: {:.4f}; SSIM: {:.4f}".format(val_psnr, val_ssim))

    optimizer = optim.Adam(model.parameters(), lr = cfg["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, **cfg["scheduler_cfg"])

    for epoch in range(1, cfg["epochs"] + 1):
        LOGGER.info("Epoch {} lr {:.5f}".format(epoch, optimizer.param_groups[0]["lr"]))

        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, DEVICE, optimizer, criterion, epoch, args.report_every, LOGGER)
        LOGGER.info("Epoch {}: Train: obj: {:.5f}; PSNR: {:.4f}; SSIM: {:.4f}".format(
            epoch, train_loss, train_psnr, train_ssim))

        with torch.no_grad():
            val_psnr, val_ssim = valid(
                model, val_loader, DEVICE, epoch, args.report_every, LOGGER)
            LOGGER.info("Epoch {}: Valid: PSNR: {:.4f}; SSIM: {:.4f}".format(
                epoch, val_psnr, val_ssim))

        if args.train_dir:
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/PSNR", train_psnr, epoch)
            writer.add_scalar("Train/SSIM", train_ssim, epoch)
            writer.add_scalar("Test/PSNR", val_psnr, epoch)
            writer.add_scalar("Test/SSIM", val_ssim, epoch)
        
            if epoch % args.save_every == 0:
                save_path = os.path.join(args.train_dir, "ckpt_{}.pt".format(epoch))
                torch.save(model.state_dict(), save_path)
                LOGGER.info("Save checkpoint to {}".format(save_path))

        scheduler.step()

    if args.train_dir:
        save_path = os.path.join(args.train_dir, "final.pt")
        torch.save(model.state_dict(), save_path)
        LOGGER.info("Save checkpoint to {}".format(save_path))
