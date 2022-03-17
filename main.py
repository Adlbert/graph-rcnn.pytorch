"""
Implementation of ECCV 2018 paper "Graph R-CNN for Scene Graph Generation".
Author: Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, Devi Parikh
Contact: jw2yang@gatech.edu
"""

import os
import pprint
import argparse
import numpy as np
import torch
import datetime
import logging
import time

import cv2 as cv
from lib.scene_parser.rcnn.structures.image_list import to_image_list
from imagefactazlib.messenger import messengerservice
from imagefactazlib.database import dbservice
from imagefactazlib.util import util as utilservice


import json
with open('config.json') as c:
    imgfactconfig = json.load(c)

from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger

def train(cfg, args):
    """
    train scene graph generation model
    """
    arguments = {}
    arguments["iteration"] = 0
    model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.train()
    return model

def test(cfg, args, model=None):
    """
    test scene graph generation model
    """
    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.test(visualize=args.visualize)

def apply(cfg, args, model=None):
    """
    apply scene graph generation model
    """
    img_id = args.id

    img = cv.imread('/mnt/e/Uni/Master/repo/graph-rcnn.pytorch/dataset/test/' + args.image)
    img = cv.resize(img, (1024,768))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img - np.array(cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3) # normalize
    img = np.transpose(img, (2, 0, 1)) # hwc to chw
    img = torch.from_numpy(img).float()

    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, arguments, args.local_rank, args.distributed)

    model.apply(img, img_id, visualize=args.visualize)

def run(cfg, args, model=None):
    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, arguments, args.local_rank, args.distributed)
    logger = logging.getLogger("scene_graph_generation.azure.messenger")
    logger.info("Started messenger")
    while(True):
        recevied, img_dto = messengerservice.receive(imgfactconfig)
        if (recevied):
            logger.info("Received: {}".format(img_dto.imageId))
            img_o = dbservice.getImage(imgfactconfig, img_dto)
            img_cv = utilservice.load(img_o)
            img = cv.resize(img_cv, (1024,768))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img - np.array(cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3) # normalize
            img = np.transpose(img, (2, 0, 1)) # hwc to chw
            img = torch.from_numpy(img).float()
            json_graph, labelDict, edgeDict = model.run(img, img_dto.imageId)
            img_o.graph = json.dumps(json_graph)
            img_o.labelDict = json.dumps(labelDict)
            img_o.edgeDict = json.dumps(edgeDict)
            dbservice.updateImage(imgfactconfig, img_o)
        else:
            logger.info("Nothing received. Waiting ...")
        time.sleep(2) # Sleep for 2 seconds




def main():
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Scene Graph Generation")
    parser.add_argument("--config-file", default="configs/baseline_res101.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--apply", action='store_true')
    parser.add_argument("--image", type=str, default='test.jpg')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--instance", type=int, default=-1)
    parser.add_argument("--use_freq_prior", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--algorithm", type=str, default='sg_baseline')
    parser.add_argument("--service", action='store_true')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.resume = args.resume
    cfg.instance = args.instance
    cfg.inference = args.inference
    cfg.MODEL.USE_FREQ_PRIOR = args.use_freq_prior
    cfg.MODEL.ALGORITHM = args.algorithm
    if args.batchsize > 0:
        cfg.DATASET.TRAIN_BATCH_SIZE = args.batchsize
    if args.session > 0:
        cfg.MODEL.SESSION = str(args.session)
    # cfg.freeze()

    if not os.path.exists("logs") and get_rank() == 0:
        os.mkdir("logs")
    logger = setup_logger("scene_graph_generation", "logs", get_rank(),
        filename="{}_{}.txt".format(args.algorithm, get_timestamp()))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    output_config_path = os.path.join("logs", 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    if not args.inference and not args.apply and not args.service:
        model = train(cfg, args)
    elif not args.apply and not args.service:
        test(cfg, args)
    elif not args.service:
        apply(cfg,args)
    else:
        run(cfg,args)

if __name__ == "__main__":
    main()
