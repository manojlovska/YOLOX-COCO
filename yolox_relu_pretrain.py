#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.nn as nn
from loguru import logger

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 80
        self.max_epoch = 180
        self.data_num_workers = 1
        self.input_size = (640, 640)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (640, 640)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = 'datasets/COCO'
        # modify 'silu' to 'relu' for deployment on DPU
        self.act = 'relu'

        self.thresh_lr_scale = 10

        self.device = torch.device('cuda')

        torch.backends.cudnn.enabled = False

        logger.info("GPU MEMORY AVAILABLE: " + str(torch.cuda.mem_get_info()))


    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models.yolox import YOLOX
            from yolox.models.yolo_pafpn_deploy import YOLOPAFPN
            from yolox.models.yolo_head import YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model


    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if isinstance(v, nn.Identity): # during nndct's QAT, 'bn' is merged to 'conv' and replaced by Identity
                        continue
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            threshold = [ 
                param for name, param in self.model.named_parameters()
                if 'threshold' in name
            ]
            # print("Threshold params: ")
            # [print(name) for name, param in self.model.named_parameters() if 'threshold' in name]
            q_param_group = { 
                'params': threshold,
                'lr': lr * self.thresh_lr_scale,
                'name': 'threshold'
            } 

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group(q_param_group)
            self.optimizer = optimizer

        return self.optimizer


    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators.coco_evaluator import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(model, model, is_distributed, half, return_outputs=return_outputs)
