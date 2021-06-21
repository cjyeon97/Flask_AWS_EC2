"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# -*- coding: utf-8 -*-
import sys
import time
import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import json
import zipfile

from . import craft_utils
from . import imgproc
from . import file_utils
from .craft import CRAFT

from collections import OrderedDict

from flask import Flask, redirect
from flask_restful import Api, Resource, reqparse

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# 크기 Top3 구하기
def top3(boxes):
    box_size = []
    top3_size = []
    top3_idx = []
    top3_boxes = []
    
    # box 크기 저장
    for box in enumerate(boxes):
        box = np.array(box, dtype=object)       # (idx, 좌표, dtype)
        # 좌표 행렬
        poly = box[1]  
        w = poly[-1, -1] - poly[0, 1]
        h = poly[1, 0] - poly[0, 0]
        box_size.append(w * h)

    box_size_temp = copy.deepcopy(box_size)
    
    if len(box_size) >= 3:
        # top3_size 저장
        for i in range(3):
            max = np.max(box_size_temp)
            top3_size.append(max)
            max_idx = box_size_temp.index(max)
            box_size_temp.pop(max_idx)

        # top3_idx 저장
        for i in range(3):
            idx = box_size.index(top3_size[i])
            top3_idx.append(idx)
        
        # boxes = boxes.reshape(15,8)
        # boxes에서 top3만 추출
        for i in range(3):
            top3_boxes.append(boxes[top3_idx[i]])
    
    else:
        # top1_size 저장
        max = np.max(box_size_temp)
        top3_size.append(max)
        max_idx = box_size_temp.index(max)

        # boxes에서 top1만 추출
        top3_boxes.append(boxes[max_idx])

    return top3_boxes

def detection_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    #### 여기서 모양이 어떤지?? -> 타입 나옴
    # print(boxes)
    
    ## 큰 box 3개 추출
    if len(boxes) > 0:
        boxes = top3(boxes)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # for k in range(len(polys)):
    #     if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

class Detection(Resource):
    def get(self):
        parser = argparse.ArgumentParser(description='CRAFT Text Detection')
        parser.add_argument('--trained_model', default='STD/craft_mlt_25k.pth', type=str, help='pretrained model')
        parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
        parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
        parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
        parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
        parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
        parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
        parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
        parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
        # parser.add_argument('--image_path', default='image/test01.jpg', type=str, help='folder path to input images')
        parser.add_argument('--upload_folder', default='flask_upload/', type=str, help='folder path to input images')

        args = parser.parse_args()

        """ For test images in a folder """
        image_list, _, _ = file_utils.get_files(args.upload_folder)

        # STD 결과이미지 저장 위치
        result_folder = 'detection_result/'
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        # load net
        net = CRAFT()     # initialize

        # print('Loading weights from checkpoint (' + args.trained_model + ')')
        if args.cuda:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
            # net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))

        if args.cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        t = time.time()

        # load data
        # 이미지 폴더 내 모두
        for k, image_path in enumerate(image_list):
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = detection_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)

            # STD 부분만 저장
            # print(type(bboxes)) => numpy.ndarray
            filename, file_ext = os.path.splitext(os.path.basename(image_path))

            if len(bboxes) == 0:
                print ('NO BOX')
                continue

            for b, box in enumerate(bboxes):

                box = box.astype(int)
                min1 = np.amin(box, axis=0)
                max2 = np.amax(box, axis=0)
                x1, y1 = min1[0], min1[1]
                x2, y2 = max2[0], max2[1]

                box = np.reshape(box, (4,1,2))

                mask1 = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.drawContours(mask1, [box], -1, 255, -1, cv2.LINE_AA)
                crop =cv2.bitwise_and(image, image, mask=mask1)
                crop = crop[:,:,::-1]
                img_trim = crop[y1:y2, x1:x2]

                if img_trim.size!=0:
                    cv2.imwrite(result_folder+filename+'_'+str(b)+'.jpg', img_trim)

        # UPLOAD_FOLDER내 파일 삭제
        for file in os.scandir(args.upload_folder):
            os.remove(file.path)

        return redirect('/recognition') 