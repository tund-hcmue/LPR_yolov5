import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import random as rd
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from termcolor import colored
import threading
import numpy as np
from tensorflow.keras.models import load_model
import math
# from google.colab.patches import cv2_imshow
from queue import Queue
# import pytesseract

class LicensePlate():
    
    def __init__(self, source = 'data/images/',
                 weights_lp = 'models_pre/lp_detect.pt', # License plate detect model
                 weights_lpr = 'models_pre/lpr_detect.pt', # character detect model
                 img_size = 480, imglpr_size = 96, name = 'exp', 
                 project = 'runs/detect', device = '', 
                 conf_thres = float(0.25), conf_thres01 = float(0.25), 
                 iou_thres = float(0.45), save_txt = True, 
                 view_img = False, save_conf = False, 
                 classes = None, agnostic_nms = False, 
                 augment = False, update = False, exist_ok = False, 
                 save_img=False, nosave = True):
        self.source = source
        self.weights_lp = weights_lp
        self.weights_lpr = weights_lpr
        self.view_img = view_img
        self.save_txt = save_txt
        self.img_size = img_size
        self.imglpr_size = imglpr_size
        self.name = name
        self.project = project
        self.device = device
        self.conf_thres = conf_thres
        self.conf_thres01 = conf_thres01
        self.iou_thres = iou_thres
        self.exist_ok = exist_ok
        self.augment = augment
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.save_conf = save_conf
        self.save_img=save_img
        self.update = update
        self.nosave = nosave
        self.img_plate = Queue()
        self.lst_number = Queue()
        self.plate_num = ''
        self.img_number = None
        
        self.model_char = load_model("models_pre/char_classifier.h5") # character classifier model

        t0 = time.time()
        #start Thread
        self.t1 = threading.Thread(name = "t1", target=self.detect, args = ())
        self.t2 = threading.Thread(name = "t2", target=self.recognition, args = ())
        self.t3 = threading.Thread(name = "t3", target=self.read_char, args = ())
        
        self.t1.start()
        self.t2.start()
        self.t3.start()
        
        self.t1.join()
        self.t2.join()
        self.t3.join()
        print(f'Done. ({time.time() - t0:.3f}s)')
    def crop_lpr(self, xywh, im0):     #detect and crop character in plate
        width = im0.shape[1]
        height = im0.shape[0]
        x = xywh[0]
        y = xywh[1]
        w = xywh[2]
        h = xywh[3]
        xmin = int((x - w/2)*width)
        ymin = int((y - h/2)*height)
        xmax = int(xmin + (w*width))
        ymax = int(ymin + (h*height))
        
        cropped = im0[ymin-1:ymax+1, xmin-2:xmax+2]
                
        return cropped
    def crop_img(self, xywh, im0):     #detect and crop plate
        width = im0.shape[1]
        height = im0.shape[0]
        x = xywh[0]
        y = xywh[1]
        w = xywh[2]
        h = xywh[3]
        xmin = int((x - w/2)*width)
        ymin = int((y - h/2)*height)
        xmax = int(xmin + (w*width))
        ymax = int(ymin + (h*height))
        
        cropped = im0[ymin:ymax, xmin:xmax]
                
        return cropped
    def rotate_image(self, image):
        # lines = []
        h, w = image.shape[:2]

        img = cv2.medianBlur(image, 3)

        edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
        lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
        angle = 0.0
        cnt = 0
        if lines is not None:
            for [x1, y1, x2, y2] in lines[0]:
                ang = np.arctan2(y2 - y1, x2 - x1)
                if math.fabs(ang) <= 30: # excluding extreme rotations
                    angle += ang
                    cnt += 1

            if cnt == 0:
                return 0.0
            angle = ((angle / cnt)*180/math.pi)
        if abs(angle) > float(30):
            pos = (True if angle < 0 else False)
            angle = 90 - abs(angle)
            angle = (angle*(-1) if pos else angle)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
    def BGR_to_thr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize( gray, (28, 28), 0, 0,interpolation = cv2.INTER_CUBIC)
        
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        roi = cv2.medianBlur(thresh, 3)
        return roi
    def read_char(self):
        char_list =  '0123456789ABCDEFGHKLMNPSTUVXYZ'
        while 1:
            plate_num = ''
            lst_number = self.lst_number.get()
            
            if lst_number is None:
                break
            else:
                print(len(lst_number))
                if (len(lst_number) >5):
                    for lpr in lst_number:
                        # x = image.img_to_array(img)
                        backtorgb = cv2.cvtColor(lpr,cv2.COLOR_GRAY2RGB)
                        x = np.array(backtorgb)
                        x = np.expand_dims(x, axis=0)

                        images = np.vstack([x])
                        classes = self.model_char.predict_classes(images)
                        
                        lp = char_list[classes[0]]
                                        
                        plate_num += lp
                else:
                    plate_num += 'None'
            print(colored("\n Number Plate :", "red"), plate_num)
    def recognition(self):
        source, weights, view_img, save_txt, imgsz = self.source, self.weights_lpr, self.view_img, self.save_txt, self.imglpr_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load modelim0s
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model striderecognition
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        while 1:
            detect_lp = self.img_plate.get()
            if (detect_lp is None):
                break
            for lp in detect_lp:
                lp = cv2.resize(lp, (190, 140), interpolation = cv2.INTER_CUBIC)
                
                # for path, img, im0s, vid_cap in dataset:
                img = self.letterbox(lp, 192, stride=32)[0]                      

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device)
                
                img = img.half() if half else img.float()  # uint8 to fp16/32
                
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                im0s = lp
                
                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=self.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres01, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
                t2 = time_synchronized()
                
                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)
                    
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    im0 = im0s
                    # det = det[det[:,3].sort()[1]]
                    lst = det.tolist()
                    
                    sortt = sorted(lst, key = lambda x: x[1], reverse=True)
                    index = math.ceil(len(lst)/float(2))
                    

                    sortt1 = sorted(sortt[:index], key = lambda x: x[3])
                    sortt2 = sorted(sortt[index:], key = lambda x: x[3])

                    det = torch.tensor(sortt1+sortt2)
                    
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        
                        # Write results
                        lst_number = []
                        plate_num = ''
                        for *xyxy, conf, cls in reversed(det):
                            
                            if (float(f' {conf:.2f}') > 0.5):
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                
                                x = xywh[0]
                                y = xywh[1]
                                w = xywh[2]
                                h = xywh[3]
                                try:
                                    img_number = self.crop_lpr(xywh, im0)
                                    img_number = self.BGR_to_thr(img_number)
                                except:
                                    pass

                                lst_number.append(img_number)
                                
                                label = f'. {conf:.2f}'
                                plot_one_box(xyxy, lp, label=label, color=colors[int(cls)], line_thickness=1)
                            else:
                                pass
                        self.lst_number.put(lst_number)
                        
                
        self.lst_number.put(None)

    def detect(self):
        source, weights, view_img, save_txt, imgsz = self.source, self.weights_lp, self.view_img, self.save_txt, self.img_size
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    lst_plate = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            
                            x = xywh[0]
                            y = xywh[1]
                            w = xywh[2]
                            h = xywh[3]

                            img_plate = self.crop_img(xywh, im0)
                            img_rotate = self.rotate_image(img_plate)
                            # cv2.imwrite(save_path, img_rotate)
                            lst_plate.append(img_rotate)

                        if save_img or view_img:  # Add bbox to image
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            label = f'Plate {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    self.img_plate.put(lst_plate)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        # print("save_path")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # videoyolov5/runs/train/exp/
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        self.img_plate.put(None)

t = LicensePlate()