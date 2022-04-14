import cv2
import numpy as np
import datetime, time

import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer, set_logging, increment_dir
from utils.torch_utils import select_device, load_classifier, time_synchronized

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import ipywidgets as widgets

imgsz = 640

my_confidence 		= 0.80 # 0.25
my_threshold  		= 0.45 # 0.45
my_filterclasses 	= None
my_weight					= '/content/CoinCounter/yolov5/weights/coin_v1-9_last.pt'

device = select_device('')
print('>> device',device.type)

# Load model
model = attempt_load(my_weight, map_location=device)	# load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())		# check img_size

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
colors = [
	(232, 182, 0), # 5Baht
	(0, 204, 255),	# 1Baht
	(69, 77, 246),	# 10Baht
	(51, 136, 222),	# 2Baht
	(222, 51, 188),	# .50Baht
	]

coin_1_amount = 0
coin_5_amount = 0
coin_2_amount = 0
coin_10_amount = 0
money_total = 0

def main_process(input_img):
	img0 = input_img.copy()

	img = letterbox(img0, new_shape=imgsz)[0]
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)

	img = torch.from_numpy(img).to(device)
	img = img.float()
	img /= 255.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)
	pred = model(img, augment=True)[0]
	pred = non_max_suppression(pred, my_confidence, my_threshold, classes=my_filterclasses, agnostic=None)

	total = 0
	class_count = [0 for _ in range(len(names))]
	for i, det in enumerate(pred):
		gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
		if det is not None and len(det):
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
			for *xyxy, conf, cls in reversed(det):
				class_count[int(cls)] += 1
				xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
				label = '%sbaht (%.1f%%)' % (names[int(cls)], conf*100)
				total += int(names[int(cls)])
				plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
	# cv2.rectangle(img0,(0,10),(250,90),(0,0,0),-1)
	img0 = cv2.putText(img0, "10Baht "+str(class_count[2])+" coin", (10,45+25*1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " 5Baht "+str(class_count[0])+" coin", (10,45+25*2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " 2Baht "+str(class_count[3])+" coin", (10,45+25*3), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " 1Baht "+str(class_count[1])+" coin", (10,45+25*4), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " Total "+str(total)+" Baht", 					(10,45+25*5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,0), 2)
	
	global coin_1_amount
	global coin_5_amount
	global coin_2_amount
	global coin_10_amount
	global money_total

	coin_10_amount += class_count[2]
	coin_5_amount += class_count[0]
	coin_2_amount += class_count[3]
	coin_1_amount += class_count[1]
	money_total += total

	return img0

if __name__ == '__main__':
	path, dirs, files = next(os.walk("./Upload_Coin"))
	file_count = len(files)

	try:
		os.mkdir("Results")
	except:
		shutil.rmtree("Results")
		os.mkdir("Results")
	print("Coin Detecting")
	for i in range(file_count):
		img = cv2.imread('./Upload_Coin/'+str(i)+'.jpg')
		img = main_process(img).copy()
		cv2.imwrite('./Results/Resoot'+str(i)+'.jpg',img)
	print("Coin Detection is Done")

	fig, ax = plt.subplots(2, 1, figsize=(10,7))
	fig.tight_layout()

	#create subplots
	#plot1
	x = np.array(["1 Baht", "2 Baht", "5 Baht", "10 Baht"])
	#เลขสมมุติ
	y = np.array([coin_1_amount, coin_2_amount, coin_5_amount, coin_10_amount])
	#plt.bar(x, y)
	ax[0].bar(x, y, color='red')
	#plot2
	#x = np.array(["1 Baht", "2 Baht", "5 Baht", "10 Baht"])
	#เลขสมมุติ
	y = np.array([coin_1_amount, coin_2_amount, coin_5_amount, coin_10_amount])
	mylabels = ["1 Baht", "2 Baht", "5 Baht", "10 Baht"]
	mycolors = ["black", "pink", "yellow", "#4CAF50"]

	#plt.bar(x, y)
	ax[1].pie(y, labels = mylabels, colors = mycolors)
	text = fig.text(0.5, 0.02, 'Total: '+str(money_total)+' Baht', ha='center', va='center', size=16)
	text.set_path_effects([path_effects.Normal()])
	plt.savefig("report.jpg")

	#OPEN Report
	file = open("report.jpg", "rb")
	image = file.read()
	widgets.Image(
		value=image,
		format='jpg',
		width=800,
		height=560,
	)