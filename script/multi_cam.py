#!/usr/bin/env python
import cv2
import time
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo
from sensor_msgs.srv import SetCameraInfo
from sensor_msgs.srv import SetCameraInfoResponse

import os
import errno
import yaml

default_camera_info_url = "file://${ROS_HOME}/camera_info/${NAME}.yaml";


class CAM:

	def __init__(self):
		rospy.init_node('multi_cam', anonymous=True)
		self.CAM_ID = rospy.get_param('~camera_ID', 3)
		rospy.loginfo(self.CAM_ID)

	def g_pipeline0(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
		return (
			"nvarguscamerasrc sensor-id=0 ! "
			"video/x-raw(memory:NVMM), "
			"width=(int)%d, height=(int)%d, "
			"format=(string)NV12, framerate=(fraction)%d/1 ! "
			"nvvidconv flip-method=%d ! "
			"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
			"videoconvert ! "
			"video/x-raw, format=(string)BGR ! appsink"
			% (capture_width,capture_height,framerate,flip_method,display_width,display_height,))

	def g_pipeline1(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
		return (
			"nvarguscamerasrc sensor-id=1 ! "
			"video/x-raw(memory:NVMM), "
			"width=(int)%d, height=(int)%d, "
			"format=(string)NV12, framerate=(fraction)%d/1 ! "
			"nvvidconv flip-method=%d ! "
			"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
			"videoconvert ! "
			"video/x-raw, format=(string)BGR ! appsink"
			% (capture_width,capture_height,framerate,flip_method,display_width,display_height,))

	def g_pipeline2(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
		return (
			"nvarguscamerasrc sensor-id=2 ! "
			"video/x-raw(memory:NVMM), "
			"width=(int)%d, height=(int)%d, "
			"format=(string)NV12, framerate=(fraction)%d/1 ! "
			"nvvidconv flip-method=%d ! "
			"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
			"videoconvert ! "
			"video/x-raw, format=(string)BGR ! appsink"
			% (capture_width,capture_height,framerate,flip_method,display_width,display_height,))

	def g_pipeline3(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
		return (
			"nvarguscamerasrc sensor-id=3 ! "
			"video/x-raw(memory:NVMM), "
			"width=(int)%d, height=(int)%d, "
			"format=(string)NV12, framerate=(fraction)%d/1 ! "
			"nvvidconv flip-method=%d ! "
			"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
			"videoconvert ! "
			"video/x-raw, format=(string)BGR ! appsink"
			% (capture_width,capture_height,framerate,flip_method,display_width,display_height,))

	def run(self):
		# Ros init

		#self.CAM_ID = rospy.get_param('/ID_', 3)

		if self.CAM_ID == 0 :
			print("0")
			cap = cv2.VideoCapture(self.g_pipeline0(flip_method=6), cv2.CAP_GSTREAMER)
		elif self.CAM_ID == 1 :
			print("1")
			cap = cv2.VideoCapture(self.g_pipeline1(flip_method=6), cv2.CAP_GSTREAMER)
		elif self.CAM_ID == 2 :
			print("2")
			cap = cv2.VideoCapture(self.g_pipeline2(flip_method=6), cv2.CAP_GSTREAMER)
		elif self.CAM_ID == 3 :
			print("3")
			cap = cv2.VideoCapture(self.g_pipeline3(flip_method=6), cv2.CAP_GSTREAMER)
		else :
			print("invalid cam_id")

		if (cap.isOpened()):
			image_pub = rospy.Publisher("multi_cam",Image, queue_size=1)
			bridge = CvBridge()
			self.prev_time = 0

			while not rospy.is_shutdown():
				now = time.time()
				ret_val, img = cap.read()
				img = cv2.flip(img, 0)
				image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
				print(str(1.0/(now-self.prev_time)) + "fps")
				self.prev_time = now
			cap.release()
			cv2.destroyAllWindows()
		else:
			print("Unable to open camera")



################ MAIN ###################

cam = CAM()
cam.run()





