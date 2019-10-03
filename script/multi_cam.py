#!/usr/bin/env python
import cv2
import time
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CAM:
	def g_pipeline_0(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
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

	def g_pipeline_1(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
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

	def g_pipeline_2(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
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

	def g_pipeline_3(self,capture_width=1920,capture_height=1080,display_width=1920,display_height=1080,framerate=120,flip_method=6,):
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
		rospy.init_node('multi_cam', anonymous=True)

		#print(self.g_pipeline_0(flip_method=2))
		cap_0 = cv2.VideoCapture(self.g_pipeline_0(flip_method=6), cv2.CAP_GSTREAMER)
		#print(self.g_pipeline_1(flip_method=2))
		cap_1 = cv2.VideoCapture(self.g_pipeline_1(flip_method=6), cv2.CAP_GSTREAMER)
		#print(self.g_pipeline_2(flip_method=2))
		cap_2 = cv2.VideoCapture(self.g_pipeline_2(flip_method=6), cv2.CAP_GSTREAMER)
		#print(self.g_pipeline_3(flip_method=2))
		cap_3 = cv2.VideoCapture(self.g_pipeline_3(flip_method=6), cv2.CAP_GSTREAMER)

		if (cap_0.isOpened() and cap_1.isOpened() and cap_2.isOpened() and cap_3.isOpened()):
			#print("1")
			image_pub0 = rospy.Publisher("multi_cam_0",Image, queue_size=1)
			image_pub1 = rospy.Publisher("multi_cam_1",Image, queue_size=1)
			image_pub2 = rospy.Publisher("multi_cam_2",Image, queue_size=1)
			image_pub3 = rospy.Publisher("multi_cam_3",Image, queue_size=1)
			bridge = CvBridge()
			self.prev_time=0

			while not rospy.is_shutdown():
				ret_val_0, img_0 = cap_0.read()
				ret_val_1, img_1 = cap_1.read()
				ret_val_2, img_2 = cap_2.read()
				ret_val_3, img_3 = cap_3.read()
				now = time.time()
				img_0 = cv2.flip(img_0, 0)
				img_1 = cv2.flip(img_1, 0)
				img_2 = cv2.flip(img_2, 0)
				img_3 = cv2.flip(img_3, 0)
				image_pub0.publish(bridge.cv2_to_imgmsg(img_0, "bgr8"))
				image_pub1.publish(bridge.cv2_to_imgmsg(img_1, "bgr8"))
				image_pub2.publish(bridge.cv2_to_imgmsg(img_2, "bgr8"))
				image_pub3.publish(bridge.cv2_to_imgmsg(img_3, "bgr8"))
				#print(str(1.0/(now-self.prev_time)) + "fps")
				self.prev_time = now
			cap_0.release()
			cap_1.release()
			cap_2.release()
			cap_3.release()
			cv2.destroyAllWindows()
		else:
			print("Unable to open camera")



        

################ MAIN ###################

cam = CAM()
cam.run()





