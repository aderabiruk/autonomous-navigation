#!/usr/bin/env python
import os
import cv2
import rospy
import numpy as np
import tensorflow as tf
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from keras.models import load_model
from cv_bridge import CvBridge, CvBridgeError

SPEED = 2.5

DISTANCE = 2
LEFT_ANGLE = 30
RIGHT_ANGLE = 30

TURNING_SPEED = 15
PI = 3.1415926535897
classes = ['Box', 'Space', 'Sphere']
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 200

bridge = CvBridge()
model = load_model(os.path.join(os.path.dirname(__file__), "shape_classifier_le_net_5.h5"))
graph = tf.get_default_graph()

def image_cb(data):	
	cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

	image = preprocess_image(cv_image)

	global graph
	with graph.as_default():
		prediction = classes[np.squeeze(np.argmax(model.predict(image), axis=1))]
		move(prediction)
		
def preprocess_image(image):
	resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
	return np.resize(resized_image, (1, IMAGE_WIDTH, IMAGE_WIDTH, 3))

def move(prediction):
	print("[*] " + str(prediction) + " Detected.")
	velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

	if prediction == classes[1]:
		print("[*] Moving Straight...")
		move_straight(velocity_publisher)
		
	elif prediction == classes[0]:
		print("[*] Stop...")
		stop(velocity_publisher)

	elif prediction == classes[2]:
		print("[*] Turning Right...")
		turn_right(velocity_publisher)

def move_straight(velocity_publisher):
	vel_msg = Twist()
	vel_msg.linear.x = abs(0.5)
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0
	vel_msg.angular.z = 0

	current_distance = 0
	t0 = rospy.Time.now().to_sec()

	while (current_distance < DISTANCE):
		velocity_publisher.publish(vel_msg)

		t1 = rospy.Time.now().to_sec()
		current_distance = SPEED * (t1 - t0)
	vel_msg.linear.x = 0
	velocity_publisher.publish(vel_msg)

def turn_right(velocity_publisher):
	angular_speed = TURNING_SPEED * 2 * PI / 360
	relative_angle = RIGHT_ANGLE* 2 * PI / 360

	vel_msg = Twist()
	vel_msg.linear.x = 0
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0

	vel_msg.angular.z = -abs(angular_speed)

	current_angle = 0
	t0 = rospy.Time.now().to_sec()

	while (current_angle < relative_angle):
		velocity_publisher.publish(vel_msg)

		t1 = rospy.Time.now().to_sec()
		current_angle = angular_speed * (t1 - t0)
	
	vel_msg.linear.z = 0
	velocity_publisher.publish(vel_msg)

def turn_left(velocity_publisher):
	angular_speed = TURNING_SPEED * 2 * PI / 360
	relative_angle = RIGHT_ANGLE* 2 * PI / 360

	vel_msg = Twist()
	vel_msg.linear.x = 0
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0

	vel_msg.angular.z = abs(angular_speed)

	current_angle = 0
	t0 = rospy.Time.now().to_sec()

	while (current_angle < relative_angle):
		velocity_publisher.publish(vel_msg)

		t1 = rospy.Time.now().to_sec()
		current_angle = angular_speed * (t1 - t0)
	
	vel_msg.linear.z = 0
	velocity_publisher.publish(vel_msg)

def stop(velocity_publisher):
	vel_msg = Twist()
	vel_msg.linear.x = 0
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0
	vel_msg.angular.z = 0

	velocity_publisher.publish(vel_msg)

def main():
    rospy.init_node('navigator', anonymous=True)
    image_subscriber = rospy.Subscriber("/mybot/camera/image_raw", Image, image_cb, queue_size=1, buff_size=2**24)
    try:
    	rospy.spin()
    except KeyboardInterrupt as e:
    	print "Shutting Down"
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
