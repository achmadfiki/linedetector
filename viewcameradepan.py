#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

bridge = CvBridge()
pub_steer_left = None
pub_steer_right = None
pub_wheel1 = None
pub_wheel2 = None

current_steer = 0.0
current_speed = 15.0

def process_lane_detection(cv_image):
    try:
        frame = cv2.resize(cv_image, (940, 780))
    except:
        return None, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Nilai threshold HSV tetap (karena trackbar dihapus)
    lower = np.array([0, 0, 119])
    upper = np.array([255, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    roi = mask[280:480, :]
    histogram = np.sum(roi, axis=0)

    if np.max(histogram) < 1000:
        return None, mask, frame

    lane_center = np.argmax(histogram)
    frame_center = frame.shape[1] // 2
    deviation = frame_center - lane_center

    return deviation, mask, frame

def camera_callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    _, _, frame = process_lane_detection(cv_image)
    if frame is None:
        return

    # Tampilkan kamera asli (jika ingin lihat gambar dari kamera)
    cv2.imshow("Camera View", frame)
    key = cv2.waitKey(1) & 0xFF
    manual_control(key)

def manual_control(key):
    global current_steer, current_speed

    if key == ord('w'):
        current_speed = 5.0
    elif key == ord('s'):
        current_speed = -3.0
    elif key == ord('a'):
        current_steer = 0.3
    elif key == ord('d'):
        current_steer = -0.3
    elif key == ord('q'):
        current_speed = 0.0
        current_steer = 0.0

    pub_steer_left.publish(current_steer)
    pub_steer_right.publish(current_steer)
    pub_wheel1.publish(current_speed)
    pub_wheel2.publish(current_speed)

def direct_drive():
    global pub_steer_left, pub_steer_right, pub_wheel1, pub_wheel2

    rospy.init_node('manual_lane_follower')

    pub_steer_left = rospy.Publisher('/catvehicle/front_left_steering_position_controller/command', Float64, queue_size=10)
    pub_steer_right = rospy.Publisher('/catvehicle/front_right_steering_position_controller/command', Float64, queue_size=10)
    pub_wheel1 = rospy.Publisher('/catvehicle/joint1_velocity_controller/command', Float64, queue_size=10)
    pub_wheel2 = rospy.Publisher('/catvehicle/joint2_velocity_controller/command', Float64, queue_size=10)

    rospy.Subscriber('/catvehicle/camera_front/image_raw_front', Image, camera_callback)

    rospy.loginfo("Manual control node started. Use W/A/S/D/Q keys.")
    rospy.spin()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        direct_drive()
    except rospy.ROSInterruptException:
        pass
