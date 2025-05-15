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

prev_error = 0.0
integral = 0.0

def nothing(x):
    pass

def init_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 119, 255, nothing)
    # cv2.createTrackbar("L - V", "Trackbars", 58, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

def get_trackbar_values():
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])

def process_lane_detection(cv_image):
    try:
        frame = cv2.resize(cv_image, (640, 480))
    except:
        return None, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = get_trackbar_values()
    mask = cv2.inRange(hsv, lower, upper)

    # Lebarkan ROI untuk deteksi lebih baik saat tikungan
    roi = mask[280:480, :]  # area bawah frame lebih luas
    histogram = np.sum(roi, axis=0)

    if np.max(histogram) < 1000:
        return None, mask, frame

    lane_center = np.argmax(histogram)
    frame_center = frame.shape[1] // 2
    deviation = frame_center - lane_center

    debug = frame.copy()
    # Garis hijau: posisi garis putus-putus terdeteksi
    cv2.line(debug, (lane_center, 280), (lane_center, 480), (0, 255, 0), 3)
    # Garis biru: posisi tengah frame
    cv2.line(debug, (frame_center, 280), (frame_center, 480), (255, 0, 0), 3)

    cv2.imshow("Threshold Mask", mask)
    cv2.imshow("Lane Visualization", debug)
    cv2.waitKey(1)

    return deviation, mask, frame

def camera_callback(data):
    global bridge, pub_steer_left, pub_steer_right, pub_wheel1, pub_wheel2
    global prev_error, integral

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    deviation, mask, frame = process_lane_detection(cv_image)
    if deviation is None:
        rospy.logwarn("Line not detected")
        return

    # PID controller parameters
    Kp = 0.0025
    Ki = 0.00001
    Kd = 0.001

    max_speed = 5.0
    threshold_tajam = 100  # Threshold deviasi piksel untuk tikungan tajam

    error = float(deviation)

    # Anti-windup integral
    integral += error
    if integral > 10000:
        integral = 10000
    elif integral < -10000:
        integral = -10000

    derivative = error - prev_error
    prev_error = error

    # Kecepatan adaptif
    if abs(error) > threshold_tajam:
        wheel_speed = max_speed * 0.2  # kurangi kecepatan saat tikungan tajam
    else:
        wheel_speed = max_speed

    steer_angle = Kp * error + Ki * integral + Kd * derivative
    steer_angle = max(min(steer_angle, 0.5), -0.5)  # Clamp steering

    pub_steer_left.publish(steer_angle)
    pub_steer_right.publish(steer_angle)
    pub_wheel1.publish(wheel_speed)
    pub_wheel2.publish(wheel_speed)

def direct_drive():
    global pub_steer_left, pub_steer_right, pub_wheel1, pub_wheel2

    rospy.init_node('lane_center_pid_follower')

    init_trackbars()

    pub_steer_left = rospy.Publisher('/catvehicle/front_left_steering_position_controller/command', Float64, queue_size=10)
    pub_steer_right = rospy.Publisher('/catvehicle/front_right_steering_position_controller/command', Float64, queue_size=10)
    pub_wheel1 = rospy.Publisher('/catvehicle/joint1_velocity_controller/command', Float64, queue_size=10)
    pub_wheel2 = rospy.Publisher('/catvehicle/joint2_velocity_controller/command', Float64, queue_size=10)

    rospy.Subscriber('/catvehicle/camera_front/image_raw_front', Image, camera_callback)

    rospy.loginfo("Lane follower PID node started.")
    rospy.spin()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        direct_drive()
    except rospy.ROSInterruptException:
        pass
