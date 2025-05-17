
# simulator urban dengan mengatur jarak atau mengikutin jarak

#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf
import threading

bridge = CvBridge()
pub_steer_left = None
pub_steer_right = None
pub_wheel1 = None
pub_wheel2 = None

current_steer = 0.0
current_speed = 0.0
vehicle_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
control_mode = "FOLLOW_LINE"
total_distance = 0.0
prev_x = None
prev_y = None

def process_lane_detection(cv_image):
    try:
        frame = cv2.resize(cv_image, (940, 780))
    except:
        return None, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 119])
    upper = np.array([255, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    window_height = 80
    window_width = 80
    num_windows = frame.shape[1] // window_width

    lane_centers = []
    y_start = frame.shape[0] - window_height
    y_end = frame.shape[0]

    for i in range(num_windows):
        start_x = i * window_width
        end_x = start_x + window_width
        window = mask[y_start:y_end, start_x:end_x]

        if np.sum(window) > 0:
            lane_center = np.mean(np.where(window > 0)[1]) + start_x
            lane_centers.append(lane_center)

    if len(lane_centers) == 0:
        return None, mask, frame

    lane_center_avg = int(np.mean(lane_centers))
    frame_center = frame.shape[1] // 2
    deviation = frame_center - lane_center_avg

    return deviation, mask, frame

def pose_callback(msg):
    global vehicle_pose
    try:
        index = msg.name.index('catvehicle')
        pose = msg.pose[index]
        orientation_q = pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation_list)

        vehicle_pose['x'] = pose.position.x
        vehicle_pose['y'] = pose.position.y
        vehicle_pose['yaw'] = np.degrees(yaw)
    except ValueError:
        pass

def camera_callback(data):
    global current_steer, control_mode, total_distance, prev_x, prev_y

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Update jarak tempuh setiap frame
    curr_x = vehicle_pose['x']
    curr_y = vehicle_pose['y']

    if prev_x is not None and prev_y is not None:
        if abs(current_speed) > 0.01:  # hanya jika kendaraan bergerak
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            step_distance = np.sqrt(dx**2 + dy**2)
            total_distance += step_distance

    prev_x = curr_x
    prev_y = curr_y

    deviation, mask, frame = process_lane_detection(cv_image)
    if frame is None:
        return

    status = determine_status()

    if control_mode == "FOLLOW_LINE":
        if deviation is not None:
            correction = -0.002 * deviation
            current_steer = np.clip(correction, -0.3, 0.3)
        else:
            current_steer = 0.0

    if deviation is not None:
        cv2.putText(frame, f"Deviasi: {deviation}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Jarak ditempuh: {total_distance:.2f} m", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    cv2.putText(frame, f"Kecepatan: {current_speed:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Stir: {current_steer:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Posisi: x={vehicle_pose['x']:.2f}, y={vehicle_pose['y']:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)
    cv2.putText(frame, f"Arah (Yaw): {vehicle_pose['yaw']:.2f} deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 200), 2)
    cv2.imshow("Mask View", mask)
    cv2.imshow("Camera View", frame)
    

    key = cv2.waitKey(1) & 0xFF
    manual_control(key)

    publish_commands()

def determine_status():
    if current_speed == 0.0:
        return "BERHENTI"
    elif current_steer > 0.1:
        return "BELOK KIRI"
    elif current_steer < -0.1:
        return "BELOK KANAN"
    else:
        return "MAJU LURUS"

def manual_control(key):
    global current_steer, current_speed, control_mode

    if key == ord('w'):
        current_speed = 5.0
        control_mode = "FOLLOW_LINE"
    elif key == ord('s'):
        current_speed = -3.0
        control_mode = "FOLLOW_LINE"
    elif key == ord('a'):
        current_steer = 0.3
        control_mode = "MANUAL_TURN"
    elif key == ord('d'):
        current_steer = -0.3
        control_mode = "MANUAL_TURN"
    elif key == ord('q'):
        current_speed = 0.0
        current_steer = 0.0
        control_mode = "FOLLOW_LINE"

def publish_commands():
    if pub_steer_left:
        pub_steer_left.publish(current_steer)
    if pub_steer_right:
        pub_steer_right.publish(current_steer)
    if pub_wheel1:
        pub_wheel1.publish(current_speed)
    if pub_wheel2:
        pub_wheel2.publish(current_speed)

def drive_until_distance(target_distance):
    global total_distance, prev_x, prev_y, current_speed

    rate = rospy.Rate(10)

    # Inisialisasi posisi sebelumnya jika belum ada
    if prev_x is None or prev_y is None:
        prev_x = vehicle_pose['x']
        prev_y = vehicle_pose['y']

    start_distance = total_distance  # catat jarak sebelum mulai

    while not rospy.is_shutdown():
        curr_x = vehicle_pose['x']
        curr_y = vehicle_pose['y']

        # Update jarak hanya jika kendaraan bergerak
        if abs(current_speed) > 0.01:  # threshold kecil untuk menghindari noise
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            step_distance = np.sqrt(dx**2 + dy**2)
            total_distance += step_distance

        prev_x = curr_x
        prev_y = curr_y

        if (total_distance - start_distance) >= target_distance:
            break

        rate.sleep()

def auto_drive_sequence():
    global current_speed, current_steer, control_mode
    rospy.sleep(3)

    rospy.loginfo("Auto Drive: Maju")
    current_speed = 5.0
    control_mode = "FOLLOW_LINE"
    current_steer = 0.0
    publish_commands()
    drive_until_distance(25.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(7.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(53.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(7.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(70.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(7.0)
    
    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(130.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(15.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(118.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(20.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(148.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(16.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(126.0)

    rospy.loginfo("Auto Drive: Belok Kiri")
    control_mode = "MANUAL_TURN"
    current_steer = 0.3
    publish_commands()
    drive_until_distance(14.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(67.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(16.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(48.0)

    rospy.loginfo("Auto Drive: Belok Kanan")
    control_mode = "MANUAL_TURN"
    current_steer = -3.0
    publish_commands()
    drive_until_distance(17.0)


    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(152.0)

    rospy.loginfo("Auto Drive: Belok Kiri")
    control_mode = "MANUAL_TURN"
    current_steer = 0.3
    publish_commands()
    drive_until_distance(16.0)

    rospy.loginfo("Auto Drive: Maju")
    control_mode = "FOLLOW_LINE"
    current_speed = 5.0
    current_steer = 0.0
    publish_commands()
    drive_until_distance(26.0)


    rospy.loginfo("Auto Drive: Stop")
    current_speed = 0.0
    current_steer = 0.0
    control_mode = "FOLLOW_LINE"
    publish_commands()

def user_input_thread():
    pass

def direct_drive():
    global pub_steer_left, pub_steer_right, pub_wheel1, pub_wheel2
    global prev_x, prev_y

    rospy.init_node('manual_auto_drive')

    # Inisialisasi posisi awal untuk jarak
    prev_x = vehicle_pose['x']
    prev_y = vehicle_pose['y']

    pub_steer_left = rospy.Publisher('/catvehicle/front_left_steering_position_controller/command', Float64, queue_size=10)
    pub_steer_right = rospy.Publisher('/catvehicle/front_right_steering_position_controller/command', Float64, queue_size=10)
    pub_wheel1 = rospy.Publisher('/catvehicle/joint1_velocity_controller/command', Float64, queue_size=10)
    pub_wheel2 = rospy.Publisher('/catvehicle/joint2_velocity_controller/command', Float64, queue_size=10)

    rospy.Subscriber('/catvehicle/camera_front/image_raw_front', Image, camera_callback)
    rospy.Subscriber('/gazebo/model_states', ModelStates, pose_callback)

    threading.Thread(target=user_input_thread, daemon=True).start()
    threading.Thread(target=auto_drive_sequence, daemon=True).start()

    rospy.loginfo("Manual + Auto Drive Node Started. Use W/A/S/D/Q to control.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        direct_drive()
    except rospy.ROSInterruptException:
        pass
