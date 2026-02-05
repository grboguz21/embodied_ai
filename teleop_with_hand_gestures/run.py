import numpy as np
from PIL import Image
from mujoco_env.y_env import SimpleEnv

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import threading
import time



class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None
    def apply(self, value):
        if self.state is None: self.state = value
        else: self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state

filter_x = EMAFilter(alpha=0.2)
filter_y = EMAFilter(alpha=0.2)
filter_z = EMAFilter(alpha=0.2)
filter_norm = [EMAFilter(alpha=0.1) for _ in range(3)]
filter_pos_x = EMAFilter(alpha=0.15)
filter_pos_y = EMAFilter(alpha=0.15)
filter_pos_z = EMAFilter(alpha=0.1)

ref_pos = None   # [x, y, z] cm
ref_angles = None # [roll, pitch, yaw] deg
last_stable_pos = [0.0, 0.0, 0.0]
prev_palm_center = None
alpha_p = 0.3  
threshold_p = 10

reference_pos = None

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1  
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

gripper_state_ = False
final_rel_angles = [0.0, 0.0, 0.0]
action = np.zeros(7)
delta_ = [0, 0]
camera_distance_m_ = 0.0
coords_ = (0.0, 0.0, 0.0)


# GLOBALS
last_stable_pos = [0.0, 0.0, 0.0]
ref_pos = None
ref_angles = None

def draw_landmarks_on_image(rgb_image, detection_result):
    global last_stable_pos, ref_pos, ref_angles

    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    gripper_state = False
    palm_normal = np.array([0.0, 0.0, -1.0])
    current_angles = [0.0, 0.0, 0.0]

    final_x, final_y, final_z = last_stable_pos
    final_rel_pos = (0.0, 0.0, 0.0)
    final_rel_angles = [0.0, 0.0, 0.0]

    KNOWN_WIDTH_CM = 2.0
    FOCAL_LENGTH = 400.0

    if not detection_result.hand_landmarks:
        ref_pos = None
        ref_angles = None
        return annotated_image, gripper_state, final_rel_angles, final_rel_pos, "None"

    for hand_landmarks, world_landmarks in zip(
        detection_result.hand_landmarks,
        detection_result.hand_world_landmarks
    ):

        p19_px = np.array([hand_landmarks[19].x * width, hand_landmarks[19].y * height])
        p20_px = np.array([hand_landmarks[20].x * width, hand_landmarks[20].y * height])
        pixel_dist = np.linalg.norm(p19_px - p20_px)

        camera_distance_m = (KNOWN_WIDTH_CM * FOCAL_LENGTH) / pixel_dist / 100.0 if pixel_dist > 0 else 0.0
        raw_dist_cm = camera_distance_m * 100.0

        p0_px = (int(hand_landmarks[0].x * width), int(hand_landmarks[0].y * height))
        p9_px = (int(hand_landmarks[9].x * width), int(hand_landmarks[9].y * height))
        palm_center_px = ((p0_px[0] + p9_px[0]) // 2, (p0_px[1] + p9_px[1]) // 2)

        raw_x_cm = ((palm_center_px[0] - width / 2) * raw_dist_cm) / FOCAL_LENGTH
        raw_y_cm = ((palm_center_px[1] - height / 2) * raw_dist_cm) / FOCAL_LENGTH

        # 
        smooth_x = filter_pos_x.apply(raw_x_cm)
        smooth_y = filter_pos_y.apply(raw_y_cm)
        smooth_z = filter_pos_z.apply(raw_dist_cm)

        THR = 0.4
        final_x = smooth_x if abs(smooth_x - last_stable_pos[0]) > THR else last_stable_pos[0]
        final_y = smooth_y if abs(smooth_y - last_stable_pos[1]) > THR else last_stable_pos[1]
        final_z = smooth_z if abs(smooth_z - last_stable_pos[2]) > THR else last_stable_pos[2]

        last_stable_pos = [final_x, final_y, final_z]

        p0_w = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
        p5_w = np.array([world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z])
        p17_w = np.array([world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

        raw_normal = np.cross(p17_w - p0_w, p5_w - p0_w)
        if np.linalg.norm(raw_normal) > 1e-6:
            raw_normal /= np.linalg.norm(raw_normal)

        palm_normal = np.array([filter_norm[i].apply(raw_normal[i]) for i in range(3)])
        palm_normal /= np.linalg.norm(palm_normal)

        raw_angles = [
            np.degrees(np.arccos(np.clip(palm_normal[i], -1.0, 1.0)))
            for i in range(3)
        ]

        current_angles = [
            filter_x.apply(raw_angles[0]),
            filter_y.apply(raw_angles[1]),
            filter_z.apply(raw_angles[2])
        ]

        if ref_angles is None:
            ref_angles = current_angles.copy()

        final_rel_angles = [
            current_angles[i] - ref_angles[i] for i in range(3)
        ]

        p4_w = np.array([world_landmarks[4].x, world_landmarks[4].y, world_landmarks[4].z])
        p8_w = np.array([world_landmarks[8].x, world_landmarks[8].y, world_landmarks[8].z])
        dist_gripper = np.linalg.norm(p4_w - p8_w) * 100

        line_color = (0, 255, 0)
        if dist_gripper < 6.0:
            gripper_state = True
            line_color = (0, 0, 255)

        p4_px = (int(hand_landmarks[4].x * width), int(hand_landmarks[4].y * height))
        p8_px = (int(hand_landmarks[8].x * width), int(hand_landmarks[8].y * height))
        cv2.line(annotated_image, p4_px, p8_px, line_color, 2)

        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (10, 10), (260, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

        c_white = (255, 255, 255)
        c_yellow = (0, 255, 255)
        c_magenta = (255, 0, 255)

        abs_vals = [abs(x) for x in palm_normal]
        max_idx = abs_vals.index(max(abs_vals))
        directions = [("left" if palm_normal[0]>0 else "right"), 
                      ("down" if palm_normal[1]>0 else "up"), 
                      ("in" if palm_normal[2]>0 else "out")]
        dir_text = directions[max_idx]

        cv2.putText(
            annotated_image,
            f"Orientation (degree) - {dir_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )

        for i, ax in enumerate(["X", "Y", "Z"]):
            cv2.putText(
                annotated_image,
                f"{ax} Degree: {int(final_rel_angles[i])} deg",
                (20, 80 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c_yellow,
                1
            )

        cv2.putText(annotated_image, "Translation (cm)", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(annotated_image, f"X: {last_stable_pos[0]:5.1f}", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_yellow, 1)
        cv2.putText(annotated_image, f"Y: {last_stable_pos[1]:5.1f}", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_yellow, 1)
        cv2.putText(annotated_image, f"Z: {last_stable_pos[2]:5.1f}", (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_yellow, 1)

        cv2.putText(
            annotated_image,
            "GRIPPER: CLOSE" if gripper_state else "GRIPPER: OPEN",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            line_color,
            2
        )

        cv2.circle(annotated_image, palm_center_px, 7, c_magenta, -1)
        cv2.arrowedLine(
            annotated_image,
            palm_center_px,
            (int(palm_center_px[0] + palm_normal[0] * 90),
             int(palm_center_px[1] + palm_normal[1] * 90)),
            c_white,
            3
        )

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    return annotated_image, gripper_state, final_rel_angles, last_stable_pos, dir_text



# using IP Camera

def camera_thread():
    global action, gripper_state_, final_rel_angles, coords_
    cap = cv2.VideoCapture("http://192.168.1.6:8080/video")
    address = "https://192.168.1.6:8080/video"
    cap.open(address)
    # cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        res = draw_landmarks_on_image(rgb_frame, detection_result)
        annotated_frame, state_, final_rel_angles, coords_, _ = res
        gripper_state_ = bool(state_)

        cv2.imshow('Hand Orientation', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def sim_thread():
    global action, coords_, final_rel_angles
    hand_z_cm_= 0
    

    xml_path = './asset/trossen_vx300s/scene.xml'
    PnPEnv = SimpleEnv(xml_path, seed = 0, state_type = 'joint_angle')
    new_plate_pos = np.array([10.5, 0.2, 0.0])
    new_mug_pos = np.array([0.35, -0.0, 0.05]) # Kupa da masada kalsın
    PnPEnv.set_obj_pose(p_mug=new_mug_pos, p_plate=new_plate_pos)

    action = np.zeros(7)    
    reset = False
        
    while PnPEnv.env.is_viewer_alive():
        PnPEnv.step_env()
        if PnPEnv.env.loop_every(HZ=20):
            curr_ee_pose = PnPEnv.get_ee_pose()

            current_gripper = gripper_state_
            hand_x_cm, hand_y_cm, hand_z_cm = coords_
            hand_y_cm = -hand_y_cm
            
            print("[INFO].. ",hand_x_cm, hand_y_cm, hand_z_cm)

            if abs(hand_z_cm) > 70 or hand_z_cm == 0:
                hand_z_cm_ = 0

            if 10 < abs(hand_z_cm) < 60:
                hand_z_cm_ = (60 - hand_z_cm) * 0.8

            # print("[INFO].. ",hand_x_cm, hand_y_cm, hand_z_cm)
            
            wrist_x = -final_rel_angles[0]
            wrist_y = -final_rel_angles[1]
            wrist_z = final_rel_angles[2]

            # print("wrist x: ", wrist_x)
            # print("wrist z: ", wrist_z)
            # action, reset, yaw, error_yaw  = PnPEnv.test_teleop_robot1(gripper_state=current_gripper, 
            #                                            w_pitch = 0.0, w_roll = 0.0 , w_yaw= 0.0, 
            #                                            w_x = hand_z_cm_, w_y = hand_x_cm, w_z = hand_y_cm*1.3)
            action, reset  = PnPEnv.test_teleop_robot1(gripper_state=current_gripper, 
                                                                        w_pitch = 0.0, w_roll = 0.0 , w_yaw= 0.0, 
                                                                        w_x = hand_z_cm_, w_y = 0.0, w_z = 0.0*1.3)

            # print("[INFO]..Target ",yaw, "-", error_yaw)
            # action, reset  = PnPEnv.test_teleop_robot1(gripper_state=current_gripper, w_pitch = 0.0, w_roll = 0.0 , w_yaw= 0.0, w_x=0.0, w_y=0.0, w_z = hand_y_cm)
            # action, reset  = PnPEnv.test_teleop_robot1(gripper_state=current_gripper, w_pitch = 0.0, w_roll = 0.0 , w_yaw= 0.0, w_x=hand_z_cm, w_y=0.0, w_z = 0.0)
            # action, reset  = PnPEnv.test_teleop_robot1(gripper_state=current_gripper, w_pitch = 0.0, w_roll = 0.0 , w_yaw= 0.0, w_x=0.0, w_y=hand_x_cm, w_z = 0.0)


            if reset:
                PnPEnv.reset(seed=0)

            # Get the end-effector pose and images
            ee_pose = PnPEnv.get_ee_pose()

            # İlk 3 eleman pozisyon (x, y, z), son 3 eleman rotasyon (roll, pitch, yaw)
            # px, py, pz, roll, pitch, yaw = ee_pose
            # print(f"Pozisyon: X={px:.2f}, Y={py:.2f}, Z={pz:.2f}")
            # print(f"Rotasyon (Radyan): R={roll:.2f}, P={pitch:.2f}, Y={yaw:.2f}")
                        

            agent_image,wrist_image = PnPEnv.grab_image()

            # resize to 256x256
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)

            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))

            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)

            joint_q = PnPEnv.step(action)

            PnPEnv.render(teleop=True)

    PnPEnv.env.close_viewer()

# Start threads
t1 = threading.Thread(target=camera_thread)
t2 = threading.Thread(target=sim_thread)
t1.start()
t2.start()
t1.join()
t2.join()
