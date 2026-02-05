import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = [
            'waist', 
            'shoulder', 
            'elbow', 
            'forearm_roll', 
            'wrist_angle', 
            'wrist_rotate'
        ]
        self.init_viewer()
        self.reset(seed)

        self.target_yaw = None
        self.target_pitch = None
        self.target_roll = None
        self.target_px = None
        self.target_py = None
        self.target_pz = None

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed != None: np.random.seed(seed=0) 
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'gripper_link',
            q_init       = q_init, # ik from zero pose
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # Set object positions
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range   = [+0.24,+0.4],
            y_range   = [-0.2,+0.2],
            z_range   = [0.82,0.82],
            min_dist  = 0.2,
            xy_margin = 0.0
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        # robotun 6 kol eklemi + 2 parmak eklemi = 8 kontrol girişi
        self.q = np.concatenate([q_zero, np.array([0.024, -0.024])])
        self.p0, self.R0 = self.env.get_pR_body(body_name='gripper_link')
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []
        # reset metodunun sonuna ekleyebilirsin
        self.home_p = np.array([0.3, 0.0, 1.0]) # Başlangıç XYZ (Metre)
        p, R = self.env.get_pR_body('gripper_link')
        rpy = r2rpy(R)

        self.home_p = p.copy()
        self.home_roll  = rpy[0]
        self.home_pitch = rpy[1]
        self.home_yaw   = rpy[2]

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'gripper_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        gripper_cmd = np.array([action[-1]]*4)
        gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='gripper_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('left_finger')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)

    def test_teleop_robot(self, w_yaw, w_x, w_y, w_z):
        dpos = np.zeros(3)
        drot_mat = np.eye(3)
        
        # 1. Mevcut pozu al (Odometri verisi)
        # ee_pose: [px, py, pz, roll, pitch, yaw]
        curr_p, curr_R = self.env.get_pR_body(body_name='gripper_link')
        curr_rpy = r2rpy(curr_R)
        
        px, py, pz = curr_p
        roll, pitch, yaw = curr_rpy

        # --- A. TETİKLEME MEKANİZMASI (Input Check) ---
        # Rotasyon (Yaw)
        if self.env.is_key_pressed_once(key=glfw.KEY_LEFT):
            self.target_yaw = yaw + np.deg2rad(w_yaw)
        elif self.env.is_key_pressed_once(key=glfw.KEY_RIGHT):
            self.target_yaw = yaw - np.deg2rad(w_yaw)

        # İleri-Geri (X)
        if self.env.is_key_pressed_once(key=glfw.KEY_W):
            self.target_px = px + w_x
        elif self.env.is_key_pressed_once(key=glfw.KEY_S):
            self.target_px = px - w_x

        # Sağ-Sol (Y)
        if self.env.is_key_pressed_once(key=glfw.KEY_D):
            self.target_py = py + w_y
        elif self.env.is_key_pressed_once(key=glfw.KEY_A):
            self.target_py = py - w_y

        # Yukarı-Aşağı (Z)
        if self.env.is_key_pressed_once(key=glfw.KEY_R):
            self.target_pz = pz + w_z
        elif self.env.is_key_pressed_once(key=glfw.KEY_F):
            self.target_pz = pz - w_z

        # --- B. TAKİP VE KONTROL MEKANİZMALARI ---

        # 1. Rotasyon Takibi (Yaw)
        if self.target_yaw is not None:
            error_yaw = self.target_yaw - yaw
            if abs(error_yaw) > np.deg2rad(2.0): # 2 derece tolerans
                step_yaw = np.clip(error_yaw * 0.15, -0.05, 0.05)
                drot_mat = rotation_matrix(angle=step_yaw, direction=[0.0, 1.0, 0.0])[:3, :3]
            else:
                self.target_yaw = None

        # 2. Pozisyon X Takibi
        if self.target_px is not None:
            error_x = self.target_px - px
            if abs(error_x) > 0.005: # 5mm tolerans
                dpos[0] = np.clip(error_x * 0.12, -0.007, 0.007)
            else:
                self.target_px = None

        # 3. Pozisyon Y Takibi
        if self.target_py is not None:
            error_y = self.target_py - py
            if abs(error_y) > 0.005:
                dpos[1] = np.clip(error_y * 0.12, -0.007, 0.007)
            else:
                self.target_py = None

        # 4. Pozisyon Z Takibi
        if self.target_pz is not None:
            error_z = self.target_pz - pz
            if abs(error_z) > 0.005:
                dpos[2] = np.clip(error_z * 0.12, -0.007, 0.007)
            else:
                self.target_pz = None

        # --- C. SİSTEM KONTROLLERİ ---
        # Acil Durdurma / Reset
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            self.target_yaw = self.target_px = self.target_py = self.target_pz = None
            return np.zeros(7, dtype=np.float32), True

        # Gripper
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state

        drot_rpy = r2rpy(drot_mat)
        action = np.concatenate([dpos, drot_rpy, np.array([self.gripper_state], dtype=np.float32)], dtype=np.float32)
        return action, False


    def test_teleop_robot1(self, gripper_state, w_roll, w_pitch, w_yaw, w_x, w_y, w_z):
        dpos = np.zeros(3)
        drot_mat = np.eye(3)
        
        curr_p, curr_R = self.env.get_pR_body(body_name='gripper_link')
        px, py, pz = curr_p
        roll, pitch, yaw = r2rpy(curr_R)

         
        self.target_px = self.home_p[0] + (w_x / 100.0) 
        self.target_py = self.home_p[1] + (w_y / 100.0)
        self.target_pz = self.home_p[2] + (w_z / 100.0)
        self.target_yaw = self.home_yaw + np.deg2rad(w_yaw)
        self.target_roll = self.home_roll + np.deg2rad(w_roll)
        self.target_pitch = self.home_pitch + np.deg2rad(w_pitch)

        # print("Target Roll: ", self.target_roll)
        # print("Target Pitch: ", self.target_pitch)
        # print("Target Yaw: ", self.target_yaw)


        # --- KONUM TAKİBİ ---
        if self.target_px is not None: dpos[0] = np.clip((self.target_px - px) * 0.15, -0.01, 0.01)
        if self.target_py is not None: dpos[1] = np.clip((self.target_py - py) * 0.15, -0.01, 0.01)
        if self.target_pz is not None: dpos[2] = np.clip((self.target_pz - pz) * 0.15, -0.01, 0.01)

        # --- ROTASYON TAKİBİ (Matris Çarpımı ile Birleştiriyoruz) ---

        # # # wrist-x
        if self.target_roll is not None:
            error_roll = self.target_roll - roll
            if abs(error_roll) > np.deg2rad(1.5):
                step_roll = np.clip(error_roll * 0.15, -0.05, 0.05)
                drot_mat = rotation_matrix(angle=step_roll, direction=[1.0, 0.0, 0.0])[:3, :3]


        # wrist-z
        if self.target_yaw is not None:
            error_yaw = self.target_yaw - yaw
            if abs(error_yaw) > np.deg2rad(1.5):
                step_yaw = np.clip(error_yaw * 0.15, -0.05, 0.05)
                drot_mat = rotation_matrix(angle=step_yaw, direction=[0.0, 1.0, 0.0])[:3, :3]


        # Pitch Takibi
        if self.target_pitch is not None:
            error_pitch = self.target_pitch + pitch
            if abs(error_pitch) > np.deg2rad(5.0):
                step_pitch = np.clip(error_pitch * 0.15, -0.05, 0.05)
                drot_mat = rotation_matrix(angle=step_pitch, direction=[0.0, 0.0, 1.0])[:3, :3]

        # # Pitch Takibi (X Ekseni)
        # if self.target_pitch is not None:
        #     error_pitch = self.target_pitch - pitch
        #     if abs(error_pitch) > np.deg2rad(1.5):
        #         step_pitch = np.clip(error_pitch * 0.15, -0.05, 0.05)
        #         combined_drot = combined_drot @ rotation_matrix(angle=step_pitch, direction=[1.0, 0.0, 0.0])[:3, :3]

        # Sonuç matrisini ata
        # drot_mat = combined_drot

        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True

        self.gripper_state = gripper_state
        drot_rpy = r2rpy(drot_mat)
        action = np.concatenate([dpos, drot_rpy, np.array([self.gripper_state], dtype=np.float32)], dtype=np.float32)
        return action, False




    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False


    def vision_teleop(self, gripper_state=None, palm_delta=None, distance_delta = None, angle_delta=None):
        
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)



        ## --------------------- Derinlik X ------------------ ##
        if distance_delta is not None:
            if distance_delta > 600 and distance_delta > 900:
                dpos += np.array([-0.007,0.0,0.0])
            elif distance_delta > 300 and distance_delta < 500:
                dpos += np.array([0.007,0.0,0.0])

            else:
                dpos += np.array([0.0,0.0,0.0])
                

        
        else:
            if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
                dpos += np.array([0.007,0.0,0.0])
            if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
                dpos += np.array([-0.007,0.0,0.0])
        ## --------------------- Derinlik X ------------------ ##




        ## --------------------- Y ------------------ ##
        if palm_delta is not None:
            if palm_delta[0] > 0:
                dpos += np.array([0.0,0.007,0.0])
            elif palm_delta[0] < 0:
                dpos += np.array([0.0,-0.007,0.0])
        else:
            if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
                dpos += np.array([0.0,-0.007,0.0])
            if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
                dpos += np.array([0.0,0.007,0.0])
        ## --------------------- Y ------------------ ##



        ## --------------------- Z ------------------ ##
        if palm_delta is not None:
            if palm_delta[1] > 0:
                dpos += np.array([0.0,0.0,-0.007])
            elif palm_delta[1] < 0:
                dpos += np.array([0.0,0.0,0.007])
        else:
            if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
                dpos += np.array([0.0,0.0,0.007])
            if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
                dpos += np.array([0.0,0.0,-0.007])
        ## --------------------- Z ------------------ ##







        ## --------------------- Gripper Z ------------------ ##
        if angle_delta is not None:
            # 1. Mevcut robotun Yaw açısını al (get_ee_pose: [x, y, z, r, p, y])
            current_ee_pose = self.get_ee_pose()
            current_robot_yaw = current_ee_pose[5] # Radyan cinsinden

            # 2. Elin açısını radyana çevir (angle_delta derece cinsinden geliyor)
            # Not: Eğer robotun başlangıç duruşu 0 değilse buraya offset gerekebilir.
            target_yaw_rad = np.radians(angle_delta)

            # 3. Tolerans Kontrolü (5 Derece = 0.087 radyan)
            tolerance_rad = np.radians(5.0)
            error_rad = target_yaw_rad - current_robot_yaw

            if abs(error_rad) > tolerance_rad:
                # 4. Yön Belirleme ve Adım Atma
                # Bir kerede çok sert dönmesin diye error'un %10'u kadar yaklaş (P-Control)
                # Veya sabit hız istiyorsan: step = np.sign(error_rad) * 0.03
                step_size = error_rad * 0.15 
                
                # Robotun aşırı hızlı dönmesini engellemek için clamp (max 0.05 radyan)
                step_size = np.clip(step_size, -0.05, 0.05)
                
                drot = rotation_matrix(angle=step_size, direction=[0.0, 1.0, 0.0])[:3, :3]
            else:
                # Tolerans içindeyse dönmeyi durdur
                drot = np.eye(3)

        else:
            if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
                drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
            if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
                drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        ## --------------------- Gripper Z ------------------ ##

        







        
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]

        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
            
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        

        ## --------------------- GRIPPER ------------------ ##
        if gripper_state is not None:
            self.gripper_state = gripper_state
        else:
            if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
                self.gripper_state = not self.gripper_state
        ## --------------------- GRIPPER ------------------ ##



        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state], dtype=np.float32)], dtype=np.float32)
        return action, False

    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('left_finger')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('left_finger') < 0.1:
            p = self.env.get_p_body('gripper_link')[2]
            if p > 0.9:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug, p_plate
    
    def set_obj_pose(self, p_mug, p_plate):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='gripper_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)