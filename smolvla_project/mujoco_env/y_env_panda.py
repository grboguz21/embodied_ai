import numpy as np
import copy
import glfw
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy

class SimpleEnv:
    def __init__(self, xml_path, action_type='eef_pose', state_type='joint_angle', seed=None):
        # 1. Ortamı Yükle
        self.env = MuJoCoParserClass(name='PandaPnP', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        
        # 2. Panda 7 Eklemlidir
        self.joint_names = [f'joint{i}' for i in range(1, 8)]
        self.gripper_state = False # False: Açık, True: Kapalı
        
        self.env.init_viewer(distance=2.5, elevation=-30)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        
        # Başlangıç pozu
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        # IK ile başlangıç noktasına git
        self.q_init, _, _ = solve_ik(
            env=self.env, joint_names_for_ik=self.joint_names,
            body_name_trgt='tcp_link', q_init=q_home,
            p_trgt=np.array([0.4, 0.0, 1.0]), R_trgt=rpy2r(np.deg2rad([180, 0, 90]))
        )
        
        # HATA ÇÖZÜMÜ BURADA: Tüm qpos yerine sadece joint_names ve parmakları set ediyoruz
        # Robotun 7 eklemi
        self.env.forward(q=self.q_init, joint_names=self.joint_names, increase_tick=False)
        # Parmakların başlangıçta açık olması (0.04)
        self.env.forward(q=[0.04, 0.04], joint_names=['finger_joint1', 'finger_joint2'], increase_tick=False)
        
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        print("PANDA HAZIR.")

    def step(self, action):
        if self.action_type == 'eef_pose':
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q_target, _, _ = solve_ik(
                env=self.env, joint_names_for_ik=self.joint_names,
                body_name_trgt='tcp_link', q_init=self.env.get_qpos_joints(self.joint_names),
                p_trgt=self.p0, R_trgt=self.R0
            )
        else:
            q_target = action[:7]

        # Gripper: 0 (Açık) - 255 (Kapalı)
        ctrl_val = 255.0 if action[-1] > 0.5 else 0.0
        
        # Adım atarken sadece aktüatörleri hedefliyoruz
        # Panda actuator listesi: 7 kol + 1 gripper (toplam 8 aktüatör)
        ctrl = np.concatenate([q_target, [ctrl_val]])
        self.env.step(ctrl)
        
        return self.get_joint_state()

    def teleop_robot(self):
        dpos = np.zeros(3)
        # Klavye Kontrolleri
        if self.env.is_key_pressed_repeat(glfw.KEY_W): dpos[0] -= 0.005
        if self.env.is_key_pressed_repeat(glfw.KEY_S): dpos[0] += 0.005
        if self.env.is_key_pressed_repeat(glfw.KEY_A): dpos[1] -= 0.005
        if self.env.is_key_pressed_repeat(glfw.KEY_D): dpos[1] += 0.005
        if self.env.is_key_pressed_repeat(glfw.KEY_R): dpos[2] += 0.005
        if self.env.is_key_pressed_repeat(glfw.KEY_F): dpos[2] -= 0.005
        
        if self.env.is_key_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state
            
        action = np.concatenate([dpos, [0,0,0], [float(self.gripper_state)]])
        return action, self.env.is_key_pressed_once(glfw.KEY_Z)

    def grab_image(self):
        img_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
        img_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric')
        return img_agent, img_ego

    def render(self, teleop=True):
        img_agent, img_ego = self.grab_image()
        self.env.viewer_rgb_overlay(add_title_to_img(img_agent, "Agent"), loc='top right')
        self.env.viewer_rgb_overlay(add_title_to_img(img_ego, "Wrist"), loc='bottom right')
        self.env.render()

    def get_joint_state(self):
        q = self.env.get_qpos_joints(self.joint_names)
        return np.concatenate([q, [1.0 if self.gripper_state else 0.0]])