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
    def __init__(self, xml_path, action_type='eef_pose', state_type='joint_angle', seed=None): 
        # 1. MuJoCo Sahnesini Yükle 
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path) 
        self.action_type = action_type 
        self.state_type = state_type 

        # 2. ViperX 300s Eklem İsimleri 
        self.joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'] 
        
        self.init_viewer() 
        self.reset(seed) 

        # Hedef değişkenleri 
        self.target_yaw = self.target_pitch = self.target_roll = None 
        self.target_px = self.target_py = self.target_pz = None 

    def init_viewer(self): 
        self.env.reset() 
        self.env.init_viewer( 
            distance=2.0, elevation=-30, transparent=False, black_sky=True, 
            use_rgb_overlay=False, loc_rgb_overlay='top right' 
        ) 

    def reset(self, seed=None): 
        ''' 
        Reset the environment 
        VX300s için optimize edilmiş başlangıç pozu 
        ''' 
        if seed is not None: np.random.seed(seed=seed)  
        
        # 1. Başlangıç Tahmini (Seed Pose):  
        # Robotu tamamen dik (0,0,0...) yerine hafif bükük başlatmak IK'nın kilitlenmesini önler. 
        # [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate] 
        q_init = np.deg2rad([0, 0, 0, 0, 0, 0])  

        # 2. Hedef Pozisyon (Target Pose): 
        # Masanın üstünde, robotun rahatça ulaşabileceği bir nokta. 
        # Z=0.45 civarı masaya yakın ama çarpma riskini azaltan bir yüksekliktir. 
        p_trgt = np.array([0.35, 0.0, 0.45])  
        
        # 3. Hedef Yönelim (Target Orientation): 
        # Gripper'ın burnunu tam olarak aşağı (masaya) bakacak şekilde ayarlar. 
        R_trgt = rpy2r(np.deg2rad([0, 0, 0])) # Pitch 90: Bileği aşağı büker 

        # 4. IK Çözümü: 
        # max_ik_tick değerini 100 yaparak ilk reset anında kesin çözüm almasını sağlıyoruz. 
        q_zero, ik_err, ik_info = solve_ik( 
            env                = self.env, 
            joint_names_for_ik = self.joint_names, 
            body_name_trgt     = 'gripper_link', 
            q_init             = q_init, 
            p_trgt             = p_trgt, 
            R_trgt             = R_trgt, 
            max_ik_tick        = 2000, 
            ik_stepsize        = 0.5 
        ) 
        
        # Robotu hesaplanan poza ışınla 
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False) 

        # Dinamik Nesne Bulma ve Yerleştirme 
        all_bodies = self.env.get_body_names() 
        self.mug_name = next((b for b in all_bodies if 'mug' in b.lower()), 'body_obj_mug_5') 
        self.plate_name = next((b for b in all_bodies if 'plate' in b.lower()), 'body_obj_plate_11') 

        obj_names = [self.mug_name, self.plate_name] 
        # Nesneleri robotun önündeki x[0.3, 0.5] y[-0.2, 0.2] alanına dağıt 
        obj_xyzs = sample_xyzs( 
            len(obj_names), 
            x_range   = [0.3, 0.5], 
            y_range   = [-0.2, 0.2], 
            z_range   = [0.82, 0.82], # Masa yüzey yüksekliği 
            min_dist  = 0.15 
        ) 
        for i, name in enumerate(obj_names): 
            try: 
                self.env.set_p_base_body(body_name=name, p=obj_xyzs[i, :]) 
                self.env.set_R_base_body(body_name=name, R=np.eye(3)) 
            except: pass 
            
        self.env.forward(increase_tick=False) 

        # Kontrol Değişkenlerini Güncelle (6 Kol + 1 Gripper) 
        self.last_q = copy.deepcopy(q_zero) 
        # Gripper'ı hafif açık (0.04) başlat 
        self.q = np.concatenate([q_zero, [0.04]])  
        
        # Mevcut EEF pozunu başlangıç referansı olarak kaydet 
        self.p0, self.R0 = self.env.get_pR_body(body_name='gripper_link') 
        self.home_p = self.p0.copy() 
        rpy = r2rpy(self.R0) 
        self.home_roll, self.home_pitch, self.home_yaw = rpy[0], rpy[1], rpy[2] 

        # Simülasyonu stabilize etmek için 50 adım ilerlet 
        for _ in range(50): self.step_env() 
        
        self.gripper_state = False 
        # print(f"INITIALIZATION DONE - IK Error: {float(ik_err):.5f}") 

    def step(self, action): 
        ''' action: [dx, dy, dz, droll, dpitch, dyaw, gripper] (7 eleman) ''' 
        if self.action_type == 'eef_pose': 
            q_curr = self.env.get_qpos_joints(joint_names=self.joint_names) 
            self.p0 += action[:3] 
            self.R0 = self.R0.dot(rpy2r(action[3:6])) 
            q, _, _ = solve_ik( 
                env=self.env, joint_names_for_ik=self.joint_names, body_name_trgt='gripper_link', 
                q_init=q_curr, p_trgt=self.p0, R_trgt=self.R0, max_ik_tick=2000 
            ) 
        elif self.action_type == 'delta_joint_angle': 
            q = action[:-1] + self.last_q 
        elif self.action_type == 'joint_angle': 
            q = action[:-1] 
        else: 
            raise ValueError('action_type not recognized') 

        self.compute_q = q 
        # VX300s gripper için tek bir skalar aktüatör girişi (7. eleman) 
        self.q = np.concatenate([q, [action[-1]]]) 

        if self.state_type == 'joint_angle': return self.get_joint_state() 
        elif self.state_type == 'ee_pose': return self.get_ee_pose() 
        elif 'delta_q' in self.state_type: return self.get_delta_q() 

    def step_env(self): 
        self.env.step(self.q) 

    def grab_image(self): 
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview') 
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric') 
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name='sideview') 
        return self.rgb_agent, self.rgb_ego 

    def render(self, teleop=False): 
        self.env.plot_time() 
        p, R = self.env.get_pR_body(body_name='gripper_link') 
        R_viz = R @ np.array([[1,0,0],[0,0,1],[0,1,0]]) 
        # self.env.plot_sphere(p=p, r=0.02, rgba=[0.9, 0.1, 0.1, 0.5]) 
        # self.env.plot_capsule(p=p, R=R_viz, r=0.01, h=0.2, rgba=[0.1, 0.9, 0.1, 0.5]) 
        
        self.grab_image() 
        # rgb_ego = add_title_to_img(self.rgb_ego, text='Egocentric', shape=(640,480)) 
        # rgb_agent = add_title_to_img(self.rgb_agent, text='Agent View', shape=(640,480)) 
        
        # self.env.viewer_rgb_overlay(rgb_agent, loc='top right') 
        # self.env.viewer_rgb_overlay(rgb_ego, loc='bottom right') 
        
        if teleop: 
            self.env.viewer_text_overlay(text1='Key Pressed', text2='%s'%(self.env.get_key_pressed_list())) 
        self.env.render() 


    def get_joint_state(self): 
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names) 
        gripper_pos = self.env.get_qpos_joint('left_finger')[0] 
        gripper_stat = 1.0 if gripper_pos > 0.04 else 0.0 
        return np.concatenate([qpos, [gripper_stat]], dtype=np.float32) 

    def get_ee_pose(self): 
        p, R = self.env.get_pR_body(body_name='gripper_link') 
        return np.concatenate([p, r2rpy(R)], dtype=np.float32) 

    def get_delta_q(self): 
        delta = self.compute_q - self.last_q 
        self.last_q = copy.deepcopy(self.compute_q) 
        gripper_stat = self.get_joint_state()[-1] 
        return np.concatenate([delta, [gripper_stat]], dtype=np.float32) 

    def check_success(self): 
        p_mug, p_plate = self.get_obj_pose() 
        # Mug tabak üzerinde mi ve gripper açık mı? 
        dist_xy = np.linalg.norm(p_mug[:2] - p_plate[:2]) 
        dist_z = np.abs(p_mug[2] - p_plate[2]) 
        gripper_open = self.env.get_qpos_joint('left_finger')[0] < 0.03 
        if dist_xy < 0.1 and dist_z < 0.05 and gripper_open: 
            if self.env.get_pR_body('gripper_link')[0][2] > 0.9: 
                return True 
        return False 

    def get_obj_pose(self): 
        try: 
            return self.env.get_p_body(self.mug_name), self.env.get_p_body(self.plate_name) 
        except: 
            return np.zeros(3), np.zeros(3) 

    def set_obj_pose(self, p_mug, p_plate): 
        self.env.set_p_base_body(body_name=self.mug_name, p=p_mug) 
        self.env.set_p_base_body(body_name=self.plate_name, p=p_plate) 
        self.env.forward(increase_tick=False) 

    def teleop_robot(self): 
        ''' 
        Klavyeyi kullanarak robotu manuel kontrol etme metodu. 
        Returns: 
            action: np.array [dx, dy, dz, droll, dpitch, dyaw, gripper] (7 eleman) 
            done: bool (Z tuşuna basılırsa True döner) 
        ''' 
        dpos = np.zeros(3) 
        drot = np.eye(3) 
        
        # Hareket Kontrolleri (W, S, A, D, R, F) 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S): dpos += [-0.007, 0, 0] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W): dpos += [0.007, 0, 0] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A): dpos += [0, 0.007, 0] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D): dpos += [0, -0.007, 0] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R): dpos += [0, 0, 0.007] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F): dpos += [0, 0, -0.007] 
        
        # Rotasyon Kontrolleri (Ok Tuşları, Q, E) 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT): 
            drot = rotation_matrix(angle=0.1, direction=[1, 0, 0])[:3, :3] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT): 
            drot = rotation_matrix(angle=-0.1, direction=[1, 0, 0])[:3, :3] 
            
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN): 
            drot = rotation_matrix(angle=0.06, direction=[0, 1, 0])[:3, :3] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP): 
            drot = rotation_matrix(angle=-0.06, direction=[0, 1, 0])[:3, :3] 

        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q): 
            drot = rotation_matrix(angle=0.06, direction=[0, 0, 1])[:3, :3] 
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E): 
            drot = rotation_matrix(angle=-0.06, direction=[0, 0, 1])[:3, :3] 
            
        # Gripper Kontrolü (Space) 
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE): 
            self.gripper_state = not self.gripper_state 
            
        # Aksiyonu oluştur: 6 kol hareketi + 1 gripper değeri 
        # Gripper değerleri: 0.057 (açık), 0.021 (kapalı) - VX300s limitleri 
        gripper_val = 0.021 if self.gripper_state else 0.057 
        action = np.concatenate([dpos, r2rpy(drot), [gripper_val]]) 
        
        return action, self.env.is_key_pressed_once(key=glfw.KEY_Z) 


    def test_teleop_robot(self, w_roll,w_pitch,w_yaw, w_x, w_y, w_z): 
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
            self.target_roll = roll + np.deg2rad(w_roll) 
        elif self.env.is_key_pressed_once(key=glfw.KEY_RIGHT): 
            self.target_roll = roll - np.deg2rad(w_roll) 

        if self.env.is_key_pressed_once(key=glfw.KEY_DOWN): 
            self.target_pitch = pitch + np.deg2rad(w_pitch) 
        elif self.env.is_key_pressed_once(key=glfw.KEY_UP): 
            self.target_pitch = pitch - np.deg2rad(w_pitch) 

        if self.env.is_key_pressed_once(key=glfw.KEY_Q): 
            self.target_yaw = yaw + np.deg2rad(w_yaw) 
        elif self.env.is_key_pressed_once(key=glfw.KEY_E): 
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
        if self.target_roll is not None: 
            error_roll = self.target_roll - roll 
            if abs(error_roll) > np.deg2rad(2.0): # 2 derece tolerans 
                step_roll = np.clip(error_roll * 0.15, -0.05, 0.05) 
                drot_mat = rotation_matrix(angle=step_roll, direction=[1.0, 0.0, 0.0])[:3, :3] 
            else: 
                self.target_roll = None 

        # 1. Rotasyon Takibi (Yaw) 
        if self.target_pitch is not None: 
            error_pitch = self.target_pitch - pitch 
            if abs(error_pitch) > np.deg2rad(2.0): # 2 derece tolerans 
                step_pitch = np.clip(error_pitch * 0.15, -0.05, 0.05) 
                drot_mat = rotation_matrix(angle=step_pitch, direction=[0.0, 1.0, 0.0])[:3, :3] 
            else: 
                self.target_pitch = None 

        # 1. Rotasyon Takibi (Yaw) 
        if self.target_yaw is not None: 
            error_yaw = self.target_yaw - yaw 
            if abs(error_yaw) > np.deg2rad(2.0): # 2 derece tolerans 
                step_yaw = np.clip(error_yaw * 0.15, -0.05, 0.05) 
                drot_mat = rotation_matrix(angle=step_yaw, direction=[0.0, 0.0, 1.0])[:3, :3] 
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
        # Yapıyı bozmadan, rotasyonları birleştirmek için birim matrisle başlıyoruz
        combined_drot = np.eye(3) 
        
        # 1. Mevcut pozu al (Odometri verisi)
        curr_p, curr_R = self.env.get_pR_body(body_name='gripper_link')
        curr_rpy = r2rpy(curr_R)
        
        px, py, pz = curr_p
        roll, pitch, yaw = curr_rpy
 
        # Hedef pozisyonları belirle
        self.target_px = self.home_p[0] + (w_x / 100.0)  
        self.target_py = self.home_p[1] + (w_y / 100.0) 
        self.target_pz = self.home_p[2] + (w_z / 100.0) 

        # Hedef oryantasyonları belirle
        self.target_yaw = self.home_yaw + np.deg2rad(w_yaw) 
        self.target_roll = self.home_roll + np.deg2rad(w_roll) 
        self.target_pitch = self.home_pitch + np.deg2rad(w_pitch) 

        # Açı farkını normalize eden yardımcı fonksiyon (Zıplamaları önler)
        def norm_ang(a): return np.arctan2(np.sin(a), np.cos(a))

        # --- ROTASYON TAKİBİ (Matris Çarpımı ile Birleştiriyoruz) --- 
        # Roll Takibi
        if self.target_roll is not None: 
            error_roll = norm_ang(self.target_roll - roll) 
            if abs(error_roll) > np.deg2rad(2.0):
                step = np.clip(error_roll * 0.15, -0.05, 0.05) 
                combined_drot = combined_drot @ rotation_matrix(angle=step, direction=[1.0, 0.0, 0.0])[:3, :3] 

        # Pitch Takibi
        if self.target_pitch is not None: 
            error_pitch = norm_ang(self.target_pitch - pitch) 
            if abs(error_pitch) > np.deg2rad(2.0):
                step = np.clip(error_pitch * 0.15, -0.05, 0.05) 
                combined_drot = combined_drot @ rotation_matrix(angle=step, direction=[0.0, 1.0, 0.0])[:3, :3] 

        # Yaw Takibi
        if self.target_yaw is not None: 
            error_yaw = norm_ang(self.target_yaw - yaw) 
            if abs(error_yaw) > np.deg2rad(2.0):
                step = np.clip(error_yaw * 0.15, -0.05, 0.05) 
                combined_drot = combined_drot @ rotation_matrix(angle=step, direction=[0.0, 0.0, 1.0])[:3, :3] 

        # --- POZİSYON TAKİBİ --- 
        if self.target_px is not None: 
            error_x = self.target_px - px 
            if abs(error_x) > 0.005: dpos[0] = np.clip(error_x * 0.12, -0.007, 0.007) 

        if self.target_py is not None: 
            error_y = self.target_py - py 
            if abs(error_y) > 0.005: dpos[1] = np.clip(error_y * 0.12, -0.007, 0.007) 

        if self.target_pz is not None: 
            error_z = self.target_pz - pz 
            if abs(error_z) > 0.005: dpos[2] = np.clip(error_z * 0.12, -0.007, 0.007) 

        # --- C. SİSTEM KONTROLLERİ --- 
        if self.env.is_key_pressed_once(key=glfw.KEY_Z): 
            return np.zeros(7, dtype=np.float32), True 

        # Gripper Durumu
        self.gripper_state = gripper_state 
        g_val = 0.021 if self.gripper_state else 0.057
        
        # Birleştirilmiş rotasyon matrisini Euler (RPY) formatına çevir
        drot_rpy = r2rpy(combined_drot) 
        action = np.concatenate([dpos, drot_rpy, [g_val]], dtype=np.float32) 
        
        return action, False