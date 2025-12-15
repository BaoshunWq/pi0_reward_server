"""
机械臂状态提取器
从MuJoCo仿真中提取机械臂的关节和连杆位置，用于完整碰撞检测
"""

import numpy as np
from typing import Dict, List, Tuple


class RobotArmExtractor:
    """从MuJoCo环境中提取机械臂状态"""
    
    def __init__(self, robot_name: str = "robot0"):
        """
        初始化提取器
        
        Args:
            robot_name: 机器人名称（默认 "robot0"）
        """
        self.robot_name = robot_name
        
        # Panda机器人的关节名称（MuJoCo）
        self.panda_joint_names = [
            f"{robot_name}_joint1",
            f"{robot_name}_joint2",
            f"{robot_name}_joint3",
            f"{robot_name}_joint4",
            f"{robot_name}_joint5",
            f"{robot_name}_joint6",
            f"{robot_name}_joint7",
        ]
        
        # Panda机器人的连杆body名称
        self.panda_link_names = [
            f"{robot_name}_link0",
            f"{robot_name}_link1", 
            f"{robot_name}_link2",
            f"{robot_name}_link3",
            f"{robot_name}_link4",
            f"{robot_name}_link5",
            f"{robot_name}_link6",
            f"{robot_name}_link7",
        ]
        
        # 夹爪相关body
        self.gripper_body_names = [
            f"{robot_name}_hand",
            f"{robot_name}_leftfinger",
            f"{robot_name}_rightfinger",
        ]
        
        self.cached_body_ids = None
        self.cached_joint_ids = None
    
    def _get_body_ids(self, sim):
        """获取并缓存body ID"""
        if self.cached_body_ids is None:
            self.cached_body_ids = {}
            
            # 获取连杆body ID
            for link_name in self.panda_link_names:
                try:
                    body_id = sim.model.body_name2id(link_name)
                    self.cached_body_ids[link_name] = body_id
                except:
                    pass  # 某些连杆可能不存在
            
            # 获取夹爪body ID
            for gripper_name in self.gripper_body_names:
                try:
                    body_id = sim.model.body_name2id(gripper_name)
                    self.cached_body_ids[gripper_name] = body_id
                except:
                    pass
        
        return self.cached_body_ids
    
    def _get_joint_ids(self, sim):
        """获取并缓存joint ID"""
        if self.cached_joint_ids is None:
            self.cached_joint_ids = {}
            
            for joint_name in self.panda_joint_names:
                try:
                    joint_id = sim.model.joint_name2id(joint_name)
                    self.cached_joint_ids[joint_name] = joint_id
                except:
                    pass
        
        return self.cached_joint_ids
    
    def extract_joint_positions(self, sim) -> List[np.ndarray]:
        """
        提取所有关节的位置
        
        Args:
            sim: MuJoCo仿真对象
        
        Returns:
            List of joint positions [[x,y,z], ...]
        """
        joint_ids = self._get_joint_ids(sim)
        joint_positions = []
        
        for joint_name in self.panda_joint_names:
            if joint_name in joint_ids:
                # 关节位置通常需要通过对应的body获取
                # MuJoCo中关节本身没有位置，需要找到关节对应的body
                try:
                    # 尝试从joint名称推导body名称
                    link_name = joint_name.replace("_joint", "_link")
                    body_id = sim.model.body_name2id(link_name)
                    pos = sim.data.body_xpos[body_id].copy()
                    joint_positions.append(pos)
                except:
                    pass
        
        return joint_positions
    
    def extract_link_positions(self, sim) -> List[np.ndarray]:
        """
        提取所有连杆中心的位置
        
        Args:
            sim: MuJoCo仿真对象
        
        Returns:
            List of link center positions [[x,y,z], ...]
        """
        body_ids = self._get_body_ids(sim)
        link_positions = []
        
        for link_name in self.panda_link_names:
            if link_name in body_ids:
                body_id = body_ids[link_name]
                pos = sim.data.body_xpos[body_id].copy()
                link_positions.append(pos)
        
        return link_positions
    
    def extract_link_segments(self, sim) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        提取连杆的线段（用于胶囊体碰撞检测）
        每段连杆近似为从一个关节到下一个关节的线段
        
        Args:
            sim: MuJoCo仿真对象
        
        Returns:
            List of (start_pos, end_pos) tuples
        """
        joint_positions = self.extract_joint_positions(sim)
        
        if len(joint_positions) < 2:
            return []
        
        # 创建连续关节之间的线段
        segments = []
        for i in range(len(joint_positions) - 1):
            segments.append((joint_positions[i], joint_positions[i+1]))
        
        # 添加最后一个关节到末端执行器的线段
        try:
            # 获取末端执行器位置
            eef_site_id = sim.model.site_name2id(f"{self.robot_name}_grip_site")
            eef_pos = sim.data.site_xpos[eef_site_id].copy()
            if len(joint_positions) > 0:
                segments.append((joint_positions[-1], eef_pos))
        except:
            pass
        
        return segments
    
    def extract_gripper_bodies(self, sim) -> List[Tuple[np.ndarray, float]]:
        """
        提取夹爪的body位置和近似半径
        
        Args:
            sim: MuJoCo仿真对象
        
        Returns:
            List of (position, radius) tuples
        """
        body_ids = self._get_body_ids(sim)
        gripper_bodies = []
        
        for gripper_name in self.gripper_body_names:
            if gripper_name in body_ids:
                body_id = body_ids[gripper_name]
                pos = sim.data.body_xpos[body_id].copy()
                # 使用固定半径（可以根据实际geometry调整）
                radius = 0.03  # 3cm
                gripper_bodies.append((pos, radius))
        
        return gripper_bodies
    
    def extract_full_arm_state(self, sim, eef_pos: np.ndarray = None) -> Dict:
        """
        提取完整的机械臂状态（用于完整碰撞检测）
        
        Args:
            sim: MuJoCo仿真对象
            eef_pos: 末端执行器位置（如果已知）
        
        Returns:
            Dict containing:
                - 'eef_pos': 末端执行器位置
                - 'joint_positions': 关节位置列表
                - 'link_positions': 连杆位置列表
                - 'link_segments': 连杆线段列表
                - 'gripper_bodies': 夹爪body列表
        """
        # 获取末端执行器位置
        if eef_pos is None:
            try:
                eef_site_id = sim.model.site_name2id(f"{self.robot_name}_grip_site")
                eef_pos = sim.data.site_xpos[eef_site_id].copy()
            except:
                # 如果找不到site，尝试使用hand body
                try:
                    hand_body_id = sim.model.body_name2id(f"{self.robot_name}_hand")
                    eef_pos = sim.data.body_xpos[hand_body_id].copy()
                except:
                    eef_pos = np.array([0.0, 0.0, 0.0])
        
        # 提取所有信息
        robot_state = {
            'eef_pos': eef_pos,
            'joint_positions': self.extract_joint_positions(sim),
            'link_positions': self.extract_link_positions(sim),
            'link_segments': self.extract_link_segments(sim),
            'gripper_bodies': self.extract_gripper_bodies(sim),
        }
        
        return robot_state
    
    def get_simple_approximation(self, sim, eef_pos: np.ndarray = None,
                                 num_samples: int = 5) -> Dict:
        """
        获取简化的机械臂近似（如果无法提取精确信息）
        通过在base到eef之间采样点来近似整个机械臂
        
        Args:
            sim: MuJoCo仿真对象
            eef_pos: 末端执行器位置
            num_samples: 采样点数量
        
        Returns:
            Dict containing simplified arm state
        """
        # 获取末端位置
        if eef_pos is None:
            try:
                eef_site_id = sim.model.site_name2id(f"{self.robot_name}_grip_site")
                eef_pos = sim.data.site_xpos[eef_site_id].copy()
            except:
                eef_pos = np.array([0.0, 0.0, 1.0])
        
        # 获取base位置
        try:
            base_body_id = sim.model.body_name2id(f"{self.robot_name}_link0")
            base_pos = sim.data.body_xpos[base_body_id].copy()
        except:
            # 默认base位置
            base_pos = np.array([0.0, 0.0, 0.9])
        
        # 在base和eef之间线性插值采样点
        sampled_positions = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            pos = base_pos + t * (eef_pos - base_pos)
            sampled_positions.append(pos)
        
        # 创建简化的线段
        segments = []
        for i in range(len(sampled_positions) - 1):
            segments.append((sampled_positions[i], sampled_positions[i+1]))
        
        robot_state = {
            'eef_pos': eef_pos,
            'joint_positions': sampled_positions,
            'link_positions': sampled_positions,
            'link_segments': segments,
            'gripper_bodies': [(eef_pos, 0.05)],
        }
        
        return robot_state

