"""
增强版危险区域管理器 - 考虑整个机械臂的碰撞检测
支持检测机械臂链上所有关节和连杆与危险区域的碰撞
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class DangerZone:
    """表示单个危险区域（与原版相同）"""
    
    def __init__(self, name: str, bounds: Dict[str, float], rgba: Optional[List[float]] = None):
        self.name = name
        self.bounds = bounds
        self.rgba = rgba if rgba is not None else [1.0, 0.0, 0.0, 0.3]
        
        self.center = np.array([
            (bounds['x_min'] + bounds['x_max']) / 2.0,
            (bounds['y_min'] + bounds['y_max']) / 2.0,
            (bounds.get('z_min', 0.0) + bounds.get('z_max', 1.0)) / 2.0
        ])
        
        self.size = np.array([
            (bounds['x_max'] - bounds['x_min']) / 2.0,
            (bounds['y_max'] - bounds['y_min']) / 2.0,
            (bounds.get('z_max', 1.0) - bounds.get('z_min', 0.0)) / 2.0
        ])
        
        self.currently_inside = False
        
    def check_collision(self, position: np.ndarray, check_z: bool = False) -> bool:
        """检查单个点是否在危险区域内"""
        x, y, z = position[0], position[1], position[2]
        
        in_x = self.bounds['x_min'] <= x <= self.bounds['x_max']
        in_y = self.bounds['y_min'] <= y <= self.bounds['y_max']
        
        if not (in_x and in_y):
            return False
        
        if check_z and 'z_min' in self.bounds and 'z_max' in self.bounds:
            in_z = self.bounds['z_min'] <= z <= self.bounds['z_max']
            return in_z
        
        return True
    
    def check_sphere_collision(self, position: np.ndarray, radius: float, check_z: bool = False) -> bool:
        """
        检查球体是否与危险区域碰撞（用于近似关节和连杆）
        
        Args:
            position: 球心位置
            radius: 球体半径
            check_z: 是否检查z轴
        
        Returns:
            True if collision detected
        """
        x, y, z = position[0], position[1], position[2]
        
        # 找到危险区域（AABB）上最近的点
        closest_x = np.clip(x, self.bounds['x_min'], self.bounds['x_max'])
        closest_y = np.clip(y, self.bounds['y_min'], self.bounds['y_max'])
        
        if check_z and 'z_min' in self.bounds and 'z_max' in self.bounds:
            closest_z = np.clip(z, self.bounds['z_min'], self.bounds['z_max'])
        else:
            closest_z = z
        
        # 计算球心到最近点的距离
        distance = np.sqrt(
            (x - closest_x)**2 + 
            (y - closest_y)**2 + 
            (z - closest_z)**2
        )
        
        return distance <= radius
    
    def check_capsule_collision(self, point1: np.ndarray, point2: np.ndarray, 
                               radius: float, check_z: bool = False) -> bool:
        """
        检查胶囊体（两个球心连线+半径）是否与危险区域碰撞
        用于近似机械臂连杆
        
        Args:
            point1, point2: 胶囊体两端点
            radius: 胶囊体半径
            check_z: 是否检查z轴
        
        Returns:
            True if collision detected
        """
        # 简化：在两点之间采样多个点进行球体检测
        num_samples = 5
        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_point = point1 + t * (point2 - point1)
            if self.check_sphere_collision(sample_point, radius, check_z):
                return True
        return False
    
    def get_info(self) -> Dict:
        """获取危险区域信息"""
        return {
            'name': self.name,
            'bounds': self.bounds,
            'center': self.center.tolist(),
            'size': self.size.tolist(),
            'rgba': self.rgba
        }


class FullArmDangerZoneManager:
    """增强版危险区域管理器 - 考虑整个机械臂"""
    
    def __init__(self, check_full_arm: bool = True):
        """
        初始化管理器
        
        Args:
            check_full_arm: 是否检查整个机械臂（True）还是只检查末端（False）
        """
        self.danger_zones: List[DangerZone] = []
        
        # 三种碰撞计数语义
        self.collision_enter_count = 0      # 进入危险区域的次数（每次进入计数+1）
        self.collision_frame_count = 0      # 接触帧数（每帧在危险区域内都计数+1）
        self.collision_event_count = 0      # 完整碰撞事件数（进入+离开算一次）
        
        # 向后兼容：collision_count = collision_enter_count
        self.collision_count = 0
        
        self.collision_history = []
        self.check_z_axis = False
        self.check_full_arm = check_full_arm
        
        self.current_zone_name = None
        self.last_position = None
        self.was_in_zone_last_step = False  # 用于检测离开事件
        
        # 机械臂近似参数（Panda机器人）
        self.joint_radius = 0.06   # 关节半径（米）
        self.link_radius = 0.04    # 连杆半径（米）
        self.gripper_radius = 0.05 # 夹爪半径（米）
        
        # 碰撞详情
        self.last_collision_details = {
            'collision_type': None,  # 'eef', 'joint', 'link', 'gripper'
            'collision_body': None,  # 具体哪个关节或连杆
            'collision_point': None, # 碰撞点位置
        }
        
    def add_danger_zone(self, name: str, bounds: Dict[str, float], 
                       rgba: Optional[List[float]] = None) -> None:
        """添加危险区域"""
        zone = DangerZone(name, bounds, rgba)
        self.danger_zones.append(zone)
        mode = "完整机械臂" if self.check_full_arm else "仅末端"
        print(f"[FullArmDangerZoneManager] 添加危险区域 '{name}' (检测模式: {mode})")
    
    def check_collision_simple(self, eef_position: np.ndarray, 
                               step_count: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        简化版碰撞检测 - 只检查末端执行器（向后兼容）
        
        Args:
            eef_position: 末端执行器位置 [x, y, z]
            step_count: 当前步数
        
        Returns:
            (is_in_danger_zone, zone_name)
        """
        if len(self.danger_zones) == 0:
            return False, None
        
        collision_detected = False
        collision_zone = None
        
        for zone in self.danger_zones:
            # 使用球体碰撞检测，考虑末端执行器半径（与完整模式一致）
            if zone.check_sphere_collision(eef_position, self.gripper_radius, self.check_z_axis):
                collision_detected = True
                collision_zone = zone.name
                
                # 每帧在危险区域内都计数
                self.collision_frame_count += 1
                
                if not zone.currently_inside:
                    # 新的进入事件
                    self.collision_enter_count += 1
                    self.collision_count = self.collision_enter_count  # 向后兼容
                    zone.currently_inside = True
                    self.current_zone_name = zone.name
                    
                    # 记录详细信息
                    self.last_collision_details = {
                        'collision_type': 'eef',
                        'collision_body': None,
                        'collision_point': eef_position.copy().tolist(),
                    }
                    
                    event = {
                        'step': step_count,
                        'zone_name': zone.name,
                        'position': eef_position.copy().tolist(),
                        'collision_id': self.collision_enter_count,
                        'collision_type': 'eef',
                        'event_type': 'enter',
                    }
                    self.collision_history.append(event)
                    
                    print(f"[FullArmDangerZoneManager] ⚠️  进入碰撞 #{self.collision_enter_count} "
                          f"在区域 '{zone.name}' - eef - 步骤 {step_count}")
                
                self.was_in_zone_last_step = True
                self.last_position = eef_position.copy()
                return True, zone.name
        
        # 没有碰撞 - 检查是否刚离开危险区域
        if self.was_in_zone_last_step:
            # 完整碰撞事件结束（进入+离开）
            self.collision_event_count += 1
            
            # 记录离开事件
            if step_count is not None:
                event = {
                    'step': step_count,
                    'zone_name': self.current_zone_name,
                    'event_type': 'exit',
                    'collision_event_id': self.collision_event_count,
                }
                self.collision_history.append(event)
                print(f"[FullArmDangerZoneManager] ✅ 离开危险区域 '{self.current_zone_name}' - "
                      f"完整事件 #{self.collision_event_count} - 步骤 {step_count}")
        
        for zone in self.danger_zones:
            if zone.currently_inside:
                zone.currently_inside = False
        
        self.current_zone_name = None
        self.was_in_zone_last_step = False
        self.last_position = eef_position.copy()
        return False, None
    
    def check_collision_full_arm(self, robot_state: Dict, 
                                 step_count: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        完整版碰撞检测 - 检查整个机械臂
        
        Args:
            robot_state: 机器人状态字典，包含：
                - 'eef_pos': 末端执行器位置 [x, y, z]
                - 'joint_positions': 所有关节位置列表 [[x,y,z], ...]（可选）
                - 'link_positions': 所有连杆中心位置（可选）
                - 'body_positions': MuJoCo body位置（可选）
            step_count: 当前步数
        
        Returns:
            (is_in_danger_zone, zone_name)
        """
        if len(self.danger_zones) == 0:
            return False, None
        
        eef_pos = robot_state.get('eef_pos')
        if eef_pos is None:
            return False, None
        
        collision_detected = False
        collision_zone = None
        collision_type = None
        collision_body = None
        collision_point = None
        
        # 检查所有危险区域
        for zone in self.danger_zones:
            # 1. 检查末端执行器（使用球体近似）
            
            if zone.check_sphere_collision(eef_pos, self.gripper_radius, self.check_z_axis):
                collision_detected = True
                collision_zone = zone.name
                collision_type = 'eef'
                collision_point = eef_pos.copy()
                # print(f"末端执行器碰撞: {collision_point}")
                
            
            # 2. 检查关节（如果提供）
            if not collision_detected and 'joint_positions' in robot_state:
                for i, joint_pos in enumerate(robot_state['joint_positions']):   # 7个关节位置是否碰撞
                    if zone.check_sphere_collision(joint_pos, self.joint_radius, self.check_z_axis):
                        collision_detected = True
                        collision_zone = zone.name
                        collision_type = 'joint'
                        collision_body = f'joint_{i}'
                        collision_point = joint_pos.copy()
                        # print(f"关节碰撞: {collision_point}")
                        break
            
            # 3. 检查连杆（如果提供）
            # import pdb; pdb.set_trace()
            if not collision_detected and 'link_segments' in robot_state:
                for i, (p1, p2) in enumerate(robot_state['link_segments']):
                    if zone.check_capsule_collision(p1, p2, self.link_radius, self.check_z_axis):
                        collision_detected = True
                        collision_zone = zone.name
                        collision_type = 'link'
                        collision_body = f'link_{i}'
                        collision_point = ((p1 + p2) / 2).copy()
                        # print(f"连杆碰撞: {collision_point}")
                        break
            
            if collision_detected:
                # 每帧在危险区域内都计数
                self.collision_frame_count += 1
                
                # 检查是否是新的进入事件
                if not zone.currently_inside:
                    self.collision_enter_count += 1
                    self.collision_count = self.collision_enter_count  # 向后兼容
                    zone.currently_inside = True
                    self.current_zone_name = zone.name
                    
                    # 记录详细信息
                    self.last_collision_details = {
                        'collision_type': collision_type,
                        'collision_body': collision_body,
                        'collision_point': collision_point.tolist() if collision_point is not None else None,
                    }
                    
                    event = {
                        'step': step_count,
                        'zone_name': zone.name,
                        'position': collision_point.tolist() if collision_point is not None else eef_pos.tolist(),
                        'collision_id': self.collision_enter_count,
                        'collision_type': collision_type,
                        'collision_body': collision_body,
                        'event_type': 'enter',
                    }
                    self.collision_history.append(event)
                    
                    collision_part = f"{collision_type}"
                    if collision_body:
                        collision_part += f" ({collision_body})"
                    
                    print(f"[FullArmDangerZoneManager] ⚠️  进入碰撞 #{self.collision_enter_count} "
                          f"在区域 '{zone.name}' - {collision_part} - 步骤 {step_count}")
                
                self.was_in_zone_last_step = True
                return True, zone.name
        
        # 没有碰撞 - 检查是否刚离开危险区域
        if self.was_in_zone_last_step:
            # 完整碰撞事件结束（进入+离开）
            self.collision_event_count += 1
            
            # 记录离开事件
            if step_count is not None:
                event = {
                    'step': step_count,
                    'zone_name': self.current_zone_name,
                    'event_type': 'exit',
                    'collision_event_id': self.collision_event_count,
                }
                self.collision_history.append(event)
                print(f"[FullArmDangerZoneManager] ✅ 离开危险区域 '{self.current_zone_name}' - "
                      f"完整事件 #{self.collision_event_count} - 步骤 {step_count}")
        
        for zone in self.danger_zones:
            if zone.currently_inside:
                zone.currently_inside = False
        
        self.current_zone_name = None
        self.was_in_zone_last_step = False
        return False, None
    
    def check_collision(self, robot_state, step_count: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        统一的碰撞检测接口
        
        Args:
            robot_state: 可以是：
                - np.ndarray: 末端执行器位置（简化模式）
                - Dict: 完整机器人状态（完整模式）
            step_count: 当前步数
        
        Returns:
            (is_in_danger_zone, zone_name)
        """
        # 判断输入类型
        
        if isinstance(robot_state, np.ndarray):
            # 简化模式：只有末端位置
            return self.check_collision_simple(robot_state, step_count)
        
        elif isinstance(robot_state, dict) and self.check_full_arm:
            # 完整模式：检查整个机械臂
            return self.check_collision_full_arm(robot_state, step_count)
        
        elif isinstance(robot_state, dict):
            # 字典但只检查末端
            eef_pos = robot_state.get('eef_pos')
            if eef_pos is not None:
                return self.check_collision_simple(eef_pos, step_count)
        
        return False, None
    
    def reset_count(self) -> None:
        """重置所有碰撞计数和状态"""
        self.collision_enter_count = 0
        self.collision_frame_count = 0
        self.collision_event_count = 0
        self.collision_count = 0  # 向后兼容
        self.collision_history = []
        self.current_zone_name = None
        self.last_position = None
        self.was_in_zone_last_step = False
        self.last_collision_details = {
            'collision_type': None,
            'collision_body': None,
            'collision_point': None,
        }
        
        for zone in self.danger_zones:
            zone.currently_inside = False
        
        print("[FullArmDangerZoneManager] 重置所有碰撞计数和历史")
    
    def get_collision_count(self) -> int:
        """获取总碰撞次数"""
        return self.collision_count
    
    def get_collision_history(self) -> List[Dict]:
        """获取完整碰撞历史"""
        return self.collision_history
    
    def is_in_danger_zone(self) -> bool:
        """检查是否当前在危险区域内"""
        return self.current_zone_name is not None
    
    def get_current_zone_name(self) -> Optional[str]:
        """获取当前危险区域名称"""
        return self.current_zone_name
    
    def get_danger_zones_info(self) -> List[Dict]:
        """获取所有危险区域信息"""
        return [zone.get_info() for zone in self.danger_zones]
    
    def get_num_zones(self) -> int:
        """获取危险区域数量"""
        return len(self.danger_zones)
    
    def get_full_info(self) -> Dict:
        """
        获取完整信息，包括三种碰撞计数语义
        
        Returns:
            Dict 包含：
                - collision_enter_count: 进入次数（每次进入危险区域+1）
                - collision_frame_count: 接触帧数（每帧在危险区域内+1）
                - collision_event_count: 完整事件数（进入+离开算一次）
                - collision_count: 向后兼容，等于collision_enter_count
        """
        return {
            'num_danger_zones': len(self.danger_zones),
            'collision_count': self.collision_count,  # 向后兼容
            'collision_enter_count': self.collision_enter_count,
            'collision_frame_count': self.collision_frame_count,
            'collision_event_count': self.collision_event_count,
            'currently_in_zone': self.current_zone_name,
            'collision_history': self.collision_history,
            'danger_zones': self.get_danger_zones_info(),
            'check_mode': 'full_arm' if self.check_full_arm else 'eef_only',
            'last_collision_details': self.last_collision_details,
        }
    
    def set_z_axis_checking(self, enable: bool) -> None:
        """启用或禁用z轴检测"""
        self.check_z_axis = enable
        print(f"[FullArmDangerZoneManager] Z轴检测 {'启用' if enable else '禁用'}")
    
    def set_collision_radii(self, joint_radius: float = None, 
                           link_radius: float = None, 
                           gripper_radius: float = None) -> None:
        """
        设置碰撞检测的半径参数
        
        Args:
            joint_radius: 关节半径（米）
            link_radius: 连杆半径（米）
            gripper_radius: 夹爪半径（米）
        """
        if joint_radius is not None:
            self.joint_radius = joint_radius
        if link_radius is not None:
            self.link_radius = link_radius
        if gripper_radius is not None:
            self.gripper_radius = gripper_radius
        
        print(f"[FullArmDangerZoneManager] 更新碰撞半径: "
              f"关节={self.joint_radius:.3f}m, "
              f"连杆={self.link_radius:.3f}m, "
              f"夹爪={self.gripper_radius:.3f}m")

