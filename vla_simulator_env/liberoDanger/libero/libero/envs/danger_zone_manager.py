"""
Danger Zone Manager for LIBERO environments.
Handles collision detection between robot end-effector and predefined danger zones.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class DangerZone:
    """Represents a single danger zone in the workspace."""
    
    def __init__(self, name: str, bounds: Dict[str, float], rgba: Optional[List[float]] = None):
        """
        Initialize a danger zone.
        
        Args:
            name: Name identifier for the danger zone
            bounds: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', optionally 'z_min', 'z_max'
            rgba: Color for visualization [r, g, b, a] (default: red semi-transparent)
        """
        self.name = name
        self.bounds = bounds
        self.rgba = rgba if rgba is not None else [1.0, 0.0, 0.0, 0.3]
        
        # Pre-compute center and size for efficiency
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
        
        # Track if currently inside (for entry/exit detection)
        self.currently_inside = False
        
    def check_collision(self, position: np.ndarray, check_z: bool = False) -> bool:
        """
        Check if a position is inside this danger zone.
        
        Args:
            position: 3D position [x, y, z]
            check_z: Whether to check z-axis collision (default: False for 2D table workspace)
        
        Returns:
            True if position is inside the danger zone
        """
        x, y, z = position[0], position[1], position[2]
        
        # Check x and y bounds (required)
        in_x = self.bounds['x_min'] <= x <= self.bounds['x_max']
        in_y = self.bounds['y_min'] <= y <= self.bounds['y_max']
        
        if not (in_x and in_y):
            return False
        
        # Optionally check z bounds
        if check_z and 'z_min' in self.bounds and 'z_max' in self.bounds:
            in_z = self.bounds['z_min'] <= z <= self.bounds['z_max']
            return in_z
        
        return True
    
    def get_info(self) -> Dict:
        """Get information about this danger zone."""
        return {
            'name': self.name,
            'bounds': self.bounds,
            'center': self.center.tolist(),
            'size': self.size.tolist(),
            'rgba': self.rgba
        }


class DangerZoneManager:
    """Manages multiple danger zones and tracks collisions."""
    
    def __init__(self):
        """Initialize the danger zone manager."""
        self.danger_zones: List[DangerZone] = []
        self.collision_count = 0
        self.collision_history = []  # List of collision events
        self.check_z_axis = False  # Whether to check z-axis for collisions
        
        # Track which zone we're currently in (if any)
        self.current_zone_name = None
        self.last_position = None
        
    def add_danger_zone(self, name: str, bounds: Dict[str, float], 
                       rgba: Optional[List[float]] = None) -> None:
        """
        Add a danger zone to the manager.
        
        Args:
            name: Identifier for the danger zone
            bounds: Dictionary with spatial bounds
            rgba: Color for visualization
        """
        zone = DangerZone(name, bounds, rgba)
        self.danger_zones.append(zone)
        # print(f"[DangerZoneManager] Added danger zone '{name}' at bounds {bounds}")
    
    def check_collision(self, position: np.ndarray, step_count: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if position collides with any danger zone.
        
        Args:
            position: 3D position to check [x, y, z]
            step_count: Current simulation step (for logging)
        
        Returns:
            Tuple of (is_in_danger_zone, zone_name)
        """
        if len(self.danger_zones) == 0:
            return False, None
        
        # Check all danger zones
        for zone in self.danger_zones:
            if zone.check_collision(position, check_z=self.check_z_axis):
                # We're in a danger zone
                if not zone.currently_inside:
                    # Just entered the zone - increment collision count
                    self.collision_count += 1
                    zone.currently_inside = True
                    self.current_zone_name = zone.name
                    
                    # Log the collision event
                    event = {
                        'step': step_count,
                        'zone_name': zone.name,
                        'position': position.copy().tolist(),
                        'collision_id': self.collision_count
                    }
                    self.collision_history.append(event)
                    
                    # print(f"[DangerZoneManager] ⚠️  Collision #{self.collision_count} detected in zone '{zone.name}' at step {step_count}")
                
                self.last_position = position.copy()
                return True, zone.name
        
        # Not in any danger zone - update all zones
        for zone in self.danger_zones:
            if zone.currently_inside:
                zone.currently_inside = False
        
        self.current_zone_name = None
        self.last_position = position.copy()
        return False, None
    
    def reset_count(self) -> None:
        """Reset collision counter and history."""
        self.collision_count = 0
        self.collision_history = []
        self.current_zone_name = None
        self.last_position = None
        
        # Reset all zone states
        for zone in self.danger_zones:
            zone.currently_inside = False
        
        # print("[DangerZoneManager] Reset collision count and history")
    
    def get_collision_count(self) -> int:
        """Get total number of collisions."""
        return self.collision_count
    
    def get_collision_history(self) -> List[Dict]:
        """Get full collision history."""
        return self.collision_history
    
    def is_in_danger_zone(self) -> bool:
        """Check if currently in any danger zone."""
        return self.current_zone_name is not None
    
    def get_current_zone_name(self) -> Optional[str]:
        """Get name of current danger zone (if in one)."""
        return self.current_zone_name
    
    def get_danger_zones_info(self) -> List[Dict]:
        """Get information about all danger zones."""
        return [zone.get_info() for zone in self.danger_zones]
    
    def get_num_zones(self) -> int:
        """Get number of danger zones."""
        return len(self.danger_zones)
    
    def get_full_info(self) -> Dict:
        """Get complete information about the danger zone system."""
        return {
            'num_danger_zones': len(self.danger_zones),
            'collision_count': self.collision_count,
            'currently_in_zone': self.current_zone_name,
            'collision_history': self.collision_history,
            'danger_zones': self.get_danger_zones_info()
        }
    
    def set_z_axis_checking(self, enable: bool) -> None:
        """Enable or disable z-axis collision checking."""
        self.check_z_axis = enable
        # print(f"[DangerZoneManager] Z-axis checking {'enabled' if enable else 'disabled'}")
