"""
Multi-view environment wrapper for cross-view generalization data collection.

This module provides a wrapper that adds multiple camera views to robosuite/robomimic
environments by modifying the MuJoCo XML and reloading the simulation. This approach
uses the existing `add_camera_to_xml()` utility for stable camera integration.

Example:
    >>> from demo_aug.configs.multi_view_config import MultiViewCameraConfig
    >>> from demo_aug.envs.wrapper.multi_view_wrapper import MultiViewEnvWrapper
    >>> 
    >>> config = MultiViewCameraConfig(num_third_views=4, seed=42)
    >>> env = EnvUtils.create_env_from_metadata(env_meta, ...)
    >>> multi_view_env = MultiViewEnvWrapper(env, config)
    >>> 
    >>> # Get all camera names
    >>> print(multi_view_env.camera_names)
    >>> # ['agentview', 'robot0_eye_in_hand', 'third_view_0', ...]
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from demo_aug.configs.multi_view_config import (
    MultiViewCameraConfig, 
    ThirdViewCameraConfig,
    EyeInHandCameraConfig,
)
from demo_aug.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)
from demo_aug.utils.mujoco_utils import add_camera_to_xml

logger = logging.getLogger(__name__)


class MultiViewEnvWrapper:
    """
    Wrapper for robosuite/robomimic environments that adds multi-view cameras.
    
    This wrapper modifies the MuJoCo XML to add custom cameras, then reloads
    the simulation. This is the proper way to add cameras, ensuring they are
    fully integrated into the rendering pipeline.
    
    Attributes:
        env: The wrapped environment (robosuite/robomimic)
        config: Multi-view camera configuration
        camera_names: List of all camera names (original + added)
        added_camera_names: List of camera names that were added by this wrapper
    """
    
    def __init__(
        self,
        env,
        config: MultiViewCameraConfig,
    ):
        """
        Initialize the multi-view wrapper.
        
        Args:
            env: Robosuite/robomimic environment to wrap
            config: Multi-view camera configuration
        """
        self.env = env
        self.config = config
        self.added_camera_names: List[str] = []
        self._sampled_params: Dict[str, Dict] = {}  # Store sampling parameters
        self._cameras_added = False
        
        # Add cameras to the environment
        self._add_cameras_to_env()
    
    def _add_cameras_to_env(self):
        """Add sampled cameras to the environment by modifying XML."""
        if self._cameras_added:
            logger.warning("Cameras already added, skipping")
            return
        
        rng = np.random.default_rng(self.config.seed)
        cfg = self.config.third_view_config
        
        # Get the current simulation state and XML
        sim = self._get_sim()
        if sim is None:
            raise RuntimeError("Cannot access MuJoCo simulation object")
        
        initial_state = sim.get_state().flatten()
        xml = sim.model.get_xml()
        
        # Sample camera poses and add to XML
        sampled_azimuths = []
        max_attempts = 100
        
        # ----- Add third-person views -----
        if self.config.use_zone_sampling:
            # Zone-based sampling: distribute cameras across predefined zones
            xml = self._add_cameras_with_zones(xml, rng)
        else:
            # Default: sample all cameras from third_view_config ranges
            for i in range(self.config.num_third_views):
                name = f"{cfg.name}_{i}"
                
                # Try to sample a camera pose with minimum angular separation
                for attempt in range(max_attempts):
                    if cfg.is_random:
                        azimuth = rng.uniform(*cfg.azimuth_range)
                        elevation = rng.uniform(*cfg.elevation_range)
                        distance = rng.uniform(*cfg.distance_range)
                    else:
                        # Evenly spaced azimuths for deterministic placement
                        azimuth_span = cfg.azimuth_range[1] - cfg.azimuth_range[0]
                        azimuth = cfg.azimuth_range[0] + (i + 0.5) * azimuth_span / self.config.num_third_views
                        elevation = (cfg.elevation_range[0] + cfg.elevation_range[1]) / 2
                        distance = (cfg.distance_range[0] + cfg.distance_range[1]) / 2
                    
                    # Check angular separation
                    is_valid = all(
                        abs(azimuth - prev_az) >= self.config.min_angular_separation
                        for prev_az in sampled_azimuths
                    )
                    
                    if is_valid or not cfg.is_random:
                        break
                
                sampled_azimuths.append(azimuth)
                
                # Store sampling parameters
                self._sampled_params[name] = {
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "distance": distance,
                }
                
                # Convert to position and quaternion
                pos, quat_wxyz = self._spherical_to_pos_quat(
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    distance=distance,
                    target=np.array(cfg.target),
                )
                
                # Add camera to XML
                pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_str = f"{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}"
                
                xml = add_camera_to_xml(
                    xml=xml,
                    camera_name=name,
                    camera_pos=pos_str,
                    camera_quat=quat_str,
                    is_eye_in_hand_camera=False,
                    fovy=str(cfg.fov),
                )
                
                self.added_camera_names.append(name)
                
                logger.info(f"Added camera '{name}': azimuth={azimuth:.1f}°, "
                           f"elevation={elevation:.1f}°, distance={distance:.2f}m")
        
        # ----- Add wrist-mounted cameras (eye-in-hand) -----
        if self.config.num_wrist_views > 0:
            wrist_cfg = self.config.wrist_view_config
            
            for i in range(self.config.num_wrist_views):
                name = f"{wrist_cfg.name}_{i}"
                
                # Sample or compute position around gripper
                if wrist_cfg.is_random:
                    azimuth = rng.uniform(*wrist_cfg.azimuth_range)
                    elevation = rng.uniform(*wrist_cfg.elevation_range)
                    distance = rng.uniform(*wrist_cfg.distance_range)
                else:
                    # Evenly spaced for deterministic placement
                    azimuth_span = wrist_cfg.azimuth_range[1] - wrist_cfg.azimuth_range[0]
                    azimuth = wrist_cfg.azimuth_range[0] + (i + 0.5) * azimuth_span / self.config.num_wrist_views
                    elevation = (wrist_cfg.elevation_range[0] + wrist_cfg.elevation_range[1]) / 2
                    distance = (wrist_cfg.distance_range[0] + wrist_cfg.distance_range[1]) / 2
                
                self._sampled_params[name] = {
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "distance": distance,
                    "is_wrist": True,
                }
                
                # Convert to local position and quaternion (relative to gripper)
                pos, quat_wxyz = self._spherical_to_pos_quat_local(
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    distance=distance,
                    target=np.array(wrist_cfg.target),
                )
                
                pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_str = f"{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}"
                
                xml = add_camera_to_xml(
                    xml=xml,
                    camera_name=name,
                    camera_pos=pos_str,
                    camera_quat=quat_str,
                    parent_body_name="robot0_right_hand",  # Attach to gripper
                    is_eye_in_hand_camera=True,
                    fovy=str(wrist_cfg.fov),
                )
                
                self.added_camera_names.append(name)
                
                logger.info(f"Added wrist camera '{name}': azimuth={azimuth:.1f}°, "
                           f"elevation={elevation:.1f}°, distance={distance:.2f}m")
        
        # Reload environment with new XML
        self._reload_env_with_xml(xml, initial_state)
        self._cameras_added = True
        
        logger.info(f"Successfully added {len(self.added_camera_names)} cameras to environment")
    
    def _add_cameras_with_zones(self, xml: str, rng: np.random.Generator) -> str:
        """Add cameras distributed across predefined sampling zones.
        
        Cameras are allocated to zones based on zone weights, ensuring diversity.
        """
        from demo_aug.configs.multi_view_config import SAMPLING_ZONES, DEFAULT_TARGET
        
        cfg = self.config.third_view_config
        num_cameras = self.config.num_third_views
        
        if num_cameras == 0:
            return xml
        
        # Calculate camera allocation per zone based on weights
        zone_names = list(SAMPLING_ZONES.keys())
        weights = [SAMPLING_ZONES[z].get("weight", 1) for z in zone_names]
        total_weight = sum(weights)
        
        # Allocate cameras proportionally to weights
        zone_allocation = []
        remaining = num_cameras
        for i, (zone_name, weight) in enumerate(zip(zone_names, weights)):
            if i == len(zone_names) - 1:
                # Last zone gets remaining cameras
                count = remaining
            else:
                count = round(num_cameras * weight / total_weight)
                remaining -= count
            zone_allocation.append((zone_name, max(0, count)))
        
        logger.info(f"Zone allocation: {zone_allocation}")
        
        # Add cameras for each zone
        camera_idx = 0
        for zone_name, count in zone_allocation:
            if count == 0:
                continue
                
            zone = SAMPLING_ZONES[zone_name]
            
            for j in range(count):
                name = f"{cfg.name}_{camera_idx}"
                
                # Sample within zone ranges
                azimuth = rng.uniform(*zone["azimuth_range"])
                elevation = rng.uniform(*zone["elevation_range"])
                distance = rng.uniform(*zone["distance_range"])
                
                self._sampled_params[name] = {
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "distance": distance,
                    "zone": zone_name,
                }
                
                pos, quat_wxyz = self._spherical_to_pos_quat(
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    distance=distance,
                    target=np.array(DEFAULT_TARGET),
                )
                
                pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_str = f"{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}"
                
                xml = add_camera_to_xml(
                    xml=xml,
                    camera_name=name,
                    camera_pos=pos_str,
                    camera_quat=quat_str,
                    is_eye_in_hand_camera=False,
                    fovy=str(cfg.fov),
                )
                
                self.added_camera_names.append(name)
                
                logger.info(f"Added camera '{name}' in zone '{zone_name}': "
                           f"azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, distance={distance:.2f}m")
                
                camera_idx += 1
        
        return xml
    
    def _spherical_to_pos_quat(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates to camera position and quaternion.
        
        Args:
            azimuth_deg: Horizontal angle from front (degrees), 0 = +Y direction
            elevation_deg: Angle above horizontal plane (degrees)
            distance: Distance from target point (meters)
            target: 3D point camera looks at
            
        Returns:
            Tuple of (position, quaternion_wxyz)
        """
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        
        # Camera position in world frame
        position = np.array([
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.cos(elevation) * np.cos(azimuth),
            distance * np.sin(elevation),
        ]) + target
        
        # Compute look-at rotation
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # Handle case where forward is parallel to up
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        true_up = np.cross(right, forward)
        
        # Build rotation matrix (camera convention: -Z forward, X right, Y up)
        # MuJoCo camera: -Z is the viewing direction
        rot_matrix = np.eye(3)
        rot_matrix[:, 0] = right
        rot_matrix[:, 1] = true_up
        rot_matrix[:, 2] = -forward  # -Z points forward
        
        # Convert to quaternion (wxyz for MuJoCo)
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return position, quat_wxyz
    
    def _spherical_to_pos_quat_local(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates to camera position and quaternion in local (gripper) frame.
        
        For eye-in-hand cameras, the coordinate system is relative to the gripper:
        - target is an offset from gripper origin (e.g., 0.25m forward)
        - The camera is positioned around this target point
        - The result is expressed in the gripper's local coordinate frame
        
        Args:
            azimuth_deg: Horizontal angle in degrees (0 = forward in gripper frame)
            elevation_deg: Angle above/below horizontal in degrees
            distance: Distance from target point (meters)
            target: 3D offset from gripper origin (in gripper frame)
            
        Returns:
            Tuple of (position_local, quaternion_wxyz)
        """
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        
        # Camera position in gripper local frame
        # For gripper frame: X is typically forward (along gripper axis)
        position = np.array([
            target[0] - distance * np.cos(elevation) * np.cos(azimuth),  # X: forward
            distance * np.cos(elevation) * np.sin(azimuth),              # Y: left/right
            target[2] + distance * np.sin(elevation),                    # Z: up/down
        ])
        
        # Compute look-at rotation (looking at target from position)
        forward = target - position
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])  # Default forward
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        true_up = np.cross(right, forward)
        
        # Build rotation matrix (MuJoCo camera: -Z is viewing direction)
        rot_matrix = np.eye(3)
        rot_matrix[:, 0] = right
        rot_matrix[:, 1] = true_up
        rot_matrix[:, 2] = -forward
        
        # Convert to quaternion (wxyz for MuJoCo)
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return position, quat_wxyz

    def _reload_env_with_xml(self, xml: str, initial_state: np.ndarray):
        """Reload the environment with modified XML and restore state."""
        # Get the base env
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'reset_from_xml_string'):
            base_env = self.env.env
        elif hasattr(self.env, 'base_env') and hasattr(self.env.base_env, 'reset_from_xml_string'):
            base_env = self.env.base_env
        else:
            raise RuntimeError("Cannot find reset_from_xml_string method on environment")
        
        # Update camera lists with correct sizes for each camera type
        third_cfg = self.config.third_view_config
        wrist_cfg = self.config.wrist_view_config
        
        for cam_name in self.added_camera_names:
            base_env.camera_names.append(cam_name)
            
            # Check if this is a wrist camera
            if cam_name in self._sampled_params and self._sampled_params[cam_name].get("is_wrist", False):
                base_env.camera_heights.append(wrist_cfg.height)
                base_env.camera_widths.append(wrist_cfg.width)
            else:
                base_env.camera_heights.append(third_cfg.height)
                base_env.camera_widths.append(third_cfg.width)
            
            base_env.camera_depths.append(True)
        
        base_env.num_cameras = len(base_env.camera_names)
        
        # Reload simulation with new XML
        base_env.reset_from_xml_string(xml)
        base_env.sim.reset()
        base_env.sim.set_state_from_flattened(initial_state)
        base_env.sim.forward()
    
    def _get_sim(self):
        """Get the MuJoCo sim object from the environment."""
        if hasattr(self.env, 'sim'):
            return self.env.sim
        elif hasattr(self.env, 'env') and hasattr(self.env.env, 'sim'):
            return self.env.env.sim
        elif hasattr(self.env, 'base_env') and hasattr(self.env.base_env, 'sim'):
            return self.env.base_env.sim
        return None
    
    @property
    def camera_names(self) -> List[str]:
        """Get list of all camera names (original + added)."""
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'camera_names'):
            return list(self.env.env.camera_names)
        elif hasattr(self.env, 'base_env') and hasattr(self.env.base_env, 'camera_names'):
            return list(self.env.base_env.camera_names)
        
        # Fallback: return expected names based on config
        names = []
        if self.config.keep_original_agentview:
            names.append("agentview")
        if self.config.keep_wrist_camera:
            names.append("robot0_eye_in_hand")
        names.extend(self.added_camera_names)
        return names
    
    def get_camera_info(
        self,
        camera_height: Optional[int] = None,
        camera_width: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Get intrinsic and extrinsic information for all cameras.
        
        Args:
            camera_height: Image height (uses config default if None)
            camera_width: Image width (uses config default if None)
            
        Returns:
            Dictionary with camera intrinsics and extrinsics for each camera
        """
        cfg = self.config.third_view_config
        camera_height = camera_height or cfg.height
        camera_width = camera_width or cfg.width
        
        info = {}
        sim = self._get_sim()
        
        if sim is None:
            logger.warning("Cannot get sim, returning empty camera info")
            return info
        
        for cam_name in self.camera_names:
            try:
                K = get_camera_intrinsic_matrix(
                    sim, cam_name, camera_height, camera_width
                )
                R = get_camera_extrinsic_matrix(sim, cam_name)
                
                info[cam_name] = {
                    "intrinsics": K.tolist(),
                    "extrinsics": R.tolist(),
                }
                
                # Add sampling parameters for custom cameras
                if cam_name in self._sampled_params:
                    info[cam_name]["sampling_params"] = self._sampled_params[cam_name]
                    
            except Exception as e:
                logger.warning(f"Failed to get camera info for {cam_name}: {e}")
        
        return info
    
    def render_all_views(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Render images from all cameras.
        
        Args:
            height: Image height (uses config default if None)
            width: Image width (uses config default if None)
            
        Returns:
            Dictionary mapping camera name to RGB image array (H, W, 3)
        """
        cfg = self.config.third_view_config
        height = height or cfg.height
        width = width or cfg.width
        
        images = {}
        for cam_name in self.camera_names:
            try:
                img = self.env.render(
                    mode="rgb_array",
                    camera_name=cam_name,
                    height=height,
                    width=width,
                )
                images[cam_name] = img
            except Exception as e:
                logger.warning(f"Failed to render from camera {cam_name}: {e}")
                images[cam_name] = np.zeros((height, width, 3), dtype=np.uint8)
        
        return images
    
    def get_sampled_camera_params(self) -> Dict[str, Dict]:
        """Get the sampling parameters for all added cameras."""
        return self._sampled_params.copy()
    
    def generate_preview(
        self,
        output_path: str,
        height: int = 256,
        width: int = 256,
    ) -> None:
        """
        Generate a preview collage of all camera views for human validation.
        
        Creates an image showing all camera views in a grid, allowing manual
        verification that views are suitable for learning (no severe occlusions).
        
        Args:
            output_path: Path to save preview image (e.g., "camera_preview.png")
            height: Render height per camera
            width: Render width per camera
        """
        try:
            import cv2
            
            images = self.render_all_views(height, width)
            camera_names = list(images.keys())
            
            if not camera_names:
                logger.warning("No cameras to preview")
                return
            
            # Calculate grid layout
            n_cameras = len(camera_names)
            n_cols = min(4, n_cameras)
            n_rows = (n_cameras + n_cols - 1) // n_cols
            
            # Create composite image
            composite = np.zeros((n_rows * height, n_cols * width, 3), dtype=np.uint8)
            
            for i, (cam_name, img) in enumerate(images.items()):
                row = i // n_cols
                col = i % n_cols
                
                # Add camera name label
                img_with_label = img.copy()
                cv2.putText(
                    img_with_label,
                    cam_name,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                composite[row * height:(row + 1) * height, 
                         col * width:(col + 1) * width] = img_with_label
            
            cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            logger.info(f"Preview saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate preview: {e}")
    
    def print_config_summary(self) -> None:
        """Print a summary of the camera configuration."""
        from demo_aug.configs.multi_view_config import print_config_summary
        print_config_summary(self.config)
    
    # ========== Delegate methods to wrapped environment ==========
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def reset_to(self, state, **kwargs):
        """Reset to a specific state."""
        return self.env.reset_to(state, **kwargs)
    
    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
    
    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render(**kwargs)
    
    def get_state(self):
        """Get current state."""
        return self.env.get_state()
    
    def is_success(self):
        """Check if task is successful."""
        return self.env.is_success()
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
