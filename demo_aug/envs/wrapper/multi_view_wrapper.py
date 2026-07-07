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
    WristCameraPerturbConfig,
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
        self._modified_xml: Optional[str] = None  # Store XML with added cameras
        self._original_xml: Optional[str] = None  # Store XML before any camera additions
        self._original_state: Optional[np.ndarray] = None  # Store initial sim state

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

        # Store original XML and state for resampling
        if self._original_xml is None:
            self._original_xml = xml
            self._original_state = initial_state.copy()
        
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
                    if cfg.fixed_positions is not None and i < len(cfg.fixed_positions):
                        # Use explicitly specified fixed positions
                        azimuth, elevation, distance = cfg.fixed_positions[i]
                    elif cfg.is_random:
                        azimuth = rng.uniform(*cfg.azimuth_range)
                        elevation = rng.uniform(*cfg.elevation_range)
                        distance = rng.uniform(*cfg.distance_range)
                    else:
                        # Evenly spaced azimuths for deterministic placement
                        azimuth_span = cfg.azimuth_range[1] - cfg.azimuth_range[0]
                        azimuth = cfg.azimuth_range[0] + (i + 0.5) * azimuth_span / self.config.num_third_views
                        elevation = (cfg.elevation_range[0] + cfg.elevation_range[1]) / 2
                        distance = (cfg.distance_range[0] + cfg.distance_range[1]) / 2

                    # Check angular separation (circular distance for wraparound)
                    is_valid = all(
                        min(abs(azimuth - prev_az), 360 - abs(azimuth - prev_az))
                        >= self.config.min_angular_separation
                        for prev_az in sampled_azimuths
                    )

                    if is_valid or not cfg.is_random or cfg.fixed_positions is not None:
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
        
        # ----- Apply perturbation to original wrist camera -----
        if self.config.wrist_perturbation.enable and self.config.keep_wrist_camera:
            xml = self._apply_wrist_perturbation(xml, rng)

        # ----- Add perturbed wrist camera copies -----
        if self.config.num_perturbed_wrist_views > 0:
            xml = self._add_perturbed_wrist_cameras(xml, rng)

        # Reload environment with new XML
        self._reload_env_with_xml(xml, initial_state)
        self._cameras_added = True
        
        logger.info(f"Successfully added {len(self.added_camera_names)} cameras to environment")

    def resample_cameras(self, seed: int):
        """Resample camera positions with a new random seed.

        This allows generating different camera views for each trajectory in a dataset.
        Uses the stored original XML (before any camera additions) as a clean base.

        Args:
            seed: Random seed for reproducible but different camera sampling.
        """
        if self._original_xml is None:
            raise RuntimeError("Cannot resample: original XML not stored. "
                               "Call _add_cameras_to_env() first.")

        # Reset state for re-adding cameras
        self.added_camera_names = []
        self._sampled_params = {}
        self._cameras_added = False
        self._modified_xml = None

        # Restore original XML in the sim so _add_cameras_to_env reads clean XML
        sim = self._get_sim()
        if sim is not None:
            # We need to reload from original XML first
            base_env = self._get_base_env()
            base_env.reset_from_xml_string(self._original_xml)
            base_env.sim.reset()
            base_env.sim.set_state_from_flattened(self._original_state)
            base_env.sim.forward()

        # Update seed and re-add cameras
        self.config.seed = seed
        self._add_cameras_to_env()

        logger.info(f"Resampled cameras with seed={seed}: {self.added_camera_names}")

    def _get_base_env(self):
        """Get the base robosuite environment object."""
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'reset_from_xml_string'):
            return self.env.env
        elif hasattr(self.env, 'base_env') and hasattr(self.env.base_env, 'reset_from_xml_string'):
            return self.env.base_env
        elif hasattr(self.env, 'reset_from_xml_string'):
            return self.env
        else:
            raise RuntimeError("Cannot find reset_from_xml_string method on environment")

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
    
    def _apply_wrist_perturbation(
        self, 
        xml: str, 
        rng: np.random.Generator
    ) -> str:
        """Apply random perturbation to the original robot0_eye_in_hand camera.
        
        This modifies the camera's position and orientation in the XML to simulate
        real-world mounting variations.
        
        Args:
            xml: MuJoCo XML string
            rng: Random number generator for reproducibility
            
        Returns:
            Modified XML string with perturbed wrist camera
        """
        import re
        
        cfg = self.config.wrist_perturbation
        
        # Sample perturbation values
        pos_delta = np.array([
            rng.uniform(*cfg.pos_x_range),
            rng.uniform(*cfg.pos_y_range),
            rng.uniform(*cfg.pos_z_range),
        ])
        
        roll_delta = rng.uniform(*cfg.roll_range)
        pitch_delta = rng.uniform(*cfg.pitch_range)
        yaw_delta = rng.uniform(*cfg.yaw_range)
        
        # Store perturbation params for logging/debugging
        self._sampled_params["robot0_eye_in_hand_perturbation"] = {
            "pos_delta": pos_delta.tolist(),
            "roll_delta": roll_delta,
            "pitch_delta": pitch_delta,
            "yaw_delta": yaw_delta,
        }
        
        logger.info(
            f"Applying wrist camera perturbation: "
            f"pos_delta={pos_delta}, roll={roll_delta:.2f}°, "
            f"pitch={pitch_delta:.2f}°, yaw={yaw_delta:.2f}°"
        )
        
        # Find robot0_eye_in_hand camera in XML
        # Pattern matches: <camera name="robot0_eye_in_hand" pos="..." quat="..." ... />
        camera_pattern = r'(<camera[^>]*name="robot0_eye_in_hand"[^>]*)(/>)'
        
        match = re.search(camera_pattern, xml)
        if not match:
            logger.warning("Could not find robot0_eye_in_hand camera in XML, skipping perturbation")
            return xml
        
        camera_tag = match.group(1)
        
        # Extract current pos attribute
        pos_match = re.search(r'pos="([^"]+)"', camera_tag)
        if pos_match:
            current_pos = np.array([float(x) for x in pos_match.group(1).split()])
            new_pos = current_pos + pos_delta
            new_pos_str = f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}"
            camera_tag = re.sub(r'pos="[^"]+"', f'pos="{new_pos_str}"', camera_tag)
        
        # Extract current quat attribute and apply rotation perturbation
        quat_match = re.search(r'quat="([^"]+)"', camera_tag)
        if quat_match:
            # MuJoCo uses wxyz quaternion format
            current_quat_wxyz = np.array([float(x) for x in quat_match.group(1).split()])
            current_quat_xyzw = np.array([
                current_quat_wxyz[1], current_quat_wxyz[2], 
                current_quat_wxyz[3], current_quat_wxyz[0]
            ])
            
            # Create perturbation rotation (Euler angles in degrees)
            perturb_rot = Rotation.from_euler(
                'xyz', 
                [roll_delta, pitch_delta, yaw_delta], 
                degrees=True
            )
            
            # Apply perturbation: new_rot = perturb_rot * current_rot
            current_rot = Rotation.from_quat(current_quat_xyzw)
            new_rot = perturb_rot * current_rot
            
            # Convert back to wxyz for MuJoCo
            new_quat_xyzw = new_rot.as_quat()
            new_quat_wxyz = np.array([
                new_quat_xyzw[3], new_quat_xyzw[0], 
                new_quat_xyzw[1], new_quat_xyzw[2]
            ])
            new_quat_str = f"{new_quat_wxyz[0]:.6f} {new_quat_wxyz[1]:.6f} {new_quat_wxyz[2]:.6f} {new_quat_wxyz[3]:.6f}"
            camera_tag = re.sub(r'quat="[^"]+"', f'quat="{new_quat_str}"', camera_tag)
        
        # Replace in XML
        xml = re.sub(camera_pattern, camera_tag + r'\2', xml)

        return xml

    def _add_perturbed_wrist_cameras(self, xml: str, rng: np.random.Generator) -> str:
        """Add N perturbed copies of robot0_eye_in_hand as separate cameras.

        Each perturbed camera is an independent perturbation of the original wrist
        camera, attached to the same parent body (robot0_right_hand). These are
        intended to be paired 1:1 with third-person views for cross-view experiments.

        Args:
            xml: MuJoCo XML string
            rng: Random number generator for reproducibility

        Returns:
            Modified XML string with added perturbed wrist cameras
        """
        import re

        cfg = self.config.wrist_perturbation

        # Extract original robot0_eye_in_hand pos/quat from XML
        camera_pattern = r'<camera[^>]*name="robot0_eye_in_hand"[^>]*/>'
        match = re.search(camera_pattern, xml)
        if not match:
            logger.warning("robot0_eye_in_hand not found, skipping perturbed wrist cameras")
            return xml

        camera_tag = match.group(0)
        pos_match = re.search(r'pos="([^"]+)"', camera_tag)
        quat_match = re.search(r'quat="([^"]+)"', camera_tag)
        fovy_match = re.search(r'fovy="([^"]+)"', camera_tag)

        if not pos_match or not quat_match:
            logger.warning("Cannot extract pos/quat from robot0_eye_in_hand")
            return xml

        original_pos = np.array([float(x) for x in pos_match.group(1).split()])
        original_quat_wxyz = np.array([float(x) for x in quat_match.group(1).split()])
        original_fovy = None
        if fovy_match:
            original_fovy = fovy_match.group(1)
        else:
            # Fallback to configured wrist FOV if the original camera has no explicit fovy.
            original_fovy = str(self.config.wrist_view_config.fov)

        for i in range(self.config.num_perturbed_wrist_views):
            name = f"robot0_eye_in_hand_perturbed_{i}"

            # Sample perturbation (same logic as _apply_wrist_perturbation)
            pos_delta = np.array([
                rng.uniform(*cfg.pos_x_range),
                rng.uniform(*cfg.pos_y_range),
                rng.uniform(*cfg.pos_z_range),
            ])
            roll_delta = rng.uniform(*cfg.roll_range)
            pitch_delta = rng.uniform(*cfg.pitch_range)
            yaw_delta = rng.uniform(*cfg.yaw_range)

            # Apply position perturbation
            new_pos = original_pos + pos_delta

            # Apply rotation perturbation
            original_quat_xyzw = np.array([
                original_quat_wxyz[1], original_quat_wxyz[2],
                original_quat_wxyz[3], original_quat_wxyz[0],
            ])
            perturb_rot = Rotation.from_euler(
                'xyz', [roll_delta, pitch_delta, yaw_delta], degrees=True
            )
            current_rot = Rotation.from_quat(original_quat_xyzw)
            new_rot = perturb_rot * current_rot
            new_quat_xyzw = new_rot.as_quat()
            new_quat_wxyz = np.array([
                new_quat_xyzw[3], new_quat_xyzw[0],
                new_quat_xyzw[1], new_quat_xyzw[2],
            ])

            # Store params
            self._sampled_params[name] = {
                "pos_delta": pos_delta.tolist(),
                "roll_delta": roll_delta,
                "pitch_delta": pitch_delta,
                "yaw_delta": yaw_delta,
                "is_wrist": True,
            }

            # Add camera to XML
            pos_str = f"{new_pos[0]} {new_pos[1]} {new_pos[2]}"
            quat_str = f"{new_quat_wxyz[0]} {new_quat_wxyz[1]} {new_quat_wxyz[2]} {new_quat_wxyz[3]}"

            xml = add_camera_to_xml(
                xml=xml,
                camera_name=name,
                camera_pos=pos_str,
                camera_quat=quat_str,
                parent_body_name="robot0_right_hand",
                is_eye_in_hand_camera=True,
                fovy=original_fovy,
            )
            self.added_camera_names.append(name)
            logger.info(
                f"Added perturbed wrist camera '{name}': "
                f"pos_delta={pos_delta}, rot=({roll_delta:.1f}, {pitch_delta:.1f}, {yaw_delta:.1f})°"
            )

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
        # Save the modified XML for future resets
        self._modified_xml = xml

        # Get the base env
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'reset_from_xml_string'):
            base_env = self.env.env
        elif hasattr(self.env, 'base_env') and hasattr(self.env.base_env, 'reset_from_xml_string'):
            base_env = self.env.base_env
        elif hasattr(self.env, 'reset_from_xml_string'):
            base_env = self.env
        else:
            raise RuntimeError("Cannot find reset_from_xml_string method on environment")
        
        # Reload simulation with new XML
        # NOTE: This will reset camera_names, so we need to update them AFTER
        base_env.reset_from_xml_string(xml)

        # NOW update camera lists AFTER reload (to prevent them being overwritten)
        third_cfg = self.config.third_view_config
        wrist_cfg = self.config.wrist_view_config

        for cam_name in self.added_camera_names:
            if cam_name not in base_env.camera_names:
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

        logger.info(f"Successfully updated camera configuration with {len(self.added_camera_names)} additional cameras")

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
        print_config_summary(self.config, preset_name=self.config.preset_name)
    
    # ========== Delegate methods to wrapped environment ==========
    
    def reset(self, **kwargs):
        """Reset the environment."""
        obs = self.env.reset(**kwargs)

        # Add observations from additional cameras (same as step / reset_to).
        if self._cameras_added and self.added_camera_names and obs is not None:
            obs = self._add_camera_observations(obs)

        return obs
    
    def reset_to(self, state, **kwargs):
        """Reset to a specific state, preserving added cameras."""
        if hasattr(self.env, 'reset_to'):
            # If state contains "model" (XML), replace it with our modified XML
            # to preserve the added cameras
            if self._cameras_added and self._modified_xml is not None and isinstance(state, dict) and "model" in state:
                # Create a copy of state with our modified XML
                modified_state = state.copy()
                modified_state["model"] = self._modified_xml
                obs = self.env.reset_to(modified_state, **kwargs)
            else:
                obs = self.env.reset_to(state, **kwargs)
        else:
            # Raw robosuite env: use sim state setting directly
            if isinstance(state, dict) and "states" in state:
                state_val = state["states"]
            else:
                state_val = state
            import numpy as np
            if isinstance(state_val, np.ndarray) and hasattr(self.env, 'sim'):
                self.env.sim.set_state_from_flattened(state_val)
                self.env.sim.forward()
            if hasattr(self.env, '_get_observations'):
                obs = self.env._get_observations()
            else:
                obs = self.env.reset()

        # Add observations from additional cameras
        if self._cameras_added and self.added_camera_names and obs is not None:
            obs = self._add_camera_observations(obs)

        return obs

    def _add_camera_observations(self, obs: dict) -> dict:
        """Add observations from additional cameras to the observation dict."""
        if not isinstance(obs, dict):
            return obs

        # Render each additional camera
        for cam_name in self.added_camera_names:
            # Get camera config
            is_wrist = self._sampled_params.get(cam_name, {}).get("is_wrist", False)
            if is_wrist:
                cfg = self.config.wrist_view_config
            else:
                cfg = self.config.third_view_config

            try:
                # Render RGB and depth directly from robosuite sim
                if hasattr(self.env, 'env'):
                    base_env = self.env.env
                else:
                    base_env = self.env

                # Render RGB and depth together for efficiency
                rgb, depth = base_env.sim.render(
                    camera_name=cam_name,
                    height=cfg.height,
                    width=cfg.width,
                    depth=True,
                )

                # MuJoCo renders images upside down, so flip them
                obs[f"{cam_name}_image"] = rgb[::-1]
                depth_flipped = depth[::-1]
                # Linearize depth from MuJoCo z-buffer [0,1] to metric depth (meters)
                from demo_aug.utils.camera_utils import get_real_depth_map
                depth_flipped = get_real_depth_map(base_env.sim, depth_flipped)
                # Ensure depth has trailing channel dim (H, W, 1) to match
                # robosuite's standard observation format.
                if depth_flipped.ndim == 2:
                    depth_flipped = depth_flipped[..., None]
                obs[f"{cam_name}_depth"] = depth_flipped

            except Exception as e:
                logger.warning(f"Failed to render camera {cam_name}: {e}")

        return obs
    
    def step(self, action):
        """Take a step in the environment, adding multi-view camera observations."""
        obs, reward, done, info = self.env.step(action)

        # Add observations from additional cameras
        if self._cameras_added and self.added_camera_names:
            obs = self._add_camera_observations(obs)

        return obs, reward, done, info
    
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
