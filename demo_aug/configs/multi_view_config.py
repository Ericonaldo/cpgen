"""
Multi-view camera configuration for cross-view generalization data collection.

This module provides configuration classes for setting up multiple camera views
during demonstration replay, enabling collection of training data from various
viewpoints.

Key Design Decisions for Avoiding Occlusions:
1. Azimuth Range: Limited to frontal views (avoid robot's back to prevent body occlusion)
2. Elevation Range: Camera stays above table level (prevent object-to-object occlusion)
3. Distance Range: Not too close (prevent gripper blocking view), not too far (objects too small)

The default ranges are calibrated based on the existing 'agentview' camera, which is known
to provide good visibility for learning tasks.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Recommended Sampling Ranges
# Based on agentview camera analysis: pos="0.5 0 1.35" (roughly azimuth=0, elevation=45°, distance=1.4m)
# =============================================================================

SAMPLING_PRESETS: Dict[str, Dict] = {
    # Default: good balance between diversity and avoiding occlusions
    "default": {
        "azimuth_range": (-45.0, 45.0),      # Front-facing only (avoid robot back)
        "elevation_range": (25.0, 55.0),     # Above table (avoid object occlusion)
        "distance_range": (1.0, 1.6),        # Similar to agentview
        "target": (0.0, 0.0, 0.8),           # Table center
    },
}

# =============================================================================
# Multi-Zone Sampling Regions
# Divides the safe viewing hemisphere into distinct zones for better diversity
# Each zone is designed to avoid major occlusions while maximizing view variety
# =============================================================================

SAMPLING_ZONES: Dict[str, Dict] = {
    # Front zone: directly in front of workspace (like agentview)
    "front": {
        "azimuth_range": (-20.0, 20.0),
        "elevation_range": (35.0, 50.0),
        "distance_range": (1.2, 1.5),
        "weight": 2,  # Higher weight = more cameras allocated here
    },
    # Front-left zone: viewing from left-front angle
    "front_left": {
        "azimuth_range": (-55.0, -25.0),
        "elevation_range": (30.0, 50.0),
        "distance_range": (1.0, 1.4),
        "weight": 1,
    },
    # Front-right zone: viewing from right-front angle
    "front_right": {
        "azimuth_range": (25.0, 55.0),
        "elevation_range": (30.0, 50.0),
        "distance_range": (1.0, 1.4),
        "weight": 1,
    },
    # Overhead zone: bird's eye view (high elevation)
    "overhead": {
        "azimuth_range": (-30.0, 30.0),
        "elevation_range": (60.0, 80.0),
        "distance_range": (1.0, 1.3),
        "weight": 1,
    },
}

# Default target point for all zones (table center)
DEFAULT_TARGET: Tuple[float, float, float] = (0.0, 0.0, 0.8)


@dataclass
class ThirdViewCameraConfig:
    """Configuration for a single third-person view camera with sampling ranges.
    
    The camera position is defined using spherical coordinates centered on a target point:
    - azimuth: horizontal angle from front (0 degrees = viewing from +Y direction)
    - elevation: angle above the horizontal plane
    - distance: distance from the target point
    
    IMPORTANT Constraints to Avoid Occlusions:
    - azimuth_range: Keep between -90° and 90° to stay in front of the robot
                     Going behind the robot (>90° or <-90°) causes body occlusion
    - elevation_range: Minimum ~20° to stay above table objects
                       Too low causes object-to-object occlusion
    - distance_range: Minimum ~0.8m to avoid gripper blocking the view
                      Maximum ~2.5m to keep objects visible and large enough
    
    Attributes:
        name: Base name for the camera (will be suffixed with index)
        target: 3D point the camera looks at (workspace center)
        azimuth_range: (min, max) horizontal angle in degrees, 0 = front
        elevation_range: (min, max) angle above horizontal in degrees
        distance_range: (min, max) distance from target in meters
        width: Image width in pixels
        height: Image height in pixels
        fov: Field of view in degrees
        is_random: If False, use fixed evenly-spaced poses within ranges
    """
    
    name: str = "third_view"
    
    # Target point (where camera looks at) - typically workspace center
    # For table tasks, this is usually around (0, 0, 0.8) which is table surface level
    target: Tuple[float, float, float] = (0.0, 0.0, 0.8)
    
    # Spherical coordinate sampling ranges (around target)
    # These defaults are calibrated to avoid common occlusion issues:
    azimuth_range: Tuple[float, float] = (-45.0, 45.0)    # Front-facing only
    elevation_range: Tuple[float, float] = (25.0, 55.0)   # Above table, not too steep
    distance_range: Tuple[float, float] = (1.0, 1.6)      # Good viewing distance
    
    # Camera intrinsics (fixed for consistency)
    width: int = 256
    height: int = 256
    fov: float = 60.0  # degrees
    
    # Sampling behavior
    is_random: bool = True  # If False, use fixed evenly-spaced poses within ranges
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "ThirdViewCameraConfig":
        """Create config from a predefined preset.
        
        Args:
            preset_name: One of 'conservative', 'default', 'diverse', 'side_left', 'side_right'
            **overrides: Override any preset values
            
        Returns:
            ThirdViewCameraConfig with preset values
            
        Example:
            >>> config = ThirdViewCameraConfig.from_preset('conservative', width=512)
        """
        if preset_name not in SAMPLING_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(SAMPLING_PRESETS.keys())}"
            )
        preset = SAMPLING_PRESETS[preset_name].copy()
        preset.update(overrides)
        return cls(**preset)


@dataclass
class EyeInHandCameraConfig:
    """Configuration for eye-in-hand cameras around the gripper.
    
    These cameras are attached to the robot's end-effector (robot0_right_hand)
    and move with the robot. They provide close-up views of the manipulation
    area from different angles relative to the gripper.
    
    The camera positions are defined using spherical coordinates centered on 
    a point relative to the gripper (typically looking forward/down).
    
    Attributes:
        name: Base name for the cameras (will be suffixed with index)
        target: 3D offset from gripper origin that cameras look at (in gripper frame)
        azimuth_range: (min, max) horizontal angle in degrees around Z-axis
        elevation_range: (min, max) angle from horizontal in degrees
        distance_range: (min, max) distance from target in meters
        width: Image width in pixels
        height: Image height in pixels
        fov: Field of view in degrees
        is_random: If False, use fixed evenly-spaced poses
    """
    
    name: str = "wrist_view"
    
    # Target point relative to gripper origin (in gripper frame)
    # Default: 25cm in front of gripper, looking at the workspace
    target: Tuple[float, float, float] = (0.0, 0.0, 0.25)
    
    # Spherical coordinate sampling ranges (relative to gripper)
    # These are much smaller than third-person views since we're close to the gripper
    azimuth_range: Tuple[float, float] = (-30.0, 30.0)     # Left/right variation
    elevation_range: Tuple[float, float] = (-20.0, 20.0)   # Up/down variation
    distance_range: Tuple[float, float] = (0.10, 0.15)     # Close to gripper (10-15cm)
    
    # Camera intrinsics
    width: int = 256
    height: int = 256
    fov: float = 75.0  # Wider FOV for close-up views
    
    # Sampling behavior
    is_random: bool = True



@dataclass
class MultiViewCameraConfig:
    """Configuration for multi-view data collection.
    
    This configuration controls how many additional camera views to generate
    and how they are positioned around the workspace.
    
    Supports two types of additional cameras:
    1. Third-person views: Fixed cameras positioned around the workspace
    2. Wrist views: Cameras attached to the gripper that move with the robot
    
    Sampling Modes:
    - use_zone_sampling=False: All cameras sample from third_view_config ranges
    - use_zone_sampling=True: Cameras distributed across SAMPLING_ZONES for diversity
    
    Attributes:
        num_third_views: Number of additional third-person views to generate
        num_wrist_views: Number of additional wrist-mounted views to generate
        use_zone_sampling: If True, distribute cameras across predefined zones
        keep_original_agentview: Whether to keep the original agentview camera
        keep_wrist_camera: Whether to keep the robot0_eye_in_hand camera  
        third_view_config: Configuration template for third-person views
        wrist_view_config: Configuration template for wrist views
        min_angular_separation: Minimum angle between sampled views in degrees
        seed: Random seed for reproducible camera sampling
    
    Example:
        >>> # Multi-zone sampling for diverse views
        >>> config = MultiViewCameraConfig(
        ...     num_third_views=5,
        ...     use_zone_sampling=True,  # Enable zone-based distribution
        ...     seed=42,
        ... )
    """
    
    # Number of additional third-person views to generate
    num_third_views: int = 4
    
    # Number of additional wrist-mounted views to generate
    num_wrist_views: int = 0
    
    # Enable zone-based sampling for better view diversity
    # When True, cameras are distributed across SAMPLING_ZONES (front, front_left, front_right, overhead)
    use_zone_sampling: bool = False
    
    # Whether to keep the original agentview (recommended: True)
    keep_original_agentview: bool = True
    
    # Whether to keep wrist camera (useful for TCP regression)
    keep_wrist_camera: bool = True
    
    # Third-person view config (used when use_zone_sampling=False)
    third_view_config: ThirdViewCameraConfig = field(
        default_factory=ThirdViewCameraConfig
    )
    
    # Wrist view config (template for eye-in-hand views)
    wrist_view_config: EyeInHandCameraConfig = field(
        default_factory=EyeInHandCameraConfig
    )
    
    # Minimum angular separation between views (degrees)
    # Higher values = more diverse views but harder to sample
    min_angular_separation: float = 20.0
    
    # Seed for reproducibility (None = random each time)
    seed: Optional[int] = None
    
    def get_all_camera_names(self) -> List[str]:
        """Get list of all camera names that will be used.
        
        Returns:
            List of camera name strings
        """
        names = []
        if self.keep_original_agentview:
            names.append("agentview")
        if self.keep_wrist_camera:
            names.append("robot0_eye_in_hand")
        for i in range(self.num_third_views):
            names.append(f"{self.third_view_config.name}_{i}")
        for i in range(self.num_wrist_views):
            names.append(f"{self.wrist_view_config.name}_{i}")
        return names
    

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        num_third_views: int = 4,
        seed: Optional[int] = None,
        **overrides,
    ) -> "MultiViewCameraConfig":
        """Create config from a predefined camera preset.
        
        Args:
            preset_name: One of 'conservative', 'default', 'diverse'
            num_third_views: Number of additional views
            seed: Random seed
            **overrides: Override any config values
            
        Returns:
            MultiViewCameraConfig with preset values
            
        Example:
            >>> config = MultiViewCameraConfig.from_preset('diverse', num_third_views=6, seed=42)
        """
        third_view_config = ThirdViewCameraConfig.from_preset(preset_name)
        return cls(
            num_third_views=num_third_views,
            third_view_config=third_view_config,
            seed=seed,
            **overrides,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_camera_ranges(config: ThirdViewCameraConfig) -> List[str]:
    """Validate camera sampling ranges and return warnings.
    
    Args:
        config: Camera configuration to validate
        
    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []
    
    # Check azimuth (staying frontal)
    if config.azimuth_range[0] < -90 or config.azimuth_range[1] > 90:
        warnings.append(
            f"[WARNING] Azimuth range {config.azimuth_range} extends beyond +/-90 deg. "
            "This may cause robot body to occlude the workspace."
        )
    
    # Check elevation (staying above table)
    if config.elevation_range[0] < 15:
        warnings.append(
            f"[WARNING] Minimum elevation {config.elevation_range[0]} deg is very low. "
            "This may cause objects on table to occlude each other."
        )
    if config.elevation_range[1] > 80:
        warnings.append(
            f"[WARNING] Maximum elevation {config.elevation_range[1]} deg is very high. "
            "Top-down views may make depth perception difficult."
        )
    
    # Check distance (not too close/far)
    if config.distance_range[0] < 0.6:
        warnings.append(
            f"[WARNING] Minimum distance {config.distance_range[0]}m is very close. "
            "Gripper may frequently occlude the target."
        )
    if config.distance_range[1] > 2.5:
        warnings.append(
            f"[WARNING] Maximum distance {config.distance_range[1]}m is quite far. "
            "Objects may appear very small in the image."
        )
    
    return warnings


def print_config_summary(config: MultiViewCameraConfig) -> None:
    """Print a human-readable summary of the configuration."""
    cfg = config.third_view_config
    
    print("=" * 60)
    print("Multi-View Camera Configuration Summary")
    print("=" * 60)
    print(f"Total cameras: {len(config.get_all_camera_names())}")
    print(f"  - Original agentview: {'Yes' if config.keep_original_agentview else 'No'}")
    print(f"  - Wrist camera: {'Yes' if config.keep_wrist_camera else 'No'}")
    print(f"  - Added third views: {config.num_third_views}")
    print()
    print(f"Third View Sampling Ranges:")
    print(f"  - Azimuth:   {cfg.azimuth_range[0]:.0f}° to {cfg.azimuth_range[1]:.0f}°")
    print(f"  - Elevation: {cfg.elevation_range[0]:.0f}° to {cfg.elevation_range[1]:.0f}°")
    print(f"  - Distance:  {cfg.distance_range[0]:.1f}m to {cfg.distance_range[1]:.1f}m")
    print(f"  - Target:    ({cfg.target[0]:.2f}, {cfg.target[1]:.2f}, {cfg.target[2]:.2f})")
    print()
    print(f"Image size: {cfg.width}x{cfg.height}, FOV: {cfg.fov}°")
    print(f"Random sampling: {cfg.is_random}, Min separation: {config.min_angular_separation}°")
    print(f"Seed: {config.seed or 'None (random)'}")
    
    # Validate and print warnings
    warnings = validate_camera_ranges(cfg)
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
    else:
        print()
        print("[OK] Configuration looks good for avoiding occlusions")
    
    print("=" * 60)


# =============================================================================
# CLI Argument Helpers
# =============================================================================

def add_multi_view_args(parser: argparse.ArgumentParser) -> None:
    """Add multi-view camera arguments to an argument parser.
    
    This centralizes all multi-view CLI argument definitions to avoid
    duplication across scripts.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group('Multi-view camera options')
    group.add_argument(
        "--enable_multi_view",
        action="store_true",
        help="Enable multi-view data collection with additional cameras",
    )
    group.add_argument(
        "--num_third_views",
        type=int,
        default=4,
        help="Number of additional third-person view cameras (default: 4)",
    )
    group.add_argument(
        "--multi_view_seed",
        type=int,
        default=None,
        help="Seed for reproducible camera pose sampling",
    )
    group.add_argument(
        "--multi_view_target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.8],
        help="Target point for cameras to look at (x, y, z)",
    )
    group.add_argument(
        "--multi_view_azimuth_range",
        type=float,
        nargs=2,
        default=[-45.0, 45.0],
        help="Azimuth angle range in degrees (min, max)",
    )
    group.add_argument(
        "--multi_view_elevation_range",
        type=float,
        nargs=2,
        default=[25.0, 55.0],
        help="Elevation angle range in degrees (min, max)",
    )
    group.add_argument(
        "--multi_view_distance_range",
        type=float,
        nargs=2,
        default=[1.0, 1.6],
        help="Distance range from target in meters (min, max)",
    )
    group.add_argument(
        "--multi_view_min_separation",
        type=float,
        default=20.0,
        help="Minimum angular separation between views in degrees",
    )
    group.add_argument(
        "--multi_view_image_size",
        type=int,
        default=256,
        help="Image size for multi-view cameras (height and width)",
    )
    group.add_argument(
        "--multi_view_preview_path",
        type=str,
        default=None,
        help="(optional) Path to save camera preview image for human validation",
    )


def create_config_from_args(
    args,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Optional[MultiViewCameraConfig]:
    """Create MultiViewCameraConfig from parsed command-line arguments.
    
    Args:
        args: Parsed argparse namespace with multi-view arguments
        image_height: Override image height (uses args.multi_view_image_size if None)
        image_width: Override image width (uses args.multi_view_image_size if None)
        
    Returns:
        MultiViewCameraConfig if multi-view enabled, None otherwise
    """
    if not getattr(args, 'enable_multi_view', False):
        return None
    
    # Use provided sizes or fall back to multi_view_image_size
    height = image_height or getattr(args, 'multi_view_image_size', 256)
    width = image_width or getattr(args, 'multi_view_image_size', 256)
    
    return MultiViewCameraConfig(
        num_third_views=args.num_third_views,
        keep_original_agentview=True,
        keep_wrist_camera=True,
        third_view_config=ThirdViewCameraConfig(
            target=tuple(args.multi_view_target),
            azimuth_range=tuple(args.multi_view_azimuth_range),
            elevation_range=tuple(args.multi_view_elevation_range),
            distance_range=tuple(args.multi_view_distance_range),
            width=width,
            height=height,
        ),
        min_angular_separation=args.multi_view_min_separation,
        seed=args.multi_view_seed,
    )

