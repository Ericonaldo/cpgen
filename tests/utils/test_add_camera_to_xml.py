import re

from demo_aug.utils.mujoco_utils import add_camera_to_xml


def _minimal_mujoco_xml() -> str:
    # worldbody must be a direct child for demo_aug.utils.mujoco_utils.add_camera_to_xml
    # and the robot0_right_hand body must exist for eye-in-hand cameras.
    return """
<mujoco>
  <worldbody>
    <body name="robot0_right_hand" />
  </worldbody>
</mujoco>
""".strip()


def test_add_camera_to_xml_sets_fovy_for_worldbody_cameras():
    xml = _minimal_mujoco_xml()
    out = add_camera_to_xml(
        xml=xml,
        camera_name="third_view_0",
        camera_pos="0 0 1",
        camera_quat="1 0 0 0",
        is_eye_in_hand_camera=False,
        fovy="60",
    )
    assert 'name="third_view_0"' in out
    assert re.search(r'name="third_view_0"[^>]*fovy="60"', out) is not None


def test_add_camera_to_xml_sets_fovy_for_eye_in_hand_cameras():
    xml = _minimal_mujoco_xml()
    out = add_camera_to_xml(
        xml=xml,
        camera_name="wrist_view_0",
        camera_pos="0 0 0.1",
        camera_quat="1 0 0 0",
        parent_body_name="robot0_right_hand",
        is_eye_in_hand_camera=True,
        fovy="75",
    )
    assert 'name="wrist_view_0"' in out
    assert re.search(r'name="wrist_view_0"[^>]*fovy="75"', out) is not None

