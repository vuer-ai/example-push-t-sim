from vuer_mujoco.schemas.se3.se3_mujoco import m2t_quat, t2m_quat, m2t_vec, t2m_vec, as_wxyz, to_vec
from vuer_mujoco.schemas.se3.se3_three import as_xyzw


def test_quat_transform():
    quat_three = (0.1, 0.2, 0.3, 1.0)

    quat_mujoco = t2m_quat(*quat_three)

    recovered_quat_three = m2t_quat(*quat_mujoco)

    assert recovered_quat_three == quat_three, "Quaternion transformation failed"


def test_vector_transform():
    vector_three = (1.0, 2.0, 3.0)

    vector_mujoco = t2m_vec(*vector_three)

    recovered_vector_three = m2t_vec(*vector_mujoco)

    assert recovered_vector_three == vector_three, "Vector transformation failed"


def test_transform_output():
    quat_three = (0.1, 0.2, 0.3, 1.0)

    quat_mujoco = t2m_quat(*quat_three) | as_wxyz

    assert repr(quat_mujoco) == "WXYZ(w=1.0, x=0.1, y=-0.3, z=0.2)", "Quaternion transformation failed"


def test_to_vec_and_to_quat_transform():
    vector_three = (1.0, 2.0, 3.0) | to_vec
    quat_three = (0.1, 0.2, 0.3, 1.0) | as_xyzw

    assert repr(vector_three) == "Vector3(x=1.0, y=2.0, z=3.0)", "Vector transformation failed"
    assert repr(quat_three) == "XYZW(x=0.1, y=0.2, z=0.3, w=1.0)", "Quaternion transformation failed"
