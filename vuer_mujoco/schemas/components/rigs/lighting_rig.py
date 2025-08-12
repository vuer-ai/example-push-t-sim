from vuer_mujoco.schemas.robots.light import make_light
from vuer_mujoco.schemas.se3 import se3_mujoco as m


def make_lighting_rig(pos=m.null_vec):
    """
    Set up a lighting rig with key, fill, and backlights for the simulation stage.
    The lights are transformed to fit MuJoCo's z-up coordinate system.

    :param stage: The target stage where lights should be applied.
    :return: An object containing configured lights for the rig.
    """

    # note: lights are additive, so need to keep everything normalized.
    #   this is especially true for directional light.

    class rig:
        key = make_light(
            "key",
            pos=pos + m.t2m_vec(1, 0.5, -0.5) | m.to_vec,
            dir=m.t2m_vec(-1, -0.5, 0.5) | m.to_vec,
            cutoff="45",  # Focused spotlight
            diffuse="0.3 0.3 0.3",  # Strongest light,
            directional="true",
        )
        fill = make_light(
            "fill",
            pos=pos + m.t2m_vec(0.8, 0.4, 0.7) | m.to_vec,
            dir=m.t2m_vec(-0.8, -0.6, -0.5) | m.to_vec,
            cutoff="60",  # Wider beam
            diffuse="0.18 0.18 0.18",  # Softer than key,
            directional="true",
        )
        back = make_light(
            "back",
            pos=pos + m.t2m_vec(-0.8, 1.2, 0) | m.to_vec,
            dir=m.t2m_vec(-1, 1, 0) | m.to_vec,
            cutoff="50",  # Rim light for separation
            diffuse="0.5 0.5 0.5",  # Medium intensity,
            directional="true",
        )

    return rig
