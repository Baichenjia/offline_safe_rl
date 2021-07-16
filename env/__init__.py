from env.ant_truncated import AntTruncatedObsEnv
from env.humanoid_truncated import HumanoidTruncatedObsEnv
from env.ant_circle import AntCircleEnv
from env.humanoid_circle import HumanoidCircleEnv
from env.ant_circle_truncated import AntCircleTruncatedEnv
from env.humanoid_circle_truncated import HumanoidCircleTruncatedEnv
from env.pendulum_safety import PendulumSafetyEnv
from env.hopper_nt import HopperNT
from env.walker2d_nt import Walker2dNT
from env.mbrl_walker import MBRLWalker
from env.reacher_safety import ReacherSafetyEnv
from env.reacher_speed import ReacherSpeedEnv
from env.pointmass import PointMass
from env.bottleneck import create_bottleneck_env
from gym.envs.registration import register

register(
    id='PointMass-v0',
    entry_point='env:PointMass'
)

register(
    id='PendulumSafe-v0',
    entry_point='env:PendulumSafetyEnv'
)

register(
    id='AntTruncated-v0',
    entry_point='env:AntTruncatedObsEnv',
)

register(
    id='HumanoidTruncated-v0',
    entry_point='env:HumanoidTruncatedObsEnv',
)

register(
    id='AntCircle-v0',
    entry_point='env:AntCircleEnv',
)

register(
    id='HumanoidCircle-v0',
    entry_point='env:HumanoidCircleEnv',
)

register(
    id='AntCircleTruncated-v0',
    entry_point='env:AntCircleTruncatedEnv',
)

register(
    id='HumanoidCircleTruncated-v0',
    entry_point='env:HumanoidCircleTruncatedEnv',
)

# register(
#     id='HopperNT-v3',
#     entry_point='env:HopperNT',
#     )

# register(
#     id='Walker2dNT-v3',
#     entry_point='env:Walker2dNT',
#     )

register(
    id='MBRLHopper-v3',
    entry_point='env:MBRLWalker',
    kwargs={
        'name': 'Hopper-v3',
        'ctrl_coeff': 0.1,
        'height_coeff': 3,
        'alive_bonus': 1,
        'target_height': 1.3,
        'velocity_idx': 5,
        'height_idx': 0,
        }
    )

register(
    id='MBRLWalker2d-v3',
    entry_point='env:MBRLWalker',
    kwargs={
        'name': 'Walker2d-v3',
        'ctrl_coeff': 0.1,
        'height_coeff': 3,
        'alive_bonus': 1,
        'target_height': 1.3,
        'velocity_idx': 8,
        'height_idx': 0,
        }
    )

register(
    id='MBRLAnt-v3',
    entry_point='env:MBRLWalker',
    kwargs={
        'name': 'Ant-v3',
        'ctrl_coeff': 0.1,
        'height_coeff': 3,
        'alive_bonus': 1,
        'target_height': 0.57,
        'velocity_idx': 13,
        'height_idx': 0,
        }
    )

register(
    id='MBRLSwimmer-v3',
    entry_point='env:MBRLWalker',
    kwargs={
        'name': 'Swimmer-v3',
        'ctrl_coeff': 0.0001,
        'height_coeff': 0,
        'alive_bonus': 0,
        'target_height': -1,
        'velocity_idx': 3,
        'height_idx': 0,
        }
    )

register(
    id='ReacherSafety-v0',
    entry_point='env:ReacherSafetyEnv'
    )

register(
    id='ReacherSpeed-v0',
    entry_point='env:ReacherSpeedEnv'
    )

register(
    id='Bottleneck-v0',
    entry_point='env:create_bottleneck_env'
    )