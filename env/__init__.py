from env.ant_truncated import AntTruncatedObsEnv
from env.ant_circle import AntCircleEnv
from env.humanoid_circle import HumanoidCircleEnv
from env.ant_circle_truncated import AntCircleTruncatedEnv
from env.humanoid_circle_truncated import HumanoidCircleTruncatedEnv
from env.pendulum_safety import PendulumSafetyEnv
from env.pointmass import PointMass
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