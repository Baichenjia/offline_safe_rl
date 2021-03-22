from env.ant_circle import AntCircleEnv
from env.humanoid_circle import HumanoidCircleEnv

from gym.envs.registration import register

register(
    id='AntCircle-v0',
    entry_point='env:AntCircleEnv',
)

register(
    id='HumanoidCircle-v0',
    entry_point='env:HumanoidCircleEnv',
)