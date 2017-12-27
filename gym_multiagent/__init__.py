import logging
from gym.envs.registration import registry, register, make, spec

logger = logging.getLogger(__name__)

register(
    id='MoveFormMA-v0',
    entry_point='gym_multiagent.envs:MoveFormFixEnv',
)

register(
    id='MoveFormMA-v1',
    entry_point='gym_multiagent.envs:MoveFormSinEnv',
)
