# gym_multiagent

# Dependency

[OpenAI Gym](https://github.com/openai/gym)

# Installation

```bash
git clone https://github.com/kbys-t/gym_MA.git
cd gym_MA
pip install -e .
```

# How to use
1. First of all,
`import gym_multiagent`

1. Select environment from `["MoveFormMA-v0", "MoveFormMA-v1"]`
```python
ENV_NAME = "MoveFormMA-v0"
env = gym.make(ENV_NAME)
```

1. Prepare agents
```python
age = []
for i in range(env.AGE_NUM):
    age.append( some_agent() )
```

1. Send listed actions and Get listed observations and rewards for all agents together
```python
observation, reward, done, info = env.step(action)
for i in range(env.AGE_NUM):
    age[i].set_State(observation[i])
    age[i].update(reward[i])
    action[i] = age[i].get_Action()
```
