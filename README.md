# gym_multiagent

# Dependency

[OpenAI Gym](https://github.com/openai/gym)

# Installation

```bash
git clone https://github.com/kbys-t/gym_MA.git
cd gym_MO
pip install -e .
```

# How to use
1. First of all,
`import gym_multiagent`

1. Select environment from `["MoveFormMA-v0"]`
```python
ENV_NAME = "MoveFormMA-v0"
env = gym.make(ENV_NAME)
```

1. Prepare agents
```python

```

1. Send actions and Get observations and rewards for all agents together
```python
action = np.concatenate((action, objective))
observation, reward, done, info = env.step(action)
```
