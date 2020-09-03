from marlpd.environment import MAPDEnvironment
import numpy as np

GRID_WIDTH = 50
GRID_HEIGHT = 50
INITIAL_AGENTS = GRID_WIDTH * GRID_HEIGHT // 15
MAX_AGENTS = GRID_HEIGHT * GRID_WIDTH
PROPORTION_COOP_AGENTS = 0.5

# Create new environment
env = MAPDEnvironment(GRID_HEIGHT, GRID_WIDTH, INITIAL_AGENTS, MAX_AGENTS, PROPORTION_COOP_AGENTS, cost_of_living=0.5, initial_action_probability=0.99)

# Get initial observations
obs = env.reset()
# Get initial policies
policies = env.agent_policies


def cooperation_logits(policies, obs):
    """
    Calculates and returns the logits for the agent's policy model.
    :param policies: List of agent's policies.
    :param obs: List of agent's observations.
    :return: List of logit values.
    """
    # Bias is stored in the first column
    logits = policies[:, 0] + np.tensordot(policies[:, 1:], obs, axes=2)
    return logits


def cooperation_probabilities(policies, obs):
    logits = cooperation_logits(policies, obs)
    probs = np.exp(logits) / (1 + np.exp(logits))
    return probs

i=0
while True:
    # Calculate action probabilities for each agent
    coop_probs = cooperation_probabilities(policies, obs)
    # Select action based on these probabilities: cooperate with given probability
    actions = np.random.rand(*coop_probs.shape) <= coop_probs
    # Take step in environment
    new_obs, rewards = env.step(actions)

    # Read new policies
    policies = env.agent_policies
    print(policies)

    # Update observations
    obs = new_obs

    # Render
    env.render()

    i = i+1
    if i % 100 == 0: print(i)