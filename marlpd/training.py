from marlpd.environment import MAPDEnvironment
import jax.numpy as jnp
from jax import grad, jit

GRID_WIDTH = 100
GRID_HEIGHT = 100
INITIAL_AGENTS = GRID_WIDTH * GRID_HEIGHT // 10
MAX_AGENTS = GRID_HEIGHT * GRID_WIDTH

# Create new environment
env = MAPDEnvironment(GRID_HEIGHT, GRID_WIDTH, INITIAL_AGENTS, MAX_AGENTS)

# Get initial observations
obs = jnp.array(env.reset())
# Get initial policies
policies = jnp.array(env.agent_policies)


@jit
def cooperation_logits(policies, obs):
    """
    Calculates and returns the logits for the agent's policy model.
    :param policies: List of agent's policies.
    :param obs: List of agent's observations.
    :return: List of logit values.
    """
    # Bias is stored in the first column
    logits = policies[:, 0] + jnp.dot(policies[:, 1:], obs)
    return logits


@jit
def cooperation_probabilities(policies, obs):
    logits = cooperation_logits(policies, obs)
    probs = jnp.exp(logits) / (1 + jnp.exp(logits))
    return probs

