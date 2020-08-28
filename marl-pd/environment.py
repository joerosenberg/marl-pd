import gym
import numpy as np
from nptyping import NDArray
from typing import Any

# Define types + type constants
OBSERVATION_DIM = 2
AgentPositions = NDArray[(Any, 2), int]  # Type of array storing positions of agents.
AgentLinearPolicyParameters = NDArray[(Any, OBSERVATION_DIM + 1), float]  # Type of array storing agent's policy
# parameters.
AgentObservations = NDArray[(Any, OBSERVATION_DIM), float]  # Type of array storing agent's observations.
AgentEnergies = NDArray[(Any, 1), int]  # Type of array storing agent's energies (accumulated payoffs that determine
# whether the agent will reproduce or die.


class MAPDEnvironment(gym.Env):
    def __init__(self, grid_height: int, grid_width: int, nb_initial_agents: int, proportion_cooperative_agents: float,
                 reproduce_threshold: int = 100, reproduce_cost: int = 50, payoff_cc: int = 3, payoff_cd: int = -1,
                 payoff_dc: int = 5, payoff_dd: int = 0, max_energy: int = 150):
        """
        Initialises a multi-agent gridworld where agents play iterated games with eachother, and agents reproduce
        (passing on their behaviour) or die off based on their accumulated payoffs.

        The parameters define the game and reproduction dynamics. By default, the parameters specify a variant of the
        prisoner's dilemma, but in principle we can specify any two-player game.

        The agent's action preferences are modelled using a linear policy as a function of observation
        features. These features are measures of how neighbouring agents have recently acted.

        We randomly initialise a proportion of agents who prefer to act cooperatively, and initialise the rest as
        preferring to defect. This is done by initialising all agent's linear policy weights to zero, and then setting
        biases depending on whether the agent is initialised as cooperative or defecting. (This gives agents who
        start off 'blindly' cooperating or defecting, regardless of their neighbours' recent actions).

        :param grid_height: Height of the gridworld that the agents live in.
        :param grid_width: Width of the gridworld that the agents live in.
        :param nb_initial_agents: Number of agents to initially populate the world with.
        :param proportion_cooperative_agents: Proportion of agents (as number between 0 and 1) to initialise as cooperative.
        :param reproduce_threshold: Energy threshold at which agents reproduce.
        :param reproduce_cost: Amount of energy to subtract from an agent when it reproduces.
        :param payoff_cc: Payoff given to an agent when it cooperates with an agent who cooperates.
        :param payoff_dc: Payoff given to an agent when it defects with an agent who cooperates.
        :param payoff_dd: Payoff given to an agent when it defects with an agent who defects.
        :param max_energy: Maximum amount of energy an agent can store.
        """

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
