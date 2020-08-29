import gym
import numpy as np
from nptyping import NDArray
from typing import Any, Union

# Define types + type constants
OBSERVATION_DIM = 2
AgentPositions = NDArray[(Any, 2), int]  # Type of array storing positions of agents.
AgentLinearPolicyParameters = NDArray[(Any, OBSERVATION_DIM + 1), float]  # Type of array storing agent's policy
# parameters.
AgentObservations = NDArray[(Any, OBSERVATION_DIM), float]  # Type of array storing agent's observations.
AgentEnergies = NDArray[(Any, 1), int]  # Type of array storing agent's energies (accumulated payoffs that determine
# whether the agent will reproduce or die.
AgentActions = NDArray[(Any, 1), bool] # Type of array corresponding to a list of agent actions.
# True represents cooperation, False represents defection.


class MAPDEnvironment(gym.Env):
    def __init__(self, grid_height: int, grid_width: int, nb_initial_agents: int, max_agents: int,
                 proportion_cooperative_agents: float, initial_action_probability: float = 0.9,
                 reproduce_threshold: float = 100, reproduce_cost: float = 50,
                 payoff_cc: float = 3, payoff_cd: float = -1, payoff_dc: float = 5, payoff_dd: float = 0,
                 max_energy: float = 150, cost_of_living: float = 0.5):
        """
        This code is based on the model in the paper "Increased Costs of Cooperation Help Cooperators in the Long Run"
        by Smaldino, Schank and McElreath published in The American Naturalist in 2013. The accompanying NetLogo code
        for the paper was also used.

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
        :param max_agents: The maximum number of agents that can be alive at any one time. Reproduction will be halted
        if the number of living agents exceeds this.
        :param proportion_cooperative_agents: Proportion of agents (as number between 0 and 1) to initialise as cooperative.
        :param reproduce_threshold: Energy threshold at which agents reproduce.
        :param reproduce_cost: Amount of energy to subtract from an agent when it reproduces.
        :param payoff_cc: Payoff given to an agent when it cooperates with an agent who cooperates.
        :param payoff_cd: Payoff given to an agent when it cooperates with an agent who defects.
        :param payoff_dc: Payoff given to an agent when it defects with an agent who cooperates.
        :param payoff_dd: Payoff given to an agent when it defects with an agent who defects.
        :param max_energy: Maximum amount of energy an agent can store.
        :param cost_of_living: The amount of energy that agents lose per time step, regardless of the action they take.
        """

        # Store simulation parameters:
        # Grid size parameters
        self.grid_height = grid_height
        self.grid_width = grid_width
        # We can't store more agents than there are spaces on the grid
        self.max_agents = min(grid_height * grid_width, max_agents)

        # Initial agent distribution parameters
        self.nb_initial_agents = nb_initial_agents
        self.proportion_cooperative_agents = proportion_cooperative_agents
        self.initial_action_probability = initial_action_probability

        # Reproduction parameters
        self.reproduce_threshold = reproduce_threshold
        self.reproduce_cost = reproduce_cost

        # Game payoff parameters
        self.payoff_cc = payoff_cc
        self.payoff_cd = payoff_cd
        self.payoff_dc = payoff_dc
        self.payoff_dd = payoff_dd

        # Energy capacity and passive use parameters
        self.max_energy = max_energy
        self.cost_of_living = cost_of_living

        # Initialise variables for storing environment state:
        # Arrays for storing agent positions and energies
        self.agent_positions: AgentPositions
        self.agent_energies: AgentEnergies

        # Array for storing agent policy parameters
        self.agent_policies: AgentLinearPolicyParameters

        # Occupancy grid stores whether or not each cell in the gridworld is occupied.
        # Allows for fast lookup of neighbours for each agent.
        self.occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.bool)

        # Set up initial state.
        self.reset()

    def step(self, action):
        pass

    def reset(self):
        """
        Resets the state of the gridworld according to the simulation parameters.
        :return:
        """
        # Generate random locations for the initial agents.
        # We produce locations by sampling random integers without replacement from 0, ..., grid_height*grid_width - 1,
        # and then calculating x, y such that n = grid_width * x + y, where 0 <= y < grid_height.
        # This gives exactly nb_initial_agents different locations.
        self.agent_positions = np.concatenate(
            np.divmod(
                np.random.randint(0,
                                  high=self.grid_height * self.grid_width,
                                  size=(self.nb_initial_agents, 1)),
                self.grid_width),
            axis=1
        )

        # Populate the occupancy grid using the agent positions.
        self.occupancy_grid[self.agent_positions] = True

        # Initialise the agent policies:
        # The first coop_agent_cutoff agents in the list are cooperative
        coop_agent_cutoff = int(self.nb_initial_agents * self.proportion_cooperative_agents)
        # We set the bias of the linear model so that the cooperative agents have an initial_action_probability chance
        # of cooperating, and the defecting agents have an initial_action_probability of defecting.
        initial_bias_magnitude = np.log(self.initial_action_probability/(1 - self.initial_action_probability))
        self.agent_policies = np.zeros((self.nb_initial_agents, OBSERVATION_DIM + 1))
        self.agent_policies[:coop_agent_cutoff, 0] = initial_bias_magnitude
        self.agent_positions[coop_agent_cutoff:, 0] = -initial_bias_magnitude

        # Initialise the agent energies
        # TODO: add separate parameter for starting energy. At the moment we initialise agents with energy equal to
        # the reproduction cost, as is done in the Smaldino paper.
        self.agent_energies = np.ones((self.nb_initial_agents, 1)) * self.reproduce_cost

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    MAPDEnvironment(10, 10, 30, 50, 0.5)
