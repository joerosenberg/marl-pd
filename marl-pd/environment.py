import gym
import numpy as np
from nptyping import NDArray
from typing import Any, Union

# Define constants:
OBSERVATION_DIM = 2  # Dimension of the agent observation vector
NEIGHBOURHOOD_RADIUS = 1  # Radius of an agent's interaction neighbourhood (using Manhattan distance).

# Define types:
AgentPositions = NDArray[(Any, 2), int]  # Type of array storing positions of agents.
AgentLinearPolicyParameters = NDArray[(Any, OBSERVATION_DIM + 1), float]  # Type of array storing agent's policy
# parameters.
AgentObservations = NDArray[(Any, OBSERVATION_DIM), float]  # Type of array storing agent's observations.
AgentEnergies = NDArray[(Any, 1), int]  # Type of array storing agent's energies (accumulated payoffs that determine
# whether the agent will reproduce or die.
AgentActions = NDArray[(Any, 1), bool]  # Type of array corresponding to a list of agent actions.
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
        # Arrays for storing agent positions, energies and partners.
        self.agent_positions: AgentPositions
        self.agent_energies: AgentEnergies
        self.agent_partners: AgentPositions
        self.has_partner: AgentActions  # TODO: Add a type synonym for this

        # Array for storing agent policy parameters
        self.agent_policies: AgentLinearPolicyParameters

        # Occupancy grid stores whether or not each cell in the gridworld is occupied.
        # Allows for fast lookup of neighbours for each agent.
        self.occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.bool)

        # Cooperation and defection grid stores the number of times that the agent living at that square has cooperated
        # or defected. Using for efficiently looking up observations for agents.
        self.cooperation_grid = np.zeros((grid_height, grid_width), dtype=np.int)
        self.defection_grid = np.zeros((grid_height, grid_width), dtype=np.int)

        # Stores the payoffs for each iteration of the game.
        self.payoffs: AgentEnergies

        # Set up initial state.
        self.reset()

    def step(self, actions: NDArray[(Any, 1), bool]):
        """

        :param actions: List of actions
        :return:
        """
        # Play game between each chosen pair of agents and produce list of energy payoffs:
        # Store agent actions in grid.
        action_grid = np.zeros_like(self.occupancy_grid, dtype=np.bool)
        action_grid[self.agent_positions[:,0], self.agent_positions[:,1]] = actions

        # Retrieve partner actions.
        partner_cooperating = action_grid[self.agent_partners[:,0], self.agent_partners[:,1]]

        # Compute payoffs
        agent_cooperating = actions
        self.payoffs = np.where(self.has_partner.flatten(),
                                np.where(
                                    agent_cooperating,
                                    np.where(partner_cooperating, self.payoff_cc, self.payoff_cd),  # If current agent coops
                                    np.where(partner_cooperating, self.payoff_dc, self.payoff_dd)  # If current agent defects
                                ),
                                0.0
                            )

        # Update agent energies using payoffs from the games and the cost of living
        self.agent_energies = np.minimum(self.agent_energies + np.expand_dims(self.payoffs, 1) - self.cost_of_living, self.max_energy)

        # Kill agents with 0 or less energy
        self._cull_agents()

        # Agent reproduction:
        # For each agent with energy > self.reproduce_cost, spawn a new agent with the same policy nearby.
        # New agents spawn with energy equal to self.reproduce_cost, and agents that reproduce lose the same amount
        # of energy.
        self._spawn_agents()

        # Select partners for each agent randomly.
        self._select_partners()

        # Produce observations
        obs = self._produce_observations()

        # Return observations and list of payoffs (trimmed to remove dead agents)
        return obs, self.payoffs

    def _cull_agents(self):
        """
        Removes any agents with 0 or less energy and cleans up their data.
        :return:
        """
        # Get indices and positions of dead agents
        dead_agent_indices = np.argwhere(self.agent_energies <= 0)
        dead_agent_positions = self.agent_positions[dead_agent_indices]

        # Remove rows in agent_positions, agent_energies and agent_policies for these agents.
        self.agent_positions = np.delete(self.agent_positions, dead_agent_indices, axis=0)
        self.agent_energies = np.delete(self.agent_energies, dead_agent_indices, axis=0)
        self.agent_policies = np.delete(self.agent_policies, dead_agent_indices, axis=0)

        # Remove rows in the payoff record for these agents. (They're dead so they don't need to learn anymore).
        self.payoffs = np.delete(self.payoffs, dead_agent_indices, axis=0)

        # Register the agents' cells as being unoccupied.
        self.occupancy_grid[dead_agent_positions[:, 0], dead_agent_positions[:, 1]] = False

        # Reset the action counters for the dead agents.
        self.cooperation_grid[dead_agent_positions[:, 0], dead_agent_positions[:, 1]] = 0
        self.defection_grid[dead_agent_positions[:, 0], dead_agent_positions[:, 1]] = 0

    def _spawn_agents(self):
        """
        Spawns offspring for agents with energy greater than the reproduction threshold.
        :return:
        """
        # Get indices of agents with enough energy to reproduce.
        parent_indices = np.argwhere(self.agent_energies >= self.reproduce_threshold)

        # If there are no agents with enough energy to reproduce, we're done.
        if parent_indices.size == 0:
            return

        # Create empty lists for storing the positions, energies and policies of the new agents.
        new_positions = []
        new_energies = []
        new_policies = []
        new_payoffs = []

        for i in np.random.permutation(parent_indices):
            agent_position = self.agent_positions[i]

            # Find empty cells near the agent.
            empty_positions = self._get_unnoccupied_neighbouring_cells(agent_position)

            # If there are no empty nearby cells, the agent doesn't reproduce, and we go to the next agent.
            if not empty_positions:
                continue

            # Choose a nearby empty position at random to place the child in.
            empty_index = np.random.randint(empty_positions.shape[0])
            child_position = empty_positions[empty_index]

            # Add the position, energy and policy of the child to the lists:
            new_positions.append(child_position)
            new_energies.append([self.reproduce_cost])
            new_policies.append(self.agent_policies[i])  # Copy the parent's policy to the child
            new_payoffs.append(0)

            # Register the child in the occupancy grid
            self.occupancy_grid[child_position[0], child_position[1]] = True

            # Deduct the cost of reproduction from the parent's energy
            self.agent_energies[i] = self.agent_energies[i] - self.reproduce_cost

        # Add all of the new data to the environment state:
        self.agent_positions = np.append(self.agent_positions, new_positions, axis=0)
        self.agent_energies = np.append(self.agent_energies, new_energies, axis=0)
        self.agent_policies = np.append(self.agent_policies, new_policies, axis=0)
        self.payoffs = np.append(self.payoffs, new_payoffs, axis=0)

    def _select_partners(self):
        """
        Selects new game partners for each agent.
        :return:
        """
        # Array to keep track of which agents have been partnered up so far.
        is_partnered = np.zeros_like(self.occupancy_grid, dtype=np.bool)

        # Reset list of agent partners, using NaNs to represent agents without partners.
        self.agent_partners = np.zeros_like(self.agent_positions)   # List of agent partner positions
        self.has_partner = np.zeros_like(self.agent_energies, dtype=np.bool)

        # Create dictionary to link back partner assignments:
        existing_partners = {}

        # Loop through the agents, assigning a partner in each iteration.
        # Iterate in a random order to avoid newer agents (which appear at the bottom of the self.agent_positions list)
        # having a worse chance at finding a partner.
        for i in np.random.permutation(self.agent_positions.shape[0]):
            # Get current agent position
            agent_position = self.agent_positions[i]

            # If the agent is registered as already being partnered with another agent, look up its partner's position
            # from the existing_partners dictionary.
            # Otherwise, select a new partner from its unpartnered neighbours at random.
            if is_partnered[tuple(agent_position)]:
                # Look up existing partner from the dictionary
                new_partner_position = existing_partners.pop(agent_position.tobytes())
                self.has_partner[i] = True
                assert new_partner_position in self.agent_positions
            else:
                # Find a partner for the current agent:
                # Get the position of the current agent
                agent_position = self.agent_positions[i]
                # Get the neighbours of the current agent
                agent_neighbour_positions = self._get_neighbours(agent_position)

                # Get the neighbours of the current agent that aren't yet partnered
                is_neighbour_partnered = is_partnered[agent_neighbour_positions[:,0], agent_neighbour_positions[:,1]]
                unpartnered_neighbour_indices = np.argwhere(np.logical_not(is_neighbour_partnered))

                # If there are none, we can't find the agent a partner, so we go to the next agent
                if len(unpartnered_neighbour_indices) == 0:
                    continue

                # Select an unpartnered agent at random
                new_partner_index = np.random.choice(unpartnered_neighbour_indices.flatten())
                new_partner_position = agent_neighbour_positions[new_partner_index]

                # Update is_partnered to register both the current agent and the chosen partner as being partnered.
                is_partnered[tuple(agent_position)] = True
                is_partnered[tuple(new_partner_position)] = True

                # Write record for the other partner to read.
                existing_partners[new_partner_position.tobytes()] = agent_position
                self.has_partner[i] = True
                assert new_partner_position in self.agent_positions

            # Record agent i's partner in self.agent_partners.
            self.agent_partners[i] = new_partner_position

    def _get_neighbours(self, agent_position: NDArray[(1, 2), int]) -> AgentPositions:
        """
        Finds the living neighbours of an agent at a given position.
        :param agent_position: Position of the agent that we wish to find neighbours of.
        :return: List of positions of neighbouring agents.
        """
        # Here, we define neighbours as agents living within a square centered at the given agent.
        # The size of the square is defined by the constant NEIGHBOURHOOD_RADIUS.

        # Calculate slice indices of clipped square of self.occupancy_grid around the current agent.
        lower_slice_indices = np.clip(agent_position - NEIGHBOURHOOD_RADIUS,
                                      a_min=0, a_max=[self.grid_height, self.grid_width])
        higher_slice_indices = np.clip(agent_position + NEIGHBOURHOOD_RADIUS + 1,
                                       a_min=0, a_max=[self.grid_height, self.grid_width])

        # Use slice indices to get neighbourhood of occupancy grid around the current agent.
        lower_y, lower_x = lower_slice_indices
        higher_y, higher_x = higher_slice_indices
        neighbour_occupancies = self.occupancy_grid[lower_y:higher_y, lower_x:higher_x]

        # Set element of neighbour_occupancies corresponding to the current agent to False, since we don't want to
        # include the current agent in the list of its neighbours.
        neighbour_occupancies[NEIGHBOURHOOD_RADIUS, NEIGHBOURHOOD_RADIUS] = False

        # Get positions of neighbours:
        # np.argwhere(neighbour_occupancies) returns the indices corresponding to neighbours in the smaller array
        # neighbour_occupancies, so we need to translate these indices to recover the actual positions.
        # lower_y and lower_x give the amount we need to translate by.
        neighbour_positions = np.array([lower_y, lower_x]) + np.argwhere(neighbour_occupancies)
        return neighbour_positions.astype(np.int)

    def _get_unnoccupied_neighbouring_cells(self, agent_position: NDArray[(1, 2), int]) -> AgentPositions:
        """
        Finds the unoccupied cells near an agent at a given position.
        :param agent_position: Position of the agent that we wish to find empty cells near.
        :return: List of positions of neighbouring empty cells.
        """
        # TODO: Refactor this along with _get_neighbours to remove duplicate code.
        # The size of the square is defined by the constant NEIGHBOURHOOD_RADIUS.

        # Calculate slice indices of clipped square of self.occupancy_grid around the current agent.
        lower_slice_indices = np.clip(agent_position - NEIGHBOURHOOD_RADIUS,
                                      a_min=0, a_max=[self.grid_height, self.grid_width])
        higher_slice_indices = np.clip(agent_position + NEIGHBOURHOOD_RADIUS + 1,
                                       a_min=0, a_max=[self.grid_height, self.grid_width])

        # Use slice indices to get neighbourhood of occupancy grid around the current agent.
        lower_y, lower_x = lower_slice_indices
        higher_y, higher_x = higher_slice_indices
        neighbour_occupancies = self.occupancy_grid[lower_y:higher_y, lower_x:higher_x]

        # Set element of neighbour_occupancies corresponding to the current agent to False, since we don't want to
        # include the current agent in the list of its neighbours.
        neighbour_occupancies[NEIGHBOURHOOD_RADIUS, NEIGHBOURHOOD_RADIUS] = False

        # Get positions of neighbours:
        # np.argwhere(neighbour_occupancies) returns the indices corresponding to neighbours in the smaller array
        # neighbour_occupancies, so we need to translate these indices to recover the actual positions.
        # lower_y and lower_x give the amount we need to translate by.
        empty_positions = np.array([lower_y, lower_x]) + np.argwhere(np.logical_not(neighbour_occupancies))
        return empty_positions

    def reset(self) -> AgentObservations:
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
                np.random.choice(self.grid_height * self.grid_width,
                                 size=(self.nb_initial_agents, 1),
                                 replace=False),
                self.grid_width),
            axis=1
        )

        # Reset and populate the occupancy grid using the agent positions.
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.bool)
        self.occupancy_grid[self.agent_positions[:,0], self.agent_positions[:,1]] = True

        # Reset the cooperation and defection grids.
        self.cooperation_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int)
        self.defection_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int)

        # Initialise the agent policies:
        # The first coop_agent_cutoff agents in the list are cooperative
        coop_agent_cutoff = int(self.nb_initial_agents * self.proportion_cooperative_agents)
        # We set the bias of the linear model so that the cooperative agents have an initial_action_probability chance
        # of cooperating, and the defecting agents have an initial_action_probability of defecting.
        initial_bias_magnitude = np.log(self.initial_action_probability/(1 - self.initial_action_probability))
        self.agent_policies = np.zeros((self.nb_initial_agents, OBSERVATION_DIM + 1))
        self.agent_policies[:coop_agent_cutoff, 0] = initial_bias_magnitude
        self.agent_policies[coop_agent_cutoff:, 0] = -initial_bias_magnitude

        # Initialise the agent energies
        # TODO: add separate parameter for starting energy. At the moment we initialise agents with energy equal to
        # the reproduction cost, as is done in the Smaldino paper.
        self.agent_energies = np.full((self.nb_initial_agents, 1), self.reproduce_cost)

        # Choose initial partners
        self._select_partners()

        # Return the initial observations (i.e. all zeroes).
        obs = self._produce_observations()
        return obs

    def _produce_observations(self) -> AgentObservations:
        """
        For each agent, gets the number of times that it has cooperated and defected, as well as the number of times
        that its current partner has cooperated and defected. Then, formats these as a list of agent observations.
        :param partner_positions: List of positions of the partners for each agent. The i_th entry in partner_positions
        corresponds to the partner for the i_th entry of self.agent_positions.
        :return: List of agent observations for the current state and agent partners.
        """

        # Retrieve agent's own cooperation and defection records
        own_coops = self.cooperation_grid[self.agent_positions[:, 0], self.agent_positions[:, 1]]
        own_defects = self.defection_grid[self.agent_positions[:, 0], self.agent_positions[:, 1]]

        # Retrieve partners' cooperation and defection records:
        # We return 0, 0 if the agent doesn't have a partner.
        # Get mask for agents that do have partners.
        indices = self.agent_partners
        partner_coops = np.where(self.has_partner.flatten(), self.cooperation_grid[indices[:, 0], indices[:, 1]], 0)
        partner_defects = np.where(self.has_partner, self.defection_grid[indices[:, 0], indices[:, 1]], 0)

        # Concatenate into one array to produce observations
        # i_th row has observations for the i_th agent in self.agent_positions
        obs = np.column_stack((own_coops, own_defects, partner_coops, partner_defects))
        return obs

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    # Try creating an environment and taking a step without crashing...
    env = MAPDEnvironment(10, 10, 45, 100, 0.5)
    print(env.step(np.ones(45, dtype=np.bool)))
