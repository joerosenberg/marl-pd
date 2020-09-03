import gym
import numpy as np
from nptyping import NDArray
from typing import Any, Union
from gym.envs.classic_control import rendering

# Define constants:
OBSERVATION_DIM = 4  # Dimension of the agent observation vector
NEIGHBOURHOOD_RADIUS = 1  # Radius of an agent's interaction neighbourhood (using Manhattan distance).

# Rendering constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

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
        self.occupancy_grid = np.zeros((grid_width, grid_height), dtype=np.bool)

        # Cooperation and defection grid stores the number of times that the agent living at that square has cooperated
        # or defected. Using for efficiently looking up observations for agents.
        self.cooperation_grid = np.zeros((grid_width, grid_height), dtype=np.int)
        self.defection_grid = np.zeros((grid_width, grid_height), dtype=np.int)

        # Stores the payoffs for each iteration of the game.
        self.payoffs: AgentEnergies

        # Viewer for rendering
        self.viewer = None

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

        # Update cooperation and defection grid for agents with partners
        cooperating_agent_locations = self.agent_positions[np.logical_and(self.has_partner.flatten(), actions)]
        defecting_agent_locations = self.agent_positions[np.logical_and(self.has_partner.flatten(), np.logical_not(actions))]

        self.cooperation_grid[cooperating_agent_locations[:,0], cooperating_agent_locations[:,1]] += 1
        self.defection_grid[defecting_agent_locations[:,0], defecting_agent_locations[:,1]] += 1

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
        dead_agent_indices = np.argwhere(self.agent_energies <= 0)[:,0]
        dead_agent_positions = self.agent_positions[dead_agent_indices]

        # If no agents are dead, stop here
        if dead_agent_indices.size == 0:
            return

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
        parent_indices = np.argwhere(self.agent_energies >= self.reproduce_threshold)[:,0]

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

            # Get empty cell near the agent.
            child_position = self._get_random_unoccupied_neighbouring_cell(agent_position)

            # If there are no empty nearby cells, the agent doesn't reproduce, and we go to the next agent.
            if child_position is None:
                continue

            # Add the position, energy and policy of the child to the lists:
            new_positions.append(child_position)
            new_energies.append([self.reproduce_cost])
            new_policies.append(self.agent_policies[i])  # Copy the parent's policy to the child
            new_payoffs.append(0)

            # Register the child in the occupancy grid
            self.occupancy_grid[child_position[0], child_position[1]] = True

            # Deduct the cost of reproduction from the parent's energy
            self.agent_energies[i] = self.agent_energies[i] - self.reproduce_cost

        # If we didn't manage to spawn any new agents, stop here
        if len(new_positions) == 0:
            return

        # Add all of the new data to the environment state:
        self.agent_positions = np.append(self.agent_positions, np.array(new_positions), axis=0)
        self.agent_energies = np.append(self.agent_energies, new_energies, axis=0)
        self.agent_policies = np.append(self.agent_policies, new_policies, axis=0)
        self.payoffs = np.append(self.payoffs, new_payoffs, axis=0)

    def _select_partners(self):
        """
        Selects new game partners for each agent.
        :return:
        """
        # Grid to keep track of which agents have been partnered up so far.
        self.is_partnered = np.zeros_like(self.occupancy_grid, dtype=np.bool)

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
            if self.is_partnered[tuple(agent_position)]:
                # Look up existing partner from the dictionary
                new_partner_position = existing_partners.pop(agent_position.astype(np.int).tobytes())
                self.has_partner[i] = True
            else:
                # Find a partner for the current agent:
                # Get a random unpartnered neighbour for the current agent
                new_partner_position = self._get_random_neighbouring_unpartnered_agent(agent_position)

                # If there are none, we can't find the agent a partner, so we go to the next agent
                if new_partner_position is None:
                    continue

                # Update is_partnered to register both the current agent and the chosen partner as being partnered.
                self.is_partnered[tuple(agent_position)] = True
                self.is_partnered[tuple(new_partner_position)] = True

                # Write record for the other partner to read.
                existing_partners[new_partner_position.astype(np.int).tobytes()] = agent_position
                self.has_partner[i] = True

            # Record agent i's partner in self.agent_partners.
            self.agent_partners[i] = new_partner_position

    def _get_neighbours(self, agent_position: NDArray[(1, 2), int]) -> AgentPositions:
        """
        Finds the living neighbours of an agent at a given position.
        :param agent_position: Position of the agent that we wish to find neighbours of.
        :return: List of positions of neighbouring agents.
        """

        # Use slice indices to get neighbourhood of occupancy grid around the current agent.
        lower_x = self.lower_xs[tuple(agent_position)]
        lower_y = self.lower_ys[tuple(agent_position)]
        upper_x = self.upper_xs[tuple(agent_position)]
        upper_y = self.upper_ys[tuple(agent_position)]

        neighbour_occupancies = self.occupancy_grid[lower_x:upper_x, lower_y:upper_y].copy()

        # Set element of neighbour_occupancies corresponding to the current agent to False, since we don't want to
        # include the current agent in the list of its neighbours.
        neighbour_occupancies[NEIGHBOURHOOD_RADIUS, NEIGHBOURHOOD_RADIUS] = False

        # Get positions of neighbours:
        # np.argwhere(neighbour_occupancies) returns the indices corresponding to neighbours in the smaller array
        # neighbour_occupancies, so we need to translate these indices to recover the actual positions.
        # lower_y and lower_x give the amount we need to translate by.
        neighbour_positions = np.array([lower_x, lower_y]) + np.argwhere(neighbour_occupancies)
        return neighbour_positions.astype(np.int)

    def _get_neighbourhood_bounds(self, agent_position):
        """
        Gets the boundaries of the neighbourhood around a cell.
        :param agent_position:
        :return:
        """
        # Use slice indices to get neighbourhood of occupancy grid around the current agent.
        lower_x = self.lower_xs[tuple(agent_position)]
        lower_y = self.lower_ys[tuple(agent_position)]
        upper_x = self.upper_xs[tuple(agent_position)]
        upper_y = self.upper_ys[tuple(agent_position)]
        return lower_x, lower_y, upper_x, upper_y

    def _get_random_unoccupied_neighbouring_cell(self, agent_position):
        """
        Gets a random unoccupied cell near an agent at a given position.
        Faster when trying to find one empty cell because we don't necessarily need to iterate through all neighbours.
        :param agent_position:
        :return:
        """
        # Get occupancies in neighbourhood around agent
        lower_x, lower_y, upper_x, upper_y = self._get_neighbourhood_bounds(agent_position)
        neighbour_occupancies = self.occupancy_grid[lower_x:upper_x, lower_y:upper_y].copy()

        # Iterate through neighbourhood in random order until we find an empty cell, then return its position
        # If we don't find one, return None.
        nbhd_width = neighbour_occupancies.shape[0]
        nbhd_height = neighbour_occupancies.shape[1]
        indices = np.transpose(
            [
                np.tile(np.arange(0, nbhd_width, dtype=np.int), nbhd_height),
                np.repeat(np.arange(0, nbhd_height, dtype=np.int), nbhd_width)
            ]
        )

        for nbhd_position in np.random.permutation(indices):
            if not neighbour_occupancies[tuple(nbhd_position)]:
                return nbhd_position + np.array([lower_x, lower_y])

        return None

    def _get_random_neighbouring_unpartnered_agent(self, agent_position):
        """
        Gets a random unpartnered agent in the neighbourhood of an agent.
        :param agent_position:
        :return:
        """
        # Get occupancies in neighbourhood around agent
        lower_x, lower_y, upper_x, upper_y = self._get_neighbourhood_bounds(agent_position)
        neighbour_occupancies = self.occupancy_grid[lower_x:upper_x, lower_y:upper_y].copy()
        neighbour_is_partnered = self.is_partnered[lower_x:upper_x, lower_y:upper_y].copy()

        # Iterate through neighbourhood in random order until we find an empty cell, then return its position
        # If we don't find one, return None.
        nbhd_width = neighbour_occupancies.shape[0]
        nbhd_height = neighbour_occupancies.shape[1]
        indices = np.transpose(
            [
                np.tile(np.arange(0, nbhd_width, dtype=np.int), nbhd_height),
                np.repeat(np.arange(0, nbhd_height, dtype=np.int), nbhd_width)
            ]
        )

        for nbhd_position in np.random.permutation(indices):
            if neighbour_occupancies[tuple(nbhd_position)] and not neighbour_is_partnered[tuple(nbhd_position)]:
                return nbhd_position + np.array([lower_x, lower_y])

        return None

    def _get_unnoccupied_neighbouring_cells(self, agent_position: NDArray[(1, 2), int]) -> AgentPositions:
        """
        Finds the unoccupied cells near an agent at a given position.
        :param agent_position: Position of the agent that we wish to find empty cells near.
        :return: List of positions of neighbouring empty cells.
        """

        lower_x, lower_y, upper_x, upper_y = self._get_neighbourhood_bounds(agent_position)

        neighbour_occupancies = self.occupancy_grid[lower_x:upper_x, lower_y:upper_y].copy()

        # Get positions of neighbours:
        # np.argwhere(neighbour_occupancies) returns the indices corresponding to neighbours in the smaller array
        # neighbour_occupancies, so we need to translate these indices to recover the actual positions.
        # lower_y and lower_x give the amount we need to translate by.
        empty_positions = np.array([lower_x, lower_y]) + np.argwhere(np.logical_not(neighbour_occupancies))
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
                self.grid_height),
            axis=1
        )

        # Reset and populate the occupancy grid using the agent positions.
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.bool)
        self.occupancy_grid[self.agent_positions[:,0], self.agent_positions[:,1]] = True

        # Reset the cooperation and defection grids.
        self.cooperation_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int)
        self.defection_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int)

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

        # Precompute the neighbourhood boundaries for all agents.
        self._compute_neighbourhood_boundaries()

        # Choose initial partners
        self._select_partners()

        # Return the initial observations (i.e. all zeroes).
        obs = self._produce_observations()
        return obs

    def _compute_neighbourhood_boundaries(self):
        """
        Computes and stores the neighbourhood boundaries for each grid cell.
        :return:
        """
        self.lower_xs = np.zeros_like(self.occupancy_grid, dtype=np.int)
        self.lower_ys = np.zeros_like(self.occupancy_grid, dtype=np.int)
        self.upper_xs = np.zeros_like(self.occupancy_grid, dtype=np.int)
        self.upper_ys = np.zeros_like(self.occupancy_grid, dtype=np.int)

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                lower_x = max(x - NEIGHBOURHOOD_RADIUS, 0)
                lower_y = max(y - NEIGHBOURHOOD_RADIUS, 0)
                upper_x = min(x + NEIGHBOURHOOD_RADIUS + 1, self.grid_width + 1)
                upper_y = min(y + NEIGHBOURHOOD_RADIUS + 1, self.grid_height + 1)
                self.lower_xs[x, y] = lower_x
                self.lower_ys[x, y] = lower_y
                self.upper_xs[x, y] = upper_x
                self.upper_ys[x, y] = upper_y

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
        partner_defects = np.where(self.has_partner.flatten(), self.defection_grid[indices[:, 0], indices[:, 1]], 0)

        # Concatenate into one array to produce observations
        # i_th row has observations for the i_th agent in self.agent_positions
        obs = np.column_stack((own_coops, own_defects, partner_coops, partner_defects))
        return obs

    def render(self, mode='human'):
        cell_width = SCREEN_WIDTH / self.grid_width  # Width of one grid cell
        cell_height = SCREEN_HEIGHT / self.grid_height  # Height of one grid cell
        scale = np.array([cell_width, cell_height])
        square_offsets = np.array([[-cell_width, -cell_height],
                                   [cell_width, -cell_height],
                                   [cell_width, cell_height],
                                   [-cell_width, cell_height]]) * 0.4

        # Initialise viewer if it hasn't yet been created
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Compute agent colours
        self._compute_colours()

        for i in range(self.agent_positions.shape[0]):
            # Get agent colour
            colour = self.colours[i]
            # Draw a square for each agent
            self.viewer.draw_polygon((self.agent_positions[i] * scale + scale/2) + square_offsets, color=colour)

            # Draw lines between interacting agents
            if self.has_partner[i]:
                start = self.agent_positions[i] * scale + scale/2
                end = self.agent_partners[i] * scale + scale/2
                self.viewer.draw_line(start, end)

        return self.viewer.render()

    def _compute_colours(self):
        """
        Computes colourings for all agents.
        :return:
        """
        # Colour based on bias parameter - linearly interpolate between red (for defect) and blue (for cooperate)
        biases = self.agent_policies[:,0]
        bias_min = np.min(biases)
        bias_max = np.max(biases)

        blue = np.array([0, 0, 1.0])
        red = np.array([1.0, 0, 0])

        normalised_biases = np.expand_dims((biases - bias_min) / (bias_max - bias_min), axis=1)

        self.colours = normalised_biases * blue + (1 - normalised_biases) * red

if __name__ == '__main__':
    # Try creating an environment and taking a step without crashing...
    env = MAPDEnvironment(10, 10, 45, 100, 0.5)
    print(env.step(np.zeros(45, dtype=np.bool)))
