from marlpd.environment import MAPDEnvironment, OBSERVATION_DIM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from test_parameters import GRID_SIZE, NB_STEPS, NB_CHECKPOINTS, NB_CLUSTERS

def policy_pca(grid_size, nb_steps, nb_checkpoints, nb_clusters):
    """
    Run environment for a bunch of steps, do PCA on policies and save graphs
    :param grid_size: height and width of grid
    :param nb_steps: number of steps to run environment for
    :param nb_checkpoints: number of graphs to produce (evenly spaced along timesteps)
    :param nb_clusters: number of clusters to produce
    :return: Cluster means
    """
    env = MAPDEnvironment(grid_size, grid_size, grid_size, grid_size**2, 0.5, initial_action_probability=0.5)

    checkpoints = np.linspace(0, nb_steps-1, num=nb_checkpoints, dtype=np.int)
    checkpointed_policies = []  # Array to store policies we want to analyse

    # Run simulation and save policies at checkpoints
    for i in range(nb_steps):
        env.step()
        if i in checkpoints:
            checkpointed_policies.append(env.agent_policies.copy())

    # Run PCA on all policies:
    all_policies = np.concatenate(checkpointed_policies)

    # Only keep two dimensions
    pca = PCA(n_components=2)
    pca.fit(all_policies)

    # Produce plot for each checkpoint
    for i in range(nb_checkpoints):
        reduced_policies = pca.transform(checkpointed_policies[i])
        plt.figure()
        plt.scatter(reduced_policies[:,0], reduced_policies[:,1])
        plt.savefig('test_mixed{}.png'.format(i))

    # Cluster the last checkpoint
    final_policies = checkpointed_policies[nb_checkpoints-1]
    final_reduced_policies = pca.transform(final_policies)
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(final_policies)
    # Plot final policies coloured by cluster
    print(kmeans.predict(final_policies))
    plt.scatter(final_reduced_policies[:,0], final_reduced_policies[:,1], c=kmeans.predict(final_policies))
    plt.savefig('testfinal_mixed.png')

    return kmeans.cluster_centers_


def always_coop_agent(obs):
    return True

def always_defect_agent(obs):
    return False

def tit_for_tat_agent(obs):
    other_cooperated = obs[4]
    other_defected = obs[6]
    # If it's the first step, cooperate
    if not other_cooperated and not other_defected:
        return True
    elif other_cooperated:
        return True
    elif other_defected:
        return False


def test_agent(policy, other_agent, nb_steps, filename):
    # Plays a linear policy agent against another agent and plots the cumulative payoff received by the policy agent
    # over time.
    PAYOFF_CC = 3
    PAYOFF_CD = -1
    PAYOFF_DC = 5
    PAYOFF_DD = 0

    # Set up initial obs
    policy_obs = np.zeros(OBSERVATION_DIM)
    other_obs = np.zeros(OBSERVATION_DIM)

    # Store policy agent's payoffs in this array
    policy_payoffs = np.zeros(nb_steps)
    # Other agent's payoffs
    other_payoffs = np.zeros(nb_steps)

    for i in range(nb_steps):
        # Get action of policy agent
        coop_prob = 1/(1 + np.exp(-policy[0] - (policy[1:] * policy_obs).sum()))
        policy_coop = np.random.random(1) <= coop_prob

        # Get action of other agent
        other_coop = other_agent(other_obs)

        # Determine payoff for policy agent
        if policy_coop and other_coop: payoff = PAYOFF_CC
        elif policy_coop and not other_coop: payoff = PAYOFF_CD
        elif not policy_coop and other_coop: payoff = PAYOFF_DC
        elif not policy_coop and not other_coop: payoff = PAYOFF_DD

        # Store policy payoff
        policy_payoffs[i] = payoff

        # Determine payoff for other agent
        if policy_coop and other_coop: payoff = PAYOFF_CC
        elif policy_coop and not other_coop: payoff = PAYOFF_DC
        elif not policy_coop and other_coop: payoff = PAYOFF_CD
        elif not policy_coop and not other_coop: payoff = PAYOFF_DD

        # Store other payoff
        other_payoffs[i] = payoff

        # Update observations for both agents
        if policy_coop:
            policy_obs[0] = 1
            policy_obs[1] = policy_obs[1]/2 + 1/2
            policy_obs[2] = 0
            policy_obs[3] = policy_obs[3]/2
        elif not policy_coop:
            policy_obs[0] = 0
            policy_obs[1] = policy_obs[1]/2
            policy_obs[2] = 1
            policy_obs[3] = policy_obs[3]/2 + 1/2

        if other_coop:
            policy_obs[4] = 1
            policy_obs[5] = policy_obs[5] / 2 + 1 / 2
            policy_obs[6] = 0
            policy_obs[7] = policy_obs[7] / 2
        elif not other_coop:
            policy_obs[4] = 0
            policy_obs[5] = policy_obs[5] / 2
            policy_obs[6] = 1
            policy_obs[7] = policy_obs[7] / 2 + 1 / 2

        other_obs[:4] = policy_obs[4:]
        other_obs[4:] = policy_obs[:4]

    # Plot cumulative payoffs
    plt.figure()
    plt.plot(np.cumsum(policy_payoffs), label='Policy agent')
    plt.plot(np.cumsum(other_payoffs), label=other_agent.__name__)
    plt.legend()
    np.set_printoptions(precision=2)
    plt.title('Agent policy: {}'.format(policy), fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('Cumulative payoff')
    plt.savefig(filename)


means = np.array(policy_pca(GRID_SIZE, NB_STEPS, NB_CHECKPOINTS, NB_CLUSTERS))
print(means)

# Compute responses
for i in range(means.shape[0]):
    test_agent(means[i], always_coop_agent, 500, 'always_coop_mixed{}.png'.format(i))
    test_agent(means[i], always_defect_agent, 500, 'always_defect_mixed{}.png'.format(i))
    test_agent(means[i], tit_for_tat_agent, 500, 'tit_for_tat_mixed{}.png'.format(i))