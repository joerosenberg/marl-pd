from marlpd.environment import MAPDEnvironment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def policy_pca(grid_size, nb_steps, nb_checkpoints, nb_clusters):
    """
    Run environment for a bunch of steps, do PCA on policies and save graphs
    :param grid_size: height and width of grid
    :param nb_steps: number of steps to run environment for
    :param nb_checkpoints: number of graphs to produce (evenly spaced along timesteps)
    :param nb_clusters: number of clusters to produce
    :return: Cluster means
    """
    env = MAPDEnvironment(grid_size, grid_size, grid_size, grid_size**2, 0.5)

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
        plt.savefig('test_{}.png'.format(i))

    # Cluster the last checkpoint
    final_policies = checkpointed_policies[nb_checkpoints-1]
    final_reduced_policies = pca.transform(final_policies)
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(final_policies)
    # Plot final policies coloured by cluster
    plt.scatter(final_reduced_policies[:,0], final_reduced_policies[:,1], c=kmeans.predict(final_policies))
    plt.savefig('testfinal.png')

    return kmeans.cluster_centers_


# Define sample observations for testing responses
both_always_coop = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
both_always_defect = np.array([0, 0, 1, 1, 0, 0, 1, 1])

means = np.array(policy_pca(10, 1000, 5, 2))
print(means)
print(means[:,1:].shape)
print(both_always_coop.shape)
# Compute responses
responses_1 = 1/(1 + np.exp(-means[:,0] - (means[:,1:] * both_always_coop).sum(axis=1)))
print(responses_1)