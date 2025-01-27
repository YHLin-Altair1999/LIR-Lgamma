import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from matplotlib.patches import Circle

def find_minimal_enclosing_radius_kdtree(positions, M):
    """
    Find approximate minimal radius that encloses M closest particles using k-d tree.

    Parameters:
    positions (np.ndarray): Array of shape (N, 3) containing particle coordinates
    M (int): Number of closest particles to consider (including the particle itself)

    Returns:
    np.ndarray: Array of shape (N, 1) containing minimal enclosing radii
    """
    # Build k-d tree
    print('Building K-D tree')
    tree = cKDTree(positions)

    # Query M nearest neighbors for each point
    # k = M because the point itself is included
    print('Sorting for distance')
    distances, _ = tree.query(positions, k=M)

    # Get the distance to the farthest neighbor (M-1 index because of 0-based indexing)
    minimal_radii = distances[:, M-1]

    return minimal_radii#.reshape(-1, 1)

if __name__ == "__main__":
    '''
    A quick demo.
    '''
    # Generate random particle positions
    N = 1000  # number of particles
    positions = np.random.rand(N, 3)

    # Find minimal radius enclosing 5 closest particles
    M = 5
    radii = find_minimal_enclosing_radius_kdtree(positions, M)
    
    fig, ax = plt.subplots()
    ax.scatter(positions[:,0], positions[:,1])
    for i in range(5):
        c = Circle((positions[i,0], positions[i,1]), radius=radii[i], clip_on=False, zorder=10, linewidth=2.5,
               edgecolor='b', facecolor='none')
        ax.add_artist(c)
    print(f"Minimal enclosing radii shape: {radii.shape}")
    print(f"First few radii: {radii[:5]}")
    fig.savefig('enclose_test.png', dpi=300)
