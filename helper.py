import numpy as np


def _grayscale_values():
    return [[v, v, v] for v in range(256)]


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.min(dist_2)


grayscale_values = _grayscale_values()[:30] + _grayscale_values()[230:]

if __name__ == '__main__':
    print(_grayscale_values())
