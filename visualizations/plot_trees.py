# visualizations/plot_trees.py
"""
Tree visualization functions.

Provides layouts for trees including disk embeddings (for duality computation),
force-directed, and hierarchical layouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
from typing import Optional, Dict, Tuple, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tree import Tree
from core.binary_tree import BinaryTree


def plot_tree(tree: Tree,
              layout: str = 'disk',
              show_node_labels: bool = False,
              show_edge_lengths: bool = False,
              node_size: int = 300,
              figsize: Tuple[float, float] = (10, 10),
              title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a tree with specified layout.

    Parameters
    ----------
    tree : Tree
        Tree to plot
    layout : str
        Layout algorithm:
        - 'disk': Leaves and root on circle edge (for duality) [DEFAULT]
        - 'radial': Radial from root at bottom
        - 'force': Force-directed with edge lengths
        - 'hierarchical': Layered by depth, root at bottom
    show_node_labels : bool
        Show node IDs (default: False)
    show_edge_lengths : bool
        Show edge lengths as labels (default: False)
    node_size : int
        Size of nodes
    figsize : tuple
        Figure size
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    Examples
    --------
    >>> tree = Tree(n_nodes=5, root=0)
    >>> tree.add_edge(0, 1, length=1.0)
    >>> tree.add_edge(0, 2, length=1.5)
    >>> fig, ax = plot_tree(tree, layout='disk')
    >>> # plt.show()  # User calls explicitly
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute layout
    if layout == 'disk':
        pos = _disk_layout(tree)
        _draw_disk_boundary(ax)
    elif layout == 'radial':
        pos = _radial_layout(tree)
    elif layout == 'force':
        pos = _force_directed_layout(tree)
    elif layout == 'hierarchical':
        pos = _hierarchical_layout(tree)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Draw tree
    _draw_tree(tree, pos, ax, show_node_labels, show_edge_lengths, node_size)

    if title:
        ax.set_title(title, fontsize=14)

    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax


def _disk_layout(tree: Tree) -> Dict[int, Tuple[float, float]]:
    """
    Disk embedding layout: leaves and root on circle boundary.

    Critical for duality computation. Distributes leaves evenly on upper
    half of circle, root on bottom. Internal nodes positioned inside disk.
    Edge lengths adjusted for visual clarity (not proportional).
    """
    pos = {}

    # Identify leaves and root
    leaves = tree.get_leaves()
    root = tree.root
    internal_nodes = [i for i in range(tree.n_nodes)
                      if i not in leaves and i != root]

    n_leaves = len(leaves)

    # Place leaves on circle (upper half, can extend to lower if many)
    if n_leaves == 0:
        # Only root
        pos[root] = (0, -1)
        return pos

    # Determine angular range for leaves
    # If few leaves, keep to upper half. If many, spread around.
    if n_leaves <= 5:
        theta_start = np.pi * 0.1  # 18 degrees from bottom
        theta_end = np.pi * 0.9  # 162 degrees
    else:
        theta_start = 0.0
        theta_end = np.pi * 1.8  # Can wrap to lower half

    # Distribute leaves evenly
    if n_leaves == 1:
        theta_vals = [np.pi / 2]  # Single leaf at top
    else:
        theta_vals = np.linspace(theta_start, theta_end, n_leaves)

    radius = 1.0
    for i, leaf in enumerate(leaves):
        theta = theta_vals[i]
        pos[leaf] = (radius * np.cos(theta), radius * np.sin(theta))

    # Place root at bottom of circle, away from leaves
    pos[root] = (0, -radius)

    # Place internal nodes inside disk using force-directed on tree structure
    if internal_nodes:
        # Use spring layout for internal nodes, constrained to interior
        # Build subgraph of internal nodes + connections to boundary
        G = nx.DiGraph()

        for node in tree.breadth_first_search():
            for child in tree.get_children(node):
                G.add_edge(node, child)

        # Initial positions: internal nodes at origin, boundary nodes fixed
        initial_pos = {node: (0, 0) for node in internal_nodes}
        initial_pos.update({node: pos[node] for node in leaves})
        initial_pos[root] = pos[root]

        # Spring layout with fixed boundary nodes
        fixed_nodes = leaves + [root]
        internal_pos = nx.spring_layout(
            G,
            pos=initial_pos,
            fixed=fixed_nodes,
            k=0.5,
            iterations=50
        )

        # Update internal node positions (constrained to disk interior)
        for node in internal_nodes:
            x, y = internal_pos[node]
            # Ensure inside disk
            dist = np.sqrt(x ** 2 + y ** 2)
            if dist > 0.9:  # Keep away from boundary
                x = x / dist * 0.9
                y = y / dist * 0.9
            pos[node] = (x, y)

    return pos


def _radial_layout(tree: Tree) -> Dict[int, Tuple[float, float]]:
    """
    Radial layout with root at bottom center.

    Arranges nodes in circular layers by depth from root.
    """
    pos = {}

    # Get depth of each node
    depths = tree.get_depth()
    max_depth = int(np.max(depths))

    # Group nodes by depth
    depth_groups = {d: [] for d in range(max_depth + 1)}
    for node in range(tree.n_nodes):
        depth_groups[int(depths[node])].append(node)

    # Place root at bottom
    pos[tree.root] = (0, 0)

    # Place other nodes in circular layers
    for depth in range(1, max_depth + 1):
        nodes_at_depth = depth_groups[depth]
        n = len(nodes_at_depth)

        if n == 0:
            continue

        # Radius increases with depth
        radius = depth

        # Distribute evenly in circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        for i, node in enumerate(nodes_at_depth):
            theta = angles[i]
            x = radius * np.cos(theta)
            y = radius * np.sin(theta) + depth  # Shift up
            pos[node] = (x, y)

    return pos


def _force_directed_layout(tree: Tree) -> Dict[int, Tuple[float, float]]:
    """
    Force-directed layout with edge lengths proportional to weights.

    Uses spring layout with edge lengths as weights.
    """
    # Build NetworkX graph with edge lengths
    G = nx.DiGraph()

    for node in tree.breadth_first_search():
        for child in tree.get_children(node):
            length = tree.get_edge_length(node, child)
            G.add_edge(node, child, weight=length)

    # Spring layout respecting edge lengths
    pos = nx.spring_layout(
        G,
        weight='weight',
        k=1.0,
        iterations=50,
        seed=42
    )

    # Flip vertically to put root at bottom
    max_y = max(y for x, y in pos.values())
    pos = {node: (x, max_y - y) for node, (x, y) in pos.items()}

    return pos


def _hierarchical_layout(tree: Tree) -> Dict[int, Tuple[float, float]]:
    """
    Hierarchical layout: nodes arranged in layers by depth, root at bottom.
    """
    pos = {}

    # Get depth of each node
    depths = tree.get_depth()
    max_depth = int(np.max(depths))

    # Group nodes by depth
    depth_groups = {d: [] for d in range(max_depth + 1)}
    for node in range(tree.n_nodes):
        depth_groups[int(depths[node])].append(node)

    # Place nodes layer by layer
    for depth in range(max_depth + 1):
        nodes_at_depth = depth_groups[depth]
        n = len(nodes_at_depth)

        if n == 0:
            continue

        # Y position increases with depth (root at bottom)
        y = depth

        # X positions evenly distributed
        if n == 1:
            x_positions = [0]
        else:
            x_positions = np.linspace(-n / 2, n / 2, n)

        for i, node in enumerate(nodes_at_depth):
            pos[node] = (x_positions[i], y)

    return pos


def _draw_tree(tree: Tree,
               pos: Dict[int, Tuple[float, float]],
               ax: plt.Axes,
               show_node_labels: bool,
               show_edge_lengths: bool,
               node_size: int):
    """Draw tree on axes with given positions."""

    # Draw edges
    for node in tree.breadth_first_search():
        x1, y1 = pos[node]

        for child in tree.get_children(node):
            x2, y2 = pos[child]

            # Draw edge
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, zorder=1)

            # Show edge length if requested
            if show_edge_lengths:
                length = tree.get_edge_length(node, child)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{length:.2f}',
                        fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='none', alpha=0.7))

    # Draw nodes
    leaves = tree.get_leaves()
    root = tree.root

    for node in range(tree.n_nodes):
        x, y = pos[node]

        # Color code: root=red, leaves=green, internal=blue
        if node == root:
            color = 'red'
            marker = 's'  # Square for root
        elif node in leaves:
            color = 'lightgreen'
            marker = 'o'
        else:
            color = 'lightblue'
            marker = 'o'

        ax.scatter(x, y, s=node_size, c=color, marker=marker,
                   edgecolors='black', linewidths=1.5, zorder=2)

        # Show node labels if requested
        if show_node_labels:
            ax.text(x, y, str(node), fontsize=10, ha='center', va='center',
                    zorder=3)


def _draw_disk_boundary(ax: plt.Axes):
    """Draw circle boundary for disk layout."""
    circle = Circle((0, 0), 1.0, fill=False, edgecolor='gray',
                    linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(circle)


def color_by_horton_strahler(tree: Tree,
                             fig: plt.Figure,
                             ax: plt.Axes,
                             cmap: str = 'viridis') -> Tuple[plt.Figure, plt.Axes]:
    """
    Apply Horton-Strahler order coloring to existing tree plot.

    Parameters
    ----------
    tree : Tree or BinaryTree
        Tree with Horton-Strahler orders computed
    fig : matplotlib.Figure
        Existing figure
    ax : matplotlib.Axes
        Existing axes with tree plot
    cmap : str
        Colormap name

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
        Updated with color coding

    Examples
    --------
    >>> tree = BinaryTree(n_nodes=7, root=0)
    >>> # ... add edges ...
    >>> fig, ax = plot_tree(tree, layout='disk')
    >>> orders = tree.horton_strahler_order()
    >>> fig, ax = color_by_horton_strahler(tree, fig, ax)
    >>> # plt.show()
    """
    # Compute Horton-Strahler orders
    if isinstance(tree, BinaryTree):
        orders = tree.horton_strahler_order()
    else:
        # For general trees, need to implement horton_strahler_order
        # For now, use depth as proxy
        orders = {i: int(tree.get_depth(i)) for i in range(tree.n_nodes)}

    # Get colormap
    max_order = max(orders.values())
    cmap_obj = plt.get_cmap(cmap)

    # Clear and redraw with colors
    # This is a simplified version - full implementation would
    # extract positions from existing plot
    # TBD: More elegant solution

    ax.clear()
    ax.text(0.5, 0.5, 'TBD: Horton-Strahler coloring\nUse plot_tree() with custom colors',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)

    return fig, ax


def add_node_annotation(ax: plt.Axes,
                        node_pos: Tuple[float, float],
                        text: str,
                        **kwargs) -> plt.Axes:
    """
    Add annotation to a specific node.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to annotate
    node_pos : tuple
        (x, y) position of node
    text : str
        Annotation text
    **kwargs
        Passed to ax.annotate()

    Returns
    -------
    ax : matplotlib.Axes

    Examples
    --------
    >>> fig, ax = plot_tree(tree)
    >>> ax = add_node_annotation(ax, (0, 0), "Root node", fontsize=12)
    """
    default_kwargs = {
        'fontsize': 10,
        'bbox': dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
    }
    default_kwargs.update(kwargs)

    ax.annotate(text, xy=node_pos, xytext=(10, 10),
                textcoords='offset points', **default_kwargs)

    return ax


def add_edge_annotation(ax: plt.Axes,
                        pos1: Tuple[float, float],
                        pos2: Tuple[float, float],
                        text: str,
                        **kwargs) -> plt.Axes:
    """
    Add annotation to an edge.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to annotate
    pos1, pos2 : tuple
        Edge endpoints (x, y)
    text : str
        Annotation text
    **kwargs
        Passed to ax.text()

    Returns
    -------
    ax : matplotlib.Axes

    Examples
    --------
    >>> fig, ax = plot_tree(tree)
    >>> ax = add_edge_annotation(ax, (0, 0), (1, 1), "Important edge")
    """
    mid_x = (pos1[0] + pos2[0]) / 2
    mid_y = (pos1[1] + pos2[1]) / 2

    default_kwargs = {
        'fontsize': 9,
        'ha': 'center',
        'bbox': dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='gray', alpha=0.8)
    }
    default_kwargs.update(kwargs)

    ax.text(mid_x, mid_y, text, **default_kwargs)

    return ax