from skcriteria.utils.dag_rank import *

# =============================================================================
# TESTING
# =============================================================================


def _ejemplo():
    import pandas as pd

    # Crear la matriz de adyacencia como DataFrame
    # Basándome en el diagrama que subiste
    adj_matrix = pd.DataFrame(
        {
            "A": [0, 0, 1, 0, 1],
            "B": [0, 0, 0, 0, 0],
            "C": [1, 1, 0, 0, 1],
            "D": [0, 0, 0, 0, 0],
            "E": [0, 0, 1, 1, 0],
        },
        index=["A", "B", "C", "D", "E"],
    )

    # Crear el grafo dirigido desde el DataFrame
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    return graph


def main():
    """Main function to test DAG conversion and ranking generation."""
    import sys
    import joblib
    import numpy as np

    # Load graph from file or use example
    if len(sys.argv) > 1:
        # Load from command line argument
        graph_file = sys.argv[1]
        print(f"Loading graph from: {graph_file}")
        g = joblib.load(graph_file)
    else:
        # Use example graph
        print("No file provided, using example graph")
        g = _ejemplo()

    # Convert to DAG
    dag, fas, method = as_dag(g)

    # Get all possible rankings
    alternatives = np.asarray(dag.nodes)
    rankings = all_rankings(alternatives, dag)

    # Display results
    print("=" * 70)
    print("GRAPH TO DAG CONVERSION")
    print("=" * 70)
    print(f"Original graph nodes: {g.number_of_nodes()}")
    print(f"Original graph edges: {g.number_of_edges()}")
    print(f"Method used: {method}")
    print(f"Feedback arc set size: {len(fas)}")
    print(f"Edges removed: {fas}")
    print(f"DAG nodes: {list(dag.nodes())}")
    print(f"DAG edges: {dag.number_of_edges()}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(dag)}")

    print("\n" + "=" * 70)
    print("TOPOLOGICAL SORTS")
    print("=" * 70)
    print(f"Alternatives: {alternatives}")
    print(f"Total rankings: {len(rankings)}")
    print(f"\nFirst 5 rankings:")
    for i, ranking in enumerate(rankings[:5]):
        print(f"  {i+1}. {ranking}")
    if len(rankings) > 5:
        print(f"  ... ({len(rankings) - 5} more)")

    # Draw graphs side by side
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Draw original graph
    pos1 = nx.spring_layout(g, seed=42)
    nx.draw(
        g,
        pos1,
        ax=ax1,
        with_labels=True,
        node_color="lightblue",
        node_size=1000,
        font_size=12,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
        connectionstyle="arc3,rad=0.1",  # Curved arrows to avoid double-headed appearance
        arrowstyle="-|>",
    )
    ax1.set_title(
        f"Original Graph\n(Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}, Is DAG: {nx.is_directed_acyclic_graph(g)})",
        fontsize=12,
        fontweight="bold",
    )

    # Draw DAG
    pos2 = nx.spring_layout(dag, seed=42)
    nx.draw(
        dag,
        pos2,
        ax=ax2,
        with_labels=True,
        node_color="lightgreen",
        node_size=1000,
        font_size=12,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
        connectionstyle="arc3,rad=0.1",  # Curved arrows to avoid double-headed appearance
        arrowstyle="-|>",
    )
    ax2.set_title(
        f"DAG (Method: {method})\n(Nodes: {dag.number_of_nodes()}, Edges: {dag.number_of_edges()}, Removed: {len(fas)} edges)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
