from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt

# This module provides a visualization utility for Connect 4 game trees.

def _collect_nodes(root_node, max_depth: int, max_nodes: int) -> Tuple[List[Tuple[int, object, int]], List[Tuple[int, int]]]:
    nodes: List[Tuple[int, object, int]] = []
    edges: List[Tuple[int, int]] = []

    queue: List[Tuple[object, int, int]] = [(root_node, 0, None)]
    next_id = 0
    index = 0
    while index < len(queue) and len(nodes) < max_nodes: # Should use stack but queue is better for visualization purposes
        node, depth, parent_id = queue[index]
        index += 1
        node_id = next_id
        next_id += 1
        nodes.append((node_id, node, depth))
        if parent_id is not None:
            edges.append((parent_id, node_id))
        if depth >= max_depth:
            continue
        for child in getattr(node, "children", []):
            if len(nodes) + (len(queue) - index) >= max_nodes:
                break
            queue.append((child, depth + 1, node_id))

    return nodes, edges


def render_tree(root_node, max_depth: int = 4, max_nodes: int = 600):
    # Render the minimax/expectiminimax tree using matplotlib

    if root_node is None:
        print("No search tree available yet. Let the AI play a move first.")
        return

    nodes, edges = _collect_nodes(root_node, max_depth, max_nodes)
    if not nodes:
        print("Tree could not be generated (no nodes collected).")
        return

    depth_levels = defaultdict(list)
    for node_id, node, depth in nodes:
        depth_levels[depth].append(node_id)

    positions = {}
    for depth, node_ids in depth_levels.items():
        count = len(node_ids)
        for idx, node_id in enumerate(node_ids):
            x = (idx + 1) / (count + 1)
            y = -depth
            positions[node_id] = (x, y)

    plt.figure(figsize=(14, 4 + 1.5 * max(1, max(depth_levels.keys()))))

    for parent_id, child_id in edges:
        if parent_id not in positions or child_id not in positions:
            continue
        x_values = [positions[parent_id][0], positions[child_id][0]]
        y_values = [positions[parent_id][1], positions[child_id][1]]
        plt.plot(x_values, y_values, color="#888888", linewidth=1)

    # Track whose turn it is at each decision depth, toggling only when
    # we move to a *new* depth level of decision nodes. Chance levels
    # do not toggle the player; they sit between decision layers.
    depth_to_player = {}
    # Depth 0 is always the AI (MAX) since the AI is deciding at the root.
    depth_to_player[0] = "AI"

    for node_id, node, depth in nodes:
        x, y = positions[node_id]
        node_type = getattr(node, "node_type", "decision")

        max_color = "#1f77b4"   # blue for MAX (AI)
        min_color = "#d62728"   # red for MIN (human)
        chance_color = "#ff7f0e"  # orange for chance

        if node_type == "chance":
            color = chance_color
        else:
            # Decision node: determine whose turn it is at this depth.
            if depth not in depth_to_player:
                # Find nearest shallower decision depth to inherit and then toggle.
                parent_depths = [d for d in depth_to_player.keys() if d < depth]
                if parent_depths:
                    last_depth = max(parent_depths)
                    last_player = depth_to_player[last_depth]
                    depth_to_player[depth] = "HUMAN" if last_player == "AI" else "AI"
                else:
                    depth_to_player[depth] = "AI"

            player = depth_to_player[depth]
            color = max_color if player == "AI" else min_color
        plt.scatter(x, y, s=300, color=color, edgecolors="black", zorder=5)
        move = getattr(node, "move", None)
        heuristic = getattr(node, "heuristic_value", None)
        probability = getattr(node, "probability", None)
        label_lines = [f"Move: {move}", f"Type: {node_type}"]
        if heuristic is not None:
            label_lines.append(f"H: {round(heuristic, 2)}")
        if probability is not None and (node_type == "chance" or abs(probability - 1.0) > 1e-9):
            label_lines.append(f"P: {round(probability, 2)}")
        plt.text(x, y + 0.05, "\n".join(label_lines), ha="center", va="bottom", fontsize=8)

    plt.title("Connect 4 Search Tree")
    plt.axis("off")
    plt.tight_layout()
    plt.show()