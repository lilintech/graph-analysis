import pandas as pd
import heapq
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def load_tasks(file_path):
    """Load tasks from CSV file"""
    df = pd.read_csv(file_path)
    tasks = [(row['Task'], row['Start'], row['End']) for _, row in df.iterrows()]
    return tasks

def build_interval_graph_efficient(tasks):
    """Build interval graph more efficiently using interval tree concept"""
    G = defaultdict(list)
    intervals = {task: (start, end) for task, start, end in tasks}
    
    # Sort by start time for efficient overlap checking
    sorted_tasks = sorted(tasks, key=lambda x: x[1])
    
    for i, (t1, s1, e1) in enumerate(sorted_tasks):
        for j in range(i + 1, len(sorted_tasks)):
            t2, s2, e2 = sorted_tasks[j]
            if s2 >= e1:  # No overlap and all subsequent tasks won't overlap either
                break
            if not (e1 <= s2 or e2 <= s1):  # Overlap condition
                G[t1].append(t2)
                G[t2].append(t1)
    
    return dict(G), intervals

def convert_to_networkx_graph(graph_dict):
    """Convert our graph dictionary to a NetworkX graph object"""
    G_nx = nx.Graph()
    for node, neighbors in graph_dict.items():
        G_nx.add_node(node)
        for neighbor in neighbors:
            G_nx.add_edge(node, neighbor)
    return G_nx

def optimal_interval_coloring(tasks):
    """
    GREEDY ALGORITHM 1: Optimal coloring for interval graphs
    Uses greedy approach with min-heap for server assignment
    Time Complexity: O(n log n)
    """
    # Create events: (time, type, task_id)
    events = []
    for task, start, end in tasks:
        events.append((start, 1, task))   # Start event
        events.append((end, -1, task))    # End event
    
    # Sort events: by time, end events before start events at same time
    events.sort(key=lambda x: (x[0], x[1]))
    
    available_servers = []  # Min-heap of available servers (GREEDY: always pick smallest)
    next_server = 0
    task_to_server = {}
    
    for time, event_type, task in events:
        if event_type == 1:  # Start event
            # GREEDY CHOICE: Take smallest available server, or create new one
            if available_servers:
                server = heapq.heappop(available_servers)
            else:
                server = next_server
                next_server += 1
            task_to_server[task] = server
        else:  # End event
            server = task_to_server[task]
            heapq.heappush(available_servers, server)  # Server becomes available
    
    return task_to_server, next_server

def dfs_traversal(graph, start_node=None):
    """DFS traversal for graph analysis"""
    visited = set()
    order = []
    
    def dfs(node):
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
    
    nodes = [start_node] if start_node else list(graph.keys())
    for node in nodes:
        if node not in visited:
            dfs(node)
    
    return order

def maximum_cardinality_search_efficient(graph):
    """More efficient MCS implementation to find Perfect Elimination Ordering"""
    if not graph:
        return []
    
    weights = {node: 0 for node in graph}
    visited = set()
    order = []
    
    # Use a max-heap for efficient maximum weight retrieval
    heap = [(-weight, node) for node, weight in weights.items()]
    heapq.heapify(heap)
    
    while heap:
        # GREEDY CHOICE: Pick node with maximum weight
        _, node = heapq.heappop(heap)
        if node in visited:
            continue
            
        visited.add(node)
        order.append(node)
        
        # Update weights of neighbors
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                weights[neighbor] += 1
                heapq.heappush(heap, (-weights[neighbor], neighbor))
    
    return order

def greedy_coloring_with_peo(graph, peo):
    """
    GREEDY ALGORITHM 2: Classic greedy coloring with Perfect Elimination Ordering
    Guaranteed optimal for chordal graphs (which interval graphs are)
    """
    coloring = {}
    # Process nodes in reverse PEO (this is key for optimality)
    for node in reversed(peo):
        used_colors = set()
        # Check colors of already-colored neighbors
        for neighbor in graph.get(node, []):
            if neighbor in coloring:
                used_colors.add(coloring[neighbor])
        
        # GREEDY CHOICE: Assign smallest available color
        color = 0
        while color in used_colors:
            color += 1
        coloring[node] = color
    
    return coloring

def plot_schedule(tasks, coloring, title, save_path=None):
    """Plot the schedule as Gantt chart"""
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort tasks by start time for better visualization
    sorted_tasks = sorted(tasks, key=lambda x: x[1])
    
    for task, start, end in sorted_tasks:
        server = coloring[task]
        ax.barh(server, end - start, left=start, height=0.6,
                color=colors[server % len(colors)], edgecolor='black', alpha=0.8)
        ax.text(start + (end - start)/2, server, task,
                va='center', ha='center', color='white', fontweight='bold', fontsize=9)
    
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Server", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show all servers
    max_server = max(coloring.values()) if coloring else 0
    ax.set_yticks(range(max_server + 1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_graph(graph_dict, intervals, coloring, title, save_path=None):
    """Plot the interval graph using NetworkX"""
    if not graph_dict:
        print(f"No graph to plot for {title}")
        return
        
    try:
        # Convert our graph dictionary to NetworkX graph
        G_nx = convert_to_networkx_graph(graph_dict)
        
        plt.figure(figsize=(10, 8))
        
        # Create layout based on intervals for better visualization
        pos = {}
        for task in graph_dict.keys():
            start, end = intervals[task]
            # Position node horizontally based on midpoint, vertically based on server
            pos[task] = ((start + end) / 2, coloring.get(task, 0))
        
        # Extract colors for nodes
        node_colors = [coloring[node] for node in G_nx.nodes()]
        
        nx.draw(G_nx, pos, 
                node_color=node_colors, 
                cmap=plt.cm.tab20,
                with_labels=True, 
                node_size=500, 
                font_size=8,
                font_weight='bold',
                edgecolors='black',
                linewidths=0.5)
        
        plt.title(title, fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting graph for {title}: {e}")

def run_comparison_analysis(file_path):
    """Run both greedy methods and compare results"""
    tasks = load_tasks(file_path)
    
    print(f"\n=== Analyzing {os.path.basename(file_path)} ===")
    print(f"Number of tasks: {len(tasks)}")
    
    # GREEDY METHOD 1: Optimal interval coloring (most efficient)
    coloring_optimal, servers_optimal = optimal_interval_coloring(tasks)
    
    # GREEDY METHOD 2: Graph-based approach with PEO coloring
    graph_dict, intervals = build_interval_graph_efficient(tasks)
    
    if graph_dict:  # Only run graph methods if graph is non-empty
        dfs_order = dfs_traversal(graph_dict)
        print(f"DFS traversal order: {dfs_order}")
        
        peo = maximum_cardinality_search_efficient(graph_dict)
        coloring_graph = greedy_coloring_with_peo(graph_dict, peo)
        servers_graph = max(coloring_graph.values()) + 1 if coloring_graph else 0
    else:
        coloring_graph = coloring_optimal
        servers_graph = servers_optimal
    
    print(f"Greedy Interval Method: {servers_optimal} servers")
    print(f"Greedy Graph Coloring: {servers_graph} servers")
    
    # Verify both greedy methods give same result
    if servers_optimal == servers_graph:
        print("✓ Both greedy methods agree on optimal solution")
    else:
        print("⚠ Greedy methods disagree - using optimal interval method")
    
    return coloring_optimal, servers_optimal, tasks, graph_dict, intervals

def run_all_tests(base_path):
    """Run tests on all CSV files"""
    file_names = [
        "task_intervals_1.csv", "task_intervals_5.csv", "task_intervals_10.csv",
        "task_intervals_15.csv", "task_intervals_20.csv", "task_intervals_25.csv", 
        "task_intervals_30.csv"
    ]
    
    results = {}
    
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_name} not found, skipping...")
            continue
            
        try:
            coloring, servers, tasks, graph_dict, intervals = run_comparison_analysis(file_path)
            results[file_name] = servers
            
            print(f"\nServer assignment for {file_name}:")
            for task in sorted(coloring.keys()):
                print(f"  {task}: Server {coloring[task]}")
            
            # Create visualizations
            plot_schedule(tasks, coloring,
                         title=f"Optimal Schedule - {file_name}",
                         save_path=f"{file_name.replace('.csv','_schedule.png')}")
            
            if graph_dict:  # Only plot graph if it exists
                plot_graph(graph_dict, intervals, coloring,
                          title=f"Interval Graph - {file_name}",
                          save_path=f"{file_name.replace('.csv','_graph.png')}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    return results

# Import networkx only if needed for plotting
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("NetworkX not available. Graph plotting disabled.")
    NETWORKX_AVAILABLE = False
    # Create a dummy function if NetworkX is not available
    def convert_to_networkx_graph(graph_dict):
        return None
    def plot_graph(graph_dict, intervals, coloring, title, save_path=None):
        print(f"NetworkX not available - cannot plot graph for {title}")

if __name__ == "__main__":
    base_path = "."  # current directory
    
    print("=" * 60)
    print("GREEDY ALGORITHMS FOR OPTIMAL TASK SCHEDULING")
    print("=" * 60)
    print("Two greedy approaches being compared:")
    print("1. Greedy Interval Coloring (with min-heap)")
    print("2. Greedy Graph Coloring (with Perfect Elimination Ordering)")
    print("=" * 60)
    
    if not NETWORKX_AVAILABLE:
        print("Note: NetworkX not available - graph visualization disabled")
    
    results = run_all_tests(base_path)
    
    print("\nSUMMARY OF RESULTS")
    for file, servers in sorted(results.items()):
        print(f"{file}: {servers} servers")
    
    total_servers = sum(results.values())
    print(f"\nTotal servers across all test cases: {total_servers}")
    print("All tasks scheduled optimally using greedy algorithms! ✓")