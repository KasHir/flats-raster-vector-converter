# Standard library imports
import os
import sys
import time

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from skimage.morphology import skeletonize
import svgwrite

# Local application/library specific imports
sys.path.insert(0, './external/sknw')
import sknw


def process_image(
        input_file_path: str, output_dir: str,
        )-> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Processes an image by converting it to grayscale, flipping, rotating, thresholding, and skeletonizing.
    Args:
        input_file_path (str): Path to the input image file.
        output_dir (str): Directory to save the processed image.

    Returns:
        tuple: Tuple containing skeletonized image, thresholded image, and original image dimensions.
    """

    # Load the image using OpenCV
    img = cv2.imread(input_file_path)

    if img is None:
        raise ValueError("Image could not be loaded. Please check the file path.")

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flip and ROtate for coordinate extraction
    flipped_gray = cv2.flip(gray, 0)
    rotated_gray = cv2.rotate(flipped_gray, cv2.ROTATE_90_CLOCKWISE)

    # Apply thresholding
    _, thresholded = cv2.threshold(rotated_gray, 192, 255, cv2.THRESH_OTSU)

    # Perform 8-bit format skeletonization
    skeletonized = skeletonize(~(thresholded != 0))
    eight_bit_skeleton = (skeletonized * 255).astype(np.uint8)

    # Save the skeletonized image
    save_path = os.path.join(output_dir, f"skeletonized_{os.path.basename(input_file_path)}")
    cv2.imwrite(save_path, eight_bit_skeleton)

    return eight_bit_skeleton, thresholded, (width, height)


def create_skeltonize_graph(skeletonize_image: np.ndarray) -> nx.MultiGraph:
    """
    Builds a multi-graph from a skeletonized image using the sknw library.

    Args:
        skeletonize_image (np.ndarray): A binary image where the skeletonized structure
                                        is represented by non-zero pixels.

    Returns:
        nx.MultiGraph: A NetworkX multi-graph representation of the skeletonized image.
    """
    return sknw.build_sknw(skeletonize_image.astype(np.uint32),multi=True)
  

"""
Graph Optimization
"""

def combine_edges(graph):
    """
    Combines edges in a NetworkX MultiGraph to reduce the total number of edges and nodes.

    Args:
        graph (nx.MultiGraph): The graph to be processed.

    Returns:
        nx.MultiGraph: The graph with combined edges.
    """
    
    combined_graph = nx.MultiGraph(graph)
        
    def merge_edges(graph, edge1, edge2, reverse_first=False, reverse_second=False):
        """
        Merges two edges in a NetworkX MultiGraph and creates a new edge.

        Args:
            graph (nx.MultiGraph): The graph where edges are merged.
            edge1 (tuple): The first edge to merge.
            edge2 (tuple): The second edge to merge.
            reverse_first (bool): If True, reverse the order of edge1's points.
            reverse_second (bool): If True, reverse the order of edge2's points.

        This function checks if both edges exist in the graph, reverses their order if necessary, 
        merges their points, and adds the new edge to the graph.
        """

        if edge1 not in graph.edges(keys=True) or edge2 not in graph.edges(keys=True):
            raise ValueError(f"Edges do not exist: {edge1}, {edge2}")
        
        pts1 = graph.edges[edge1]['pts']  # edge1's points
        pts2 = graph.edges[edge2]['pts']  # edge2's points

        # Reverse edges's nodes if needed
        def get_ordered_nodes(edge, reverse):
            """Return start and end nodes of an edge, reversed if specified."""
            return (edge[1], edge[0]) if reverse else (edge[0], edge[1])

        start_node_1, end_node_1 = get_ordered_nodes(edge1, reverse_first)
        start_node_2, end_node_2 = get_ordered_nodes(edge2, reverse_second)

        # Reverse the order of edge's points if node direction was swapped
        if not np.array_equal(graph.nodes[start_node_1]['o'], pts1[0]):
            pts1 = np.flip(pts1, axis=0)

        if not np.array_equal(graph.nodes[start_node_2]['o'], pts2[0]):
            pts2 = np.flip(pts2, axis=0)

        # Merge edge's points and reorder them because of undirected graph in sknw
        new_pts = np.concatenate((pts1, pts2), axis=0)
        if start_node_1 > end_node_2:
            new_pts = np.flip(new_pts, axis=0)
            
        # Remove the old two edges and add new merged edge
        graph.remove_edge(*edge1)
        graph.remove_edge(*edge2)

        new_key = graph.new_edge_key(start_node_1, end_node_2)
        graph.add_edge(start_node_1, end_node_2, new_key, pts=new_pts)
        
        print(f"merged:({edge1[0]}, {edge1[1]}) , ({edge2[0]}, {edge2[1]}) -> ({start_node_1}, {end_node_2})")

    
    def create_edge_mappings(graph: nx.MultiGraph) -> (dict, dict):
        """
        Create two dictionaries to map nodes to their outgoing and incoming edges.

        Args:
            graph (nx.MultiGraph): The MultiGraph from which to create edge mappings.

        Returns:
            tuple of two dicts: The first dictionary (outgoing_edges_dict) maps each node to its outgoing edges,
                                and the second dictionary (incoming_edges_dict) maps each node to its incoming edges.
        """
        
        outgoing_edges_dict = {}
        incoming_edges_dict = {}
        for edge in graph.edges(keys=True):
            start_node, end_node = edge[:2]
            outgoing_edges_dict.setdefault(start_node, []).append(edge)
            incoming_edges_dict.setdefault(end_node, []).append(edge)
        
        return outgoing_edges_dict, incoming_edges_dict

    def process_edges(graph) -> bool:
        """
        Processes edges in the graph to find and merge mergeable edges.
        
        Args:
            graph (nx.MultiGraph): The graph to process.
        
        Return:
            bool: True if the edge count decreased after merging, False otherwise.
        """

        def merge_self_loops(graph, outgoing_edges_dict, incoming_edges_dict):
            """
            Merges self-loop edges in a graph.

            Args:
                graph (nx.MultiGraph): The graph to process.
                outgoing_edges_dict (dict): Dictionary mapping nodes to their outgoing edges.
                incoming_edges_dict (dict): Dictionary mapping nodes to their incoming edges.

            Returns:
                bool: True if any changes were made, False otherwise.
            """

            changes_made = False

            for node, edges_list in outgoing_edges_dict.items():
                for edge1 in edges_list:
                    start_edge1, end_edge1 = edge1[:2]  # start and end nodes of edge1

                    # Check for self-loop
                    if start_edge1 == end_edge1:

                        # Generate a list of candidate edges to merge with
                        candidate_edges = (outgoing_edges_dict[node] + incoming_edges_dict.get(node, []))

                        for edge2 in candidate_edges:
                            start_edge2, end_edge2 = edge2[:2]

                            if edge1 == edge2:
                                continue  # Skip if it's the same edge

                            # Merge logic for self-loop
                            if end_edge1 == start_edge2:
                                merge_edges(graph, edge1, edge2)
                                changes_made = True
                                return changes_made
                            elif end_edge1 == end_edge2:
                                merge_edges(graph, edge1, edge2, reverse_second=True)
                                changes_made = True
                                return changes_made

            return changes_made

        def merge_non_self_loop_edges(graph, outgoing_edges_dict, incoming_edges_dict):
            """
            Merges non-self-loop edges in the graph.

            Args:
                graph (nx.MultiGraph): The graph to process.
                outgoing_edges_dict (dict): Dictionary mapping nodes to their outgoing edges.
                incoming_edges_dict (dict): Dictionary mapping nodes to their incoming edges.

            Returns:
                bool: True if any changes were made, False otherwise.
            """

            changes_made = False

            for node, edges_list in outgoing_edges_dict.items():
                for edge1 in edges_list:
                    start_edge1, end_edge1 = edge1[:2]

                    # Generate a list of candidate edges to merge with
                    candidate_edges = (outgoing_edges_dict[node] +
                                    incoming_edges_dict.get(node, []) +
                                    outgoing_edges_dict.get(end_edge1, []) +
                                    incoming_edges_dict.get(end_edge1, []))

                    for edge2 in candidate_edges:
                        if edge1 == edge2:
                            continue

                        start_edge2, end_edge2 = edge2[:2]

                        # Determine how edges should be merged based on their start and end nodes
                        if end_edge1 == start_edge2:
                            merge_edges(graph, edge1, edge2)
                            changes_made = True
                            return changes_made
                        elif end_edge1 == end_edge2:
                            merge_edges(graph, edge1, edge2, reverse_second=True)
                            changes_made = True
                            return changes_made
                        elif start_edge1 == start_edge2:
                            merge_edges(graph, edge1, edge2, reverse_first=True)
                            changes_made = True
                            return changes_made
                        elif start_edge1 == end_edge2:
                            merge_edges(graph, edge1, edge2, reverse_first=True, reverse_second=True)
                            changes_made = True
                            return changes_made

            return changes_made

        # Create edge's node dictionaries to improve serch performance
        outgoing_edges_dict, incoming_edges_dict = create_edge_mappings(graph)

        initial_edges_count = len(graph.edges())
        changes_made = True
        
        while changes_made:
            changes_made = False

            # Search and merge self-loop edges to avoid creating isolated self-loops
            if merge_self_loops(graph, outgoing_edges_dict, incoming_edges_dict):
                changes_made = True
                outgoing_edges_dict, incoming_edges_dict = create_edge_mappings(graph)
                continue
            
            # Search and merge non self-loops if all self-loops are merged
            if not changes_made:
                if not changes_made:
                    changes_made = merge_non_self_loop_edges(graph, outgoing_edges_dict, incoming_edges_dict)

            # Update edge's node dictionaries when the graph is changed
            if changes_made:
                outgoing_edges_dict, incoming_edges_dict = create_edge_mappings(graph)
        
        return len(graph.edges()) < initial_edges_count
                
    if process_edges(combined_graph):
        print("Edges were optimized!")
    else:
        print("No edge merging was possible.")

    return combined_graph


def remove_isolates(graph: nx.MultiGraph) -> nx.MultiGraph:
    remove_isolates_graph = nx.MultiGraph(graph)
    isolates = list(nx.isolates(remove_isolates_graph))
    remove_isolates_graph.remove_nodes_from(isolates)
    return remove_isolates_graph


def remap_node_ids(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Relabels the node IDs of a MultiGraph to be a sequence of integers starting at 0.
    """
    mapping = {old_id: new_id for new_id, old_id in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping)


"""
Path Optimization by OR-Tools TSP solver
"""

def create_distance_matrix(graph: nx.MultiGraph) -> dict[int, dict[int, int]]:
    """
    Creates a distance matrix for nodes, primarily for use with the OR-Tools TSP solver.

    Args:
        graph (nx.MultiGraph): The graph to compute distances for.

    Returns:
        dict[int, dict[int, int]]: A dictionary where each node ID maps to another dictionary, 
                                   containing distances to other nodes. Distances are represented as integers.
    """
    node_positions = {n: graph.nodes[n]['o'].tolist() for n in graph.nodes}
    distance_matrix = {}
    
    # Calculate the Euclidean distance between each pair of nodes
    for i, pos_i in node_positions.items():
        distance_matrix[i] = {
            j: int(np.linalg.norm(np.array(pos_i) - np.array(pos_j)))
            for j, pos_j in node_positions.items() if i != j
        }
        distance_matrix[i][i] = 0

    return distance_matrix


def calculate_optimized_route(
        distance_matrix: dict[int, dict[int, int]], graph: nx.MultiGraph, verbose: bool = False
        ) -> list[int]:
    """
    Calculates the optimized route for a given graph using the OR-Tools TSP solver.

    Args:
        distance_matrix (dict[int, dict[int, int]]): A dictionary of dictionaries containing the distances between nodes.
        graph (nx.MultiGraph): The graph representing the nodes and their connections.
        verbose (bool): If True, additional details will be printed to the console.

    Returns:
        list[int]: A list of node IDs representing the optimized route.
    """
    start_node = list(graph.nodes())[0]
    num_nodes = len(distance_matrix)
    if verbose:
        print(f"start_node: {start_node}")
        print(f"num_nodes: {num_nodes}")
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, start_node)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        distance = distance_matrix[from_node][to_node]
        return distance

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Apply edge constraints: start and end nodes of the edges are fixed as routes
    for (s, e) in graph.edges():
        if s != e:
            routing.solver().Add(
                routing.NextVar(manager.NodeToIndex(s)) == manager.NodeToIndex(e))
          
    # Setting first solution strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve and handle potential errors
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise Exception("No solution found by the solver.")

    # Get the optimized route as list and print the solution
    index = routing.Start(0)
    total_distance = 0
    route = [start_node]
    plan_output = "Route for vehicle 0:\n"

    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance = routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        total_distance += route_distance

    plan_output += f" {manager.IndexToNode(index)}\nTotal distance: {total_distance} pixels\n"
    print(plan_output)

    return route

  
"""
Export PNG file function
"""
def export_graph_png(graph: nx.MultiGraph, colors: list[str],
                save_dir: str, img_size: tuple[int, int], save_name: str):
    """
    Draws a graph and saves it as a PNG file.

    Args:
        graph (nx.MultiGraph): The graph to be drawn.
        colors (list[str]): A list of colors for drawing different edges.
        save_dir (str): Directory where the image will be saved.
        img_size (tuple[int, int]): Size of the image (width, height).
        save_name (str): Name of the saved file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # Adjust the graph drawing size to match the aspect ratio of the original image
    aspect_ratio = img_size[0] / img_size[1]
    fig_size = (10.0 * aspect_ratio, 10.0)
    plt.rcParams['figure.figsize'] = fig_size
    
    # Draw edges
    for color_index, (s, e) in enumerate(graph.edges()):
        current_color = colors[color_index % len(colors)]
        
        for edge_data in graph[s][e].values():
            pts = np.array(edge_data['pts'])
            plt.plot(pts[:, 0], pts[:, 1], current_color)

    # Save the graph to a file
    plt.savefig(os.path.join(save_dir, f"{save_name}_graph.png"))
    plt.close()


"""
Export SVG file functions
"""
def draw_edge(dwg: svgwrite.Drawing, graph: nx.MultiGraph, 
              start_node: int, end_node: int, 
              stroke_color: str, stroke_width: float = 0.8) -> None:
    """
    Draws an edge between two nodes on the SVG drawing.
    
    This function iterates over all the data associated with the edge between the specified
    start and end nodes, extracts the polyline points, and adds them to the SVG drawing.
    """
    for edge_data in graph[start_node][end_node].values():
        polyline_points = edge_data['pts'].tolist()
        dwg.add(dwg.polyline(
            points=polyline_points,
            stroke=stroke_color,
            stroke_width=stroke_width,
            fill='none'
        ))

def export_graph_svg(graph: nx.MultiGraph, colors: list[str], 
                     save_dir: str, save_name: str) -> None:
    """
    Exports an SVG file of the graph without a specific route.

    Args:
        graph (nx.MultiGraph): The graph to be exported.
        colors (list[str]): A list of colors used for different edges in the graph.
        save_dir (str): The directory where the SVG file will be saved.
        save_name (str): The name of the SVG file to be saved.

    This function iterates through each edge in the graph and draws it using the specified colors.
    It saves the resulting SVG to the specified directory.
    """

    outfile_name = os.path.join(save_dir, f"{save_name}.svg")
    dwg = svgwrite.Drawing(outfile_name, profile='tiny')

    for color_index, (s, e) in enumerate(graph.edges()):
        current_color = colors[color_index % len(colors)]
        draw_edge(dwg, graph, s, e, current_color)

    try:
        dwg.save()
        print('Saved:', outfile_name)
    except Exception as e:
        print('Error saving file:', e)
    finally:
        del dwg

def export_routed_graph_svg(graph: nx.MultiGraph, route: list[int], 
                            file_name: str, include_invalid_path: bool = False) -> None:
    """
    Exports an SVG file representing a routed graph.

    Args:
        graph (nx.MultiGraph): The graph from which the SVG is generated.
        route (list[int]): A list of node IDs representing the route to be highlighted.
        file_name (str): The name of the SVG file to be saved.
        include_invalid_path (bool, optional): If True, invalid paths (paths not present in the graph) are drawn. Defaults to False.

    This function draws each edge in the route on the SVG. If `include_invalid_path` is True, 
    it also draws lines between non-sequential nodes in the route, marking these as invalid paths.
    """

    def draw_invalid_path(start_node, end_node):
        pt_s = graph.nodes[start_node]['o'].tolist()
        pt_e = graph.nodes[end_node]['o'].tolist()
        dwg.add(dwg.polyline(
            points=[pt_s, pt_e],
            stroke='gray',
            stroke_width=0.5,
            fill='none'
        ))

    dwg = svgwrite.Drawing(file_name, profile='tiny')
    last_end_node = route[0]
    
    for i in range(len(route) - 1):
        start_node, end_node = route[i], route[i + 1]

        # Draw self-loop edges
        if (start_node, start_node) in graph.edges():
            draw_edge(dwg, graph, start_node, start_node, 'red')
            last_end_node = start_node

        # Draw the other edges
        if (start_node, end_node) in graph.edges():
            draw_edge(dwg, graph, start_node, end_node, 'black')
            last_end_node = end_node

        # Draw path between edge and the next edge
        if include_invalid_path and last_end_node != end_node:
            draw_invalid_path(last_end_node, end_node)

    try:
        dwg.save()
        print(f"saved: {file_name}")
    except Exception as e:
        print('Error saving file:', e)
    finally:
        del dwg



"""
Functions for using the Main()
"""
def show_graph_info(graph):
    isolates = nx.number_of_isolates(graph)
    print(f"Number of isolated nodes: {isolates}")
    print(f"Graph information: {graph}")


def optimize_graph(graph, output_dir, image_name, img_size, colors):
    """
    To optimizes the graph by combining similar edges, 
    which simplifies the graph structure and improves processing efficiency.
    """
    print("Optimizing graph starts.")
    combined_graph = combine_edges(graph)
    export_graph_png(combined_graph, colors, output_dir, img_size, image_name + '_combined')
    export_graph_svg(combined_graph, colors, output_dir, image_name + '_combined')
    export_graph_svg(combined_graph, ['black'], output_dir, image_name + '_combined_black')
    
    print("Graph info Before optimization")
    show_graph_info(graph)

    graph_removed_isolates = remove_isolates(combined_graph)
    print("Graph info After optimization")
    show_graph_info(graph_removed_isolates)
    return remap_node_ids(graph_removed_isolates)


def optimize_path(graph, output_dir, image_name):
    """
    To solve the Traveling Salesman Problem (TSP) to find the most efficient path.
    It includes benchmarking to measure the performance of the optimization process.
    """
    print("Optimizing path (Solving TSP) starts.")
    start_time = time.time()
    distance_matrix = create_distance_matrix(graph)
    print(f"Create distance_matrix took {time.time() - start_time} seconds.")
    start_time = time.time()
    route = calculate_optimized_route(distance_matrix, graph)
    print(f"TSP solving took {time.time() - start_time} seconds.")
    export_routed_graph_svg(graph, route, f'{output_dir}/{image_name}_optimized_path.svg')
    export_routed_graph_svg(graph, route, f'{output_dir}/{image_name}_optimized_path_invalid.svg', include_invalid_path=True)


def main(input_file_path=None, output_dir=None):
    print("flats-raster-vector-converter is running")

    # Configurable parameters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'black']

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Image processing
    processed_image, data, img_size = process_image(input_file_path, output_dir)
    graph = create_skeltonize_graph(processed_image)
    image_name = os.path.splitext(os.path.basename(input_file_path))[0]  # Extract image name from path
    export_graph_png(graph, colors, output_dir, img_size, image_name + '_raw')

    # Graph optimization
    optimized_graph = optimize_graph(graph, output_dir, image_name, img_size, colors)

    # Path optimization
    optimize_path(optimized_graph, output_dir, image_name)

    print("Completed")


if __name__ == "__main__":
    # Default file paths used during testing or if no arguments are provided
    default_input_file_path = 'img_line/zunda.png'
    default_output_dir = 'out/'

    # Check if command-line arguments are provided
    if len(sys.argv) == 3:
        input_file_path = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        input_file_path = default_input_file_path
        output_dir = default_output_dir
        print(f"No command-line arguments provided. Using default paths: {input_file_path}, {output_dir}")

    main(input_file_path, output_dir)