import os
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import svgwrite
from skimage.morphology import skeletonize

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ローカルライブラリのインポート
sys.path.insert(0, './external/sknw')
import sknw

import time
#import cProfile

def process_image(file_path, save_dir):
    # 画像をOpenCVで読み込む
    img = cv2.imread(file_path)

    # 画像サイズを取得 (幅と高さ)
    height, width = img.shape[:2]

    # グレースケールに変換する
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 上下反転（座標抽出する際に反転するため）
    gray_flip = cv2.flip(gray, 0)

    # 90度回転（座標抽出時に回転するため）
    gray_rotate = cv2.rotate(gray_flip, cv2.ROTATE_90_CLOCKWISE)

    # Thresholding
    retval, dst = cv2.threshold(gray_rotate, 192, 255, cv2.THRESH_OTSU)

    # Skeletonization
    ske = skeletonize(~(dst != 0))

    # Convert the skeleton to 8-bit (0 or 255)
    ske_8bit = (ske * 255).astype(np.uint8)

    # Save the skeletonized image
    cv2.imwrite(os.path.join(save_dir, 'skeletonized_' + os.path.basename(file_path)), ske_8bit)

    # ここで、処理された画像や他の重要なデータを返す
    return ske_8bit, dst, (width, height)


def create_skeltonize_graph(skeletonize_image):
    
    return sknw.build_sknw(skeletonize_image.astype(np.uint32),multi=True)
  

"""
Graph Optimization
"""

def combine_edges(graph):
    combined_graph = nx.MultiGraph(graph)  # マルチグラフのコピーを作成
        
    def merge_edges(graph, edge1, edge2, reverse_first=False, reverse_second=False):
        """
        2つのエッジを結合し、新しいエッジを作成する関数。
        必要に応じてエッジの順序を逆転させる。
        """
        if edge1 in graph.edges(keys=True) and edge2 in graph.edges(keys=True):# エッジがグラフに存在するか確認
            #print(f"merge: {edge1}, {edge2}")
            pts1 = graph.edges[edge1]['pts']  # edge1の座標
            pts2 = graph.edges[edge2]['pts']  # edge2の座標
    
            # エッジの順序を反転（必要に応じて）
            if reverse_first:
                pts1 = np.flip(pts1, axis=0)
                start_node_1, end_node_1 = edge1[1], edge1[0]
                #print("reverse_1st-----------------------")
            else:
                start_node_1, end_node_1 = edge1[:2]
    
            if reverse_second:
                pts2 = np.flip(pts2, axis=0)
                start_node_2, end_node_2 = edge2[1], edge2[0]
                #print("reverse_2nd")
            else:
                start_node_2, end_node_2 = edge2[:2]
    
            # direction check
            print("direction check")
            if not np.array_equal(graph.nodes[start_node_1]['o'], pts1[0]):
                print("edge1 direction NG")
                pts1 = np.flip(pts1, axis=0)

                if np.array_equal(graph.nodes[start_node_1]['o'], pts1[0]):
                    print("edge1 direction OK")

            if not np.array_equal(graph.nodes[start_node_2]['o'], pts2[0]):
                print("edge2 direction NG")
                pts2 = np.flip(pts2, axis=0)

                if np.array_equal(graph.nodes[start_node_2]['o'], pts2[0]):
                    print("edge2 direction OK")


            # 新しいエッジの座標を計算
            merged_coordinate = graph.nodes[end_node_1]['o']
            new_pts = np.vstack((pts1, merged_coordinate, pts2))
    

            # 新しいエッジの順序が逆の場合、座標を反転
            if start_node_1 > end_node_2:
                new_pts = np.flip(new_pts, axis=0)
                #print('pts reverse')
                
            # 元のエッジを削除し、新しいエッジを追加
            graph.remove_edge(edge1[0], edge1[1], key=edge1[2])
            graph.remove_edge(edge2[0], edge2[1], key=edge2[2])
            new_key = graph.new_edge_key(start_node_1, end_node_2)
            #graph.add_edge(start_node_1, end_node_2, key=edge1[2], pts=new_pts)
            graph.add_edge(start_node_1, end_node_2, new_key, pts=new_pts)
            print(f"merged:({edge1[0]}, {edge1[1]}) , ({edge2[0]}, {edge2[1]}) -> {start_node_1}, {end_node_2}")

            #DEBUG_NO = str(edge1[0]) +','+ str(edge1[1]) +','+ str(edge2[0]) + ','+ str(edge2[1])
            #draw_graph(graph, colors, save_dir2, img_size, save_name=filename+DEBUG_NO)
        else:
             print(f"エッジが存在しません: {edge1}, {edge2}")

    # 各ノードから出発するエッジを格納する辞書を作成
    def create_edges_reverse_dict(graph):
        edges_dict = {}
        edges_reverse_dict = {}
        for edge in graph.edges(keys=True):
            start_node, end_node = edge[:2]
            edges_dict.setdefault(start_node, []).append(edge)
            edges_reverse_dict.setdefault(end_node, []).append(edge)

        return edges_dict, edges_reverse_dict
                
    def process_edges(graph):
        #start_time = time.time() # for benchmark
        
        changes_made = True
        # 全エッジのリストを取得
        edges_dict, edges_reverse_dict = create_edges_reverse_dict(graph)
        #print(f"edges_dict: {edges_dict}")
        
        while changes_made:
            changes_made = False  # このループでの変更がないと仮定
            
            #for node, edges_list in edges_dict.items():
            # 各ノードから出発するエッジ間で結合を試みる
            #print("join loop edge if exists ===============================")

            for node, edges_list in edges_dict.items():
                #print(f"node:{node}, {edges_list}")
                
                for edge1 in edges_list:
                    #node_edge1end = edge1[1]
                    #print(f"edge1: {edge1}; ({node},{node_edge1end})")
                    
                    for edge2 in (edges_dict[node] + edges_reverse_dict.get(node, [])):
                        #print(f"edge1: {edge1}, edge2: {edge2}")
                        
                        e1s = edge1[0]
                        e1e = edge1[1]
                        e2s = edge2[0]
                        e2e = edge2[1]

                        if edge1 == edge2:
                            #print("skip as same")
                            break
                        if e1s == e1e:
                            #print(f"{e1s} == {e1e}")
                            if e1e == e2s:
                                merge_edges(graph, edge1, edge2)
                                changes_made = True
                                break
                            elif e1e == e2e:
                                merge_edges(graph, edge1, edge2, reverse_second = True)                                
                                changes_made = True
                                break
                        elif e2s == e2e:
                            #print(f"{e1s} == {e2e}")
                            if e1e == e2s:
                                merge_edges(graph, edge1, edge2)
                                changes_made = True
                                break
                            elif e1s == e2s:
                                merge_edges(graph, edge1, edge2, reverse_first = True)                                
                                changes_made = True
                                break
                            
                    if changes_made:
                        edges_dict, edges_reverse_dict = create_edges_reverse_dict(graph)
                        break
        
                if changes_made:
                    break
        
          
            if not changes_made:
                
                for node, edges_list in edges_dict.items():
                    #print(f"node:{node}, {edges_list}")
                    
                    for edge1 in edges_list:
                        node_edge1end = edge1[1]
                        #print(f"edge1: {edge1}; ({node},{node_edge1end})")


                        for edge2 in (edges_dict[node] + edges_reverse_dict.get(node, []) + edges_dict.get(node_edge1end, [])
                                       + edges_reverse_dict.get(node_edge1end, [])):
                            #print(f"edge2: {edge2}")
                            
                            e1s = edge1[0]
                            e1e = edge1[1]
                            e2s = edge2[0]
                            e2e = edge2[1]
            
                            if edge1 == edge2:
                                #print("skip")
                                a=1
                                #break                    
                            elif e1e == e2s:
                                #print(f"connect FF: ({e1s}, {e1e}), ({e2s}, {e2e})")
                                merge_edges(graph, edge1, edge2)
                                changes_made = True
                                break
                            elif e1e == e2e:
                                #print(f"connect FT: ({e1s}, {e1e}), ({e2s}, {e2e})")
                                merge_edges(graph, edge1, edge2, reverse_second=True)
                                changes_made = True
                                break
                            elif e1s == e2s:
                                #print(f"connect TF: ({e1s}, {e1e}), ({e2s}, {e2e})")
                                merge_edges(graph, edge1, edge2, reverse_first=True)
                                changes_made = True
                                break
                            elif e1s == e2e:
                                #print(f"connect TT: ({e1s}, {e1e}), ({e2s}, {e2e})")
                                merge_edges(graph, edge1, edge2, reverse_first=True, reverse_second=True)
                                changes_made = True
                                break
                                
                        if changes_made:
                            edges_dict, edges_reverse_dict = create_edges_reverse_dict(graph)
                            break
            
                    if changes_made:
                        break

            # エッジの結合または接続が行われた場合、関連するエッジを更新
            if changes_made:
                edges_dict, edges_reverse_dict = create_edges_reverse_dict(graph)
            
            # for benchmark
            #end_time = time.time()
            #print(f"Process edges took {end_time - start_time} seconds.")

    # グラフ内のエッジを処理
    process_edges(combined_graph)

    return combined_graph


def number_of_isolates(graph):
    isolate_count = 0
    for node in graph.nodes():
        if graph.degree(node) == 0:
            isolate_count += 1
    return isolate_count



def remove_isolates(graph):
    remove_isolates_graph = nx.MultiGraph(graph)  # マルチグラフのコピーを作成
    isolates = [node for node in remove_isolates_graph.nodes() if remove_isolates_graph.degree(node) == 0]
    for node in isolates:
        remove_isolates_graph.remove_node(node)
    return remove_isolates_graph

def remap_node_ids(graph):
    mapping = {old_id: new_id for new_id, old_id in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping)


"""
Path Optimization by OR-Tools TSP solver
"""

# 1. ポイント間の距離行列の作成
def create_distance_matrix(graph):
    # ノードIDを使用してポイントのリストを作成
    points = {n: graph.nodes[n]['o'].tolist() for n in graph.nodes}
    distance_matrix = {}
    
    for i, point_i in points.items():
        distances = {}
        for j, point_j in points.items():
            if i != j:
                # ユークリッド距離を計算
                distances[j] = int(np.linalg.norm(np.array(point_i) - np.array(point_j)))
            else:
                distances[j] = 0
        distance_matrix[i] = distances
        
    return distance_matrix, points

# 2. OR-Toolsを使用して最短ルートを計算
def calculate_optimized_route(distance_matrix, graph):
    start_node=list(graph.nodes())[0] # graphの中にあるいずれかのnodeを始点とする
    print(f"start_node: {start_node}")
    
    # ノードの数を取得
    num_nodes = len(distance_matrix)
    print(f"num_nodes: {num_nodes}")
    
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, start_node)
    routing = pywrapcp.RoutingModel(manager)

    
    def distance_callback(from_index, to_index):
        #print("distance_callback")
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        # 距離の計算
        distance = distance_matrix[from_node][to_node]
    
        # デバッグ情報の出力
        #print(f"Distance from node {from_node} to node {to_node}: {distance}")
        
        return distance #distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # OR-Tools の制約をノードIDに基づいて設定
    for (s, e) in graph.edges():
        if s != e:
            routing.solver().Add(
                routing.NextVar(manager.NodeToIndex(s)) == manager.NodeToIndex(e))
          
    # パラメータ設定
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    def print_solution(manager, routing, solution):
        """Prints solution on console."""
        print(f"Objective: {solution.ObjectiveValue()} miles")
        index = routing.Start(0)
        plan_output = "Route for vehicle 0:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} ->"
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += f" {manager.IndexToNode(index)}\n"
        print(plan_output)
        plan_output += f"Route distance: {route_distance}miles\n"
    
    
    # 解の計算
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print_solution(manager, routing, solution)

    
    # ルートの取得
    index = routing.Start(0)
    total_distance = 0
    route = []
    route.append(start_node)
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        
        route_distance = routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        total_distance += route_distance
        
    print('Total distance of the route: {}'.format(total_distance))    
    return route
    


  
"""
PNG file Generator
"""
def draw_graph(graph, colors, save_dir, img_size, save_name):#, img_size):

    # グラフ描画サイズを元の画像のアスペクト比に合わせる
    aspect_ratio = img_size[0] / img_size[1]
    fig_size = (10.0 * aspect_ratio, 10.0)
    plt.rcParams['figure.figsize'] = fig_size
    
    # グラフ描画のための設定
    color_index = 0
    
    # draw edges by pts
    for s, e in graph.edges():
        current_color = colors[color_index % len(colors)]
        
        pt_s = graph.nodes[s]['o'].tolist()
        pt_e = graph.nodes[e]['o'].tolist()

        for edge_key, edge_data in graph[s][e].items():
            pts = edge_data['pts'].tolist()
            pt_all = pts
            
            pt_all_np = np.array(pt_all)

            plt.plot(pt_all_np[:, 0], pt_all_np[:, 1], current_color)
            #print(f"{s}, {e}, {edge_key}, {current_color}") 
        color_index += 1

    # グラフをファイルに保存
    plt.savefig(os.path.join(save_dir, save_name + '_graph.png'))
    plt.close()

"""
SVG file Generator
"""
# without invalid path
def draw_svg(graph, colors, save_dir, save_name):
    outfile_name = os.path.join(save_dir, save_name + '.svg')
    dwg = svgwrite.Drawing(outfile_name, profile='tiny')

    color_index = 0

    for s, e in graph.edges():
        #pt_s = graph.nodes[s]['o'].tolist()
        #pt_e = graph.nodes[e]['o'].tolist()
        #print(f"{s}, {e}, {pt_s}, {pt_e}")
        for edge_key, edge_data in graph[s][e].items():
            pts = edge_data['pts'].tolist()
            current_color = colors[color_index % len(colors)]
            
            dwg.add(dwg.polyline(points=pts,
                                 stroke=current_color,
                                 stroke_width=0.8,
                                 fill='none',
                                 id=f"{s}_{e}_{edge_key}"
                                ))
            color_index += 1
    
    try:
        dwg.save()
        print('Saved:', outfile_name)
    except Exception as e:
        print('Error saving file:', e)
    finally:
        del dwg


# with invalid path
def add_polyline_to_dwg(dwg, points, stroke_color, stroke_width=1, fill_color='none'):
    dwg.add(dwg.polyline(
        points=points,
        stroke=stroke_color,
        stroke_width=stroke_width,
        fill=fill_color,
        id='id_'  # + '{0:03}'.format(num),
    ))
    
def generate_svg(graph, route, file_name, invalid_path=False):
    dwg = svgwrite.Drawing(file_name, profile='tiny')

    last_start_node = route[0]
    last_end_node = route[0]
    invalid_start_node = route[0]
    draw_flag = False
    
    for i in range(len(route) - 1):
        print(f"route:{i}, {route[i]}")
        start_node, end_node = route[i], route[i + 1]
        self_loop = (start_node, start_node) in graph.edges()
        edge_exists = (start_node, end_node) in graph.edges()

        if self_loop:
            print(f"Self loop at {start_node}")
            draw_flag = True
            last_end_node = start_node
            stroke_color = 'red'

            pt_s = graph.nodes[start_node]['o'].tolist()
            for g in graph[start_node][start_node].values():
                draw_points = [pt_s] + g['pts'].tolist() + [pt_s]
                add_polyline_to_dwg(dwg, draw_points, stroke_color)

        if (start_node, start_node) in graph.edges():
            print(f"{start_node}, {start_node} exists")
            draw_flag = True
            lsat_start_node = start_node
            last_end_node = start_node

            pt_s = graph.nodes[start_node]['o'].tolist()
            pt_e = graph.nodes[start_node]['o'].tolist()
            stroke_color = 'black'

            for g in graph[start_node][start_node].values():
                draw_points = g['pts'].tolist()
                draw_points = [pt_s] + draw_points + [pt_e]
                
                # ポリゴンラインで生成する場合
                add_polyline_to_dwg(dwg, draw_points, stroke_color)
                #print(f"draw: {start_node}, {end_node}")
        else:
            draw_flag = False
            invalid_start_node = last_end_node

        if (start_node, end_node) in graph.edges():
            print(f"{start_node}, {end_node} exists")
            draw_flag = True
            lsat_start_node = start_node
            last_end_node = end_node

            pt_s = graph.nodes[start_node]['o'].tolist()
            pt_e = graph.nodes[end_node]['o'].tolist()
            stroke_color = 'black' if start_node != end_node else 'red'

            for g in graph[start_node][end_node].values():
                draw_points = g['pts'].tolist()
                draw_points = [pt_s] + draw_points + [pt_e]
                
                # ポリゴンラインで生成する場合
                add_polyline_to_dwg(dwg, draw_points, stroke_color)
                #print(f"draw: {start_node}, {end_node}")

        else:
            draw_flag = False
            invalid_start_node = last_end_node
        
        if invalid_path and draw_flag:
            #draw_flag = False
            
            #print(f"{invalid_start_node}, {lsat_start_node} : Z up path between edge and edge")

            pt_s = graph.nodes[invalid_start_node]['o'].tolist()
            pt_e = graph.nodes[lsat_start_node]['o'].tolist()
            
            draw_points = [pt_s] + [pt_e]
            
            # ポリゴンラインで生成する場合
            add_polyline_to_dwg(dwg, draw_points, 'gray', stroke_width=0.5)
            #print(f"invalid: {invalid_start_node}, {lsat_start_node}")
    dwg.save()
    print(f"saved: {file_name}")
    del dwg



def main():
    print("flats-raster-vector-converter is running")

    # Set file names and paths
    filename = 'irasutoya'
    file_path = './img_line/' + filename + '.png'
    save_dir = './out/'
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'black']
    
    processed_image, data, img_size = process_image(file_path, save_dir)

    graph = create_skeltonize_graph(processed_image)

    draw_graph(graph, colors, save_dir, img_size, filename + '_raw')


    """
    Graph Optimization
    This section optimizes the graph by combining similar edges, which simplifies the graph structure and improves processing efficiency.
    """

    print("Optimizing graph starts.")

    # Combine Lines of Graph
    combined_graph = combine_edges(graph)

    save_dir2 = './out/'
    draw_graph(combined_graph, colors, save_dir2, img_size, filename + '_combined')

    # show graph information
    #isolates = number_of_isolates(combined_graph)
    #print(f"Number of isolated nodes: {isolates}")
    print(f"Graph information before optimizing: {combined_graph}")

    save_dir_svg = './out/'
    draw_svg(combined_graph, colors, save_dir_svg, filename + '_combined')
    draw_svg(combined_graph, ['black'], save_dir_svg, filename + '_combined' + '_black')

    # remove unused node and remap nodes
    graph_removed_isolates = remove_isolates(combined_graph)
    remapped_graph = remap_node_ids(graph_removed_isolates)

    print("Optimizing graph has finished.")
    print(f"Graph information after optimizing: {remapped_graph}")
    

    """
    Path Optimization with Benchmark
    This section involves solving the Traveling Salesman Problem (TSP) to find the most efficient path.
    It includes benchmarking to measure the performance of the optimization process.
    """
    print("Optimizing path (Solving TSP) starts.")

    # oprimizing root by OR-Tools TSP solver
    start_time = time.time()

    distance_matrix, points = create_distance_matrix(remapped_graph)
    route = calculate_optimized_route(distance_matrix, remapped_graph)

    end_time = time.time()
    print(f"Process edges took {end_time - start_time} seconds.")
    
    print("Optimizing path (Solving TSP) has finished.")

    # show optimized route
    print(route)

    # export optimized path as svg files
    generate_svg(remapped_graph, route, './out/' + filename + '_optimized_path.svg')
    generate_svg(remapped_graph, route, './out/' + filename + '_optimized_path_invalid.svg', invalid_path=True)

    print("completed")


if __name__ == "__main__":
    main()