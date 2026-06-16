# Lib for data simulation based on a graph 
# Composed by: 
#   1. graph generation (Modified Albert-Barabasi Model) or import a graph (networkx) from a pickle file
#   2. Graph analysis
#   3. Build a system of equations based on the causal parents (Structural Causal Models) and a graph, atributing the coefficients under a t-student distribution
#   4. Generate samples with measurement noise 
#   5. Generation of a faulty data: Process perturbation, sensor bias, and correlation changes.

import copy
import itertools
import pickle
import random
import time
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

 ##-------------- Auxiliary Functions to convert graphs and graph representation (Root vs Non-root nodes) ----------------##

def convert_to_igraph(g):
        
    """ARGS:
    g (graph) - networkx graph

    OUTPUTS:
    ig_graph (graph) - igraph graph
    """    
    ig_graph = ig.Graph.from_networkx(g)
    if "name" not in ig_graph.vs.attributes():
        ig_graph.vs["name"] = [str(node) for node in g.nodes()]
    return ig_graph
    
def plot_disjoint_communities_igraph(graph, partition, ax=None):

    """Convert a NetworkX graph into an igraph graph object.
    
    ARGS:
    graph (igraph) - graph
    partition - list of lists
    """
    
    layout=graph.layout('fr')
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
        is_standalone = True
    else:
        is_standalone = False
    
    # Communities colors
    colors = ['red', 'lightblue', 'green', 'blue', 'mediumvioletred', 'orange', 'white']
    node_colors = [None] * len(graph.vs)

    for i, community in enumerate(partition):
        for node in community:
            node_colors[graph.vs.find(name=str(node)).index] = colors[i % len(colors)]
        
    ig.plot(graph, layout=layout, vertex_label=graph.vs["name"], target=ax, vertex_color=node_colors, vertex_size=40, edge_width=1, bbox=(1200, 1200), margin=40)
    
    labels = ['Root-nodes', 'Non root-nodes']
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=legend_handles, loc='upper left')
    
    if is_standalone:
        plt.show()

##-------------- Graph Generation or Importation from a pickle file ----------------##

def graph_generation_or_importation(number_of_nodes, 
                                    degree_mean=1, 
                                    execute_graph=True, 
                                    file_name=None, 
                                    ax = None):
    
    """Generate a random directed Barabasi graph or import an existing NetworkX graph from a pickle file.
    
    ARGS:

    number_of_nodes - number of the nodes in the network
    degree_mean - The mean of the degree of the nodes
    execute_graph - True, if you want a graph generation, False if you want to import a pickle file with a networkx graph
    file_name - if execute_graph = False, file_name is the directory, i.e., 'file_name.pkl'

    OUTPUTS:
    g - network (networkx)
    nodes - list with the name of the variables
    """

    if execute_graph:
        number_of_out_edges = list(np.random.poisson(degree_mean,number_of_nodes))
        number_of_out_edges = [max(1, degree) for degree in number_of_out_edges]  # Guarantee of no out-degree = 0
        nodes = [f"X{i+1}" for i in range (number_of_nodes)]

        g = ig.Graph.Barabasi(n=number_of_nodes, m=number_of_out_edges,outpref=False, directed=True)
        g.vs["name"] = nodes
        root_nodes_b = []
        non_root_nodes_b = []

        for vertex in g.vs:
            if len(vertex.predecessors()) == 0:
                root_nodes_b.append(vertex["name"])
            else:
                non_root_nodes_b.append(vertex["name"])

        partition = [root_nodes_b, non_root_nodes_b]

        plot_disjoint_communities_igraph(g, partition, ax = ax)
        g1 = g.to_networkx()
        g2 = nx.relabel_nodes(g1, {old_node: new_node for old_node, new_node in zip(g1.nodes, nodes)})
        return g2 , nodes
    
    else:
        if file_name is None:
            raise ValueError("file_name must be provided when execute_graph is False")
        
        with open(file_name, 'rb') as f:
            barabasi = pickle.load(f)
            nodes = pickle.load(f)

        barabasi1 = convert_to_igraph(barabasi)
        barabasi1.vs["name"] = nodes
        root_nodes = []
        non_root_nodes = []

        for vertex in barabasi1.vs:
            if len(vertex.predecessors()) == 0:
                root_nodes.append(vertex["name"])
            else:
                non_root_nodes.append(vertex["name"])

        partition = [root_nodes, non_root_nodes]
        
        plot_disjoint_communities_igraph(barabasi1, partition, ax = ax)
        g11 = nx.relabel_nodes(barabasi, {old_node: new_node for old_node, new_node in zip(barabasi.nodes, nodes)})
        return g11, nodes
    
##-------------- Graph Analysis ----------------##

def has_edge(G, u, v):
    """Check if an edge exists between two nodes in either direction."""
    return G.has_edge(u, v) or G.has_edge(v, u)

def classify_triplets(G):
    """Find and classify all node triplets in the graph into closed triangles and open triplets.

    ARGS:
    G (graph) - networkx graph

    OUTPUTS:
    closed_triangles - list of tuples, each containing 3 mutually connected nodes
    open_triplets - list of tuples, each containing 3 nodes with exactly 2 edges between them
    """
    closed_triangles = []
    open_triplets = []
    nodes = list(G.nodes())
    
    for u, v, w in itertools.combinations(nodes, 3): 
    
        uv = has_edge(G, u, v)
        vw = has_edge(G, v, w)
        wu = has_edge(G, w, u)
       
        edges_present = sum([uv, vw, wu])
        
        if edges_present == 3:
            closed_triangles.append((u, v, w))
        
        elif edges_present == 2:
            open_triplets.append((u, v, w))
    
    return closed_triangles, open_triplets

def graph_analysis(g):

    """Analyze a graph and print key structural metrics.

    Calculates and prints fundamental graph statistics including the counts of 
    nodes, edges, root nodes, non-root nodes, sinks, open/closed triplets, 
    graph density, and whether the network qualifies as a Directed Acyclic Graph (DAG).

    ARGS:
    g (graph) - networkx graph to be analyzed
    """

    sinks = []
    for node in g.nodes:
        if len(list(nx.descendants(g, node))) == 0:
            sinks.append(node)

    dag = nx.is_directed_acyclic_graph(g)

    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    dens = nx.density(g)

    graph = convert_to_igraph(g)
 
    root_nodes_b = [vertex["name"] for vertex in graph.vs if len(vertex.predecessors()) == 0]
    non_root_nodes_b = [vertex["name"] for vertex in graph.vs if len(vertex.predecessors()) > 0]

    num_roots = len(root_nodes_b)
    num_sinks = len(sinks)
    num_non = len(non_root_nodes_b)
    closed_trip, open_trip= classify_triplets(g)

    print("SOME METRICS!")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Number of Root nodes: {num_roots}")
    print(f"Number of Non-Root nodes: {num_non}")
    print(f"Number of open triplets: {len(open_trip)}")
    print(f"Number of closed triplets: {len(closed_trip)}")
    print(f"Graph Density: {dens:.4f}")
    print(f"Is a DAG? {'Yes' if dag else 'No'}")
    
##-------------- Get Parameters for the Equation System ----------------##

def get_params(g, nodes, process_fault=0, freedom_degree=3,faulty_node=None,view_equation=True):
    """Generate parameters for a Structural Causal Model (SCM) based on a causal graph.
    
    ARGS: 
    g - generated graph (networkx)
    nodes -list with the name of the variables
    freedom_degree- freedom degree of t distribution to bij (coefs)
    process_fault - Fault magnitude : mu = mu + process_fault*var
    faulty_node - list of Name of the node/s where the fault will be introduced. 
                    e.g. Fault on X2: faulty_node=['X2'], Simultaneous faults on X2 and X5: faulty_node=['X2','X5']

    OUTPUTS:
    params(dic) - Params of the distribution of root nodes (mean, var)
    intercept (dic) - intercept of the equations of non-root
    nonroot_coefs (dic) - Bji, coeficient of the parent j in child i, by the order of g.predecessors(node) 
    """
    np.random.seed(42) # ensuring the reproducibility of Bij
    random.seed(42) # ensuring the reproducibility of distribution parameters (root-nodes)
    params = {}
    intercept = {}
    nonroot_coefs = {}

    for node in nodes:
        parents = [parent for parent in g.predecessors(node)]
        if len(parents) == 0:
            mu = random.uniform(1, 3)
            var = random.uniform(1, 3)
            if faulty_node and node in faulty_node:
                mu = mu + process_fault * np.sqrt(var)
            params[node] = (mu, var)
            if view_equation:
                print(f"{node}: (mean, var) = ({mu:.2f}, {var:.2f})")
            else:
                pass
        else:
            intercept[node] = np.random.choice([0,1]) # intercept 0 or 1
            coefs = np.random.standard_t(df= freedom_degree,size=len(parents)) 
            nonroot_coefs[node] =  coefs 

            expression = " + ".join([f"({c:.2f})*{pa}" for c, pa in zip(coefs, parents)])
            if view_equation:
                print(f"{node} = {intercept[node]}+{expression}")
            else:
                pass

    return params, intercept, nonroot_coefs

##-------------- Perform a Correlation Fault ----------------##

def correlation_fault(non_root_coefs, g, k=0, affect_edge=None):
    """Inject a correlation fault into specific graph edges by scaling their causal coefficients.
    
    ARGS: 
    non_root_coefs - coefficients (dic)
    g - generated graph (networkx)
    k -  percentage factor, if k = 0 there isn't fault (float) 
    affected_edge - Pair (parent,child) in a list, e.g [('X2','X1')]. For multiple faults, e.g. [ [('X2','X1')] , [('X5','X9')] ] 

    OUTPUTS:
    nonroot_coefs_mod (dic) - modified Bji
    """

    non_root_coefs_mod = copy.deepcopy(non_root_coefs)
    
    if affect_edge is None:
        affect_edge = []

    for edge in affect_edge:
        parent_node, child_node = edge
        
        if child_node in non_root_coefs_mod:
            try:
                parents = list(g.predecessors(child_node))
                parent_index = parents.index(parent_node)
            
                if parent_index < len(non_root_coefs_mod[child_node]):
                    
                    old_coef = non_root_coefs_mod[child_node][parent_index]

                    non_root_coefs_mod[child_node][parent_index] = old_coef*(1+(k/100)) 
            except ValueError:
                pass

    return non_root_coefs_mod

##-------------- Data Generation ----------------##

def generate_data(num_observations, g, params, nonroot_coefs, intercept):
    """Simulate dataset observations from a Structural Causal Model (SCM) using topological sorting.
    
    ARGS: 
    num_observations - number of observations
    g - networkx graph
    params(dic) - Distribution Parameters of root nodes (mean, var)
    intercept (dic) - intercept of the equations for non-root nodes
    nonroot_coefs (dic) - Bji, coeficient of the parent j in child i 
    
    OUTPUTS:
    node_values (dic) - Samples of the variables
    """

    np.random.seed(int(time.time() *1000) % (2**32))
    tpsort= list(nx.topological_sort(g))
    node_values = {node: np.zeros(num_observations) for node in tpsort}
    
    # Sampling to root nodes
    for node in tpsort:
        parents = list(g.predecessors(node))
        if not parents:
            mu, var = params[node]
            node_values[node] = np.random.normal(loc=mu, scale=np.sqrt(var), size=num_observations)

    # Data for non-root nodes
    for node in tpsort:
        parents = list(g.predecessors(node))
        if parents:
            intercept_val = intercept[node]
            coefs = nonroot_coefs[node]
            parent_values = np.array([node_values[parent] for parent in parents])
            coefs = coefs[:, np.newaxis]  # Reshape coefs to (n, 1)
            _, var = params.get(node, (0, 1))  
            e_i = np.random.normal(loc=0, scale=np.sqrt(var), size=num_observations)
            node_values[node] = intercept_val + np.sum(coefs * parent_values, axis=0) + e_i

    return node_values  

##-------------- Add Measurement Noise and Perform Sensor Bias Fault ----------------##

def add_measurement_noise(data, nodes, snr_db,sensor_fault=0,faulty_nodes=None):
    """ARGS: 
    data - node_values (dic) - Samples of the variables
    snr_db (int) - signal-to-noise ration measured in decibel 
    sensor_fault (int) - factor of the fault. If =0 there is no fault. 
    Faulty_nodes= location ['X2']
    data_faulty = data_var + sensor_fault[node]*variance_noise[node]

    OUTPUTS:
    noisy_data (dic) - Samples of the variables with measurement noise
    """

    if faulty_nodes is None:
        faulty_nodes=[]
    noisy_data = {}
    
    signal_variance = {node: np.var(values) for node, values in data.items()}
    snr_linear = 10 ** (snr_db / 10)
    
    for node, values in data.items():
        noise_std = np.sqrt(signal_variance[node] / snr_linear)
        noise = np.random.normal(loc=0, scale=noise_std, size=len(values))
        noisy_data[node] = values + noise 
        
        if node in faulty_nodes:
            sensor_var=sensor_fault*noise_std
            sensor_bias= np.random.normal(loc=0,scale=sensor_var,size=len(values))
            noisy_data[node]+=sensor_bias
    data_test = pd.DataFrame(noisy_data)
    noc_data = data_test[nodes]

    return noc_data


##-------------- Save and Export NOC datasets ----------------##

def save_datasets_cal_val(g,nodes,filename,num_observations=1000):
    
    """Apply Gaussian measurement noise to simulated data based on a target Signal-to-Noise Ratio (SNR) and inject sensor bias faults.
    
    ARGS: 
    g (graph) - causal graph (networkx)
    nodes = list of the label nodes
    filename (str) - name for the file ex. 'insert_name.csv'
    num_observations (float) - number of observations
    """
     
    root_nodes = []
    non_root_nodes=[]

    for node in nodes:
        parents = [pa for pa in g.predecessors(node)]
        if len(parents) == 0:
            root_nodes.append(node)
        else:
            non_root_nodes.append(node)

    partition = [root_nodes,non_root_nodes]
 
    bbb= convert_to_igraph(g)
    plot_disjoint_communities_igraph(bbb,partition)

    params, intercept, nonroot_coefs = get_params(g,nodes,process_fault=0,faulty_node=None)
    
    raw_data = generate_data(num_observations, g, params, nonroot_coefs, intercept)
    noc_data = add_measurement_noise(raw_data, nodes, snr_db=10)
    
    noc_data.to_csv(filename,index=False)
    print(f'Dataset successfully saved to: {filename}') 

   