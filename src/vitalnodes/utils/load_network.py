import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Optional


class DatasetConfigs:
    def __init__(self):
        # Dataset configurations: (file_path, time_window, column_names, header)
        self.dataset_configs = {
        'ABM-contacts': ('data/micro_abm_contacts.csv', 
                          1, 
                          ['Timestamp', 'PersonId1', 'PersonId2', 'Intensity', 'LocationId', 'LocationType'], 
                          0),

        'highschool-2011': ('data/highschool_2011.csv', 
                            3600, 
                            ['Timestamp', 'PersonId1', 'PersonId2', 'Class1', 'Class2'], 
                            None),

        'highschool-2012': ('data/highschool_2012.csv', 
                            3600, 
                            ['Timestamp', 'PersonId1', 'PersonId2', 'Class1', 'Class2'], 
                            None),

        'highschool-2013': ('data/highschool_2013.csv', 
                            3600, 
                            ['Timestamp', 'PersonId1', 'PersonId2', 'Class1', 'Class2'], 
                            None),

        'hospital-contact': ('data/hospital-contact.dat', 
                             3600, 
                             ['Timestamp', 'PersonId1', 'PersonId2', 'Role1', 'Role2'], 
                             None),

        'hypertext-2009': ('data/hypertext2009.dat', 
                           1800, 
                           ['Timestamp', 'PersonId1', 'PersonId2'],
                           None),

        'primaryschool': ('data/primaryschool.csv', 
                          1800, 
                          ['Timestamp', 'PersonId1', 'PersonId2', 'Class1', 'Class2'], 
                          None),

        'sfhh': ('data/sfhh.dat', 
                 1800, 
                 ['Timestamp', 'PersonId1', 'PersonId2'], 
                 None),

        'workplace': ('data/workplace.dat', 
                      3600, 
                      ['Timestamp', 'PersonId1', 'PersonId2'], 
                      None),

        'UCI': ('data/UCI.txt', 
                86400,  # Using 1 day (86400 seconds) as window 
                ['PersonId1', 'PersonId2', 'Timestamp'], 
                None),

        'Email': ('data/Email.txt', 
                  86400,  # Using 1 day as window
                 ['PersonId1', 'PersonId2', 'Timestamp'], 
                 None),
    }


dataset_configs = DatasetConfigs().dataset_configs


def load_abm_network(file_path: str, remap: bool = True,
                     node_attributes: Optional[Dict[int, Dict[str, int]]] = None,
                     required_ageGroup: Optional[str] = None) -> List[nx.Graph]:
    """
    Loads a temporal network from a CSV file into a list of NetworkX graphs,
    ensuring all nodes appear in every snapshot and including additional attributes.
    
    Optionally remaps the node IDs consistently across snapshots.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    remap : bool, optional
        If True, remap node IDs consistently using remap_nodes.
    node_attributes : dict, optional
        A dictionary mapping node IDs to attributes (used for filtering).
    required_ageGroup : str, optional
        If provided, only nodes with node_attributes[node]['age_group_name'] == required_ageGroup are kept.
    
    Returns
    -------
    temporal_network : list of nx.Graph
        List of graph snapshots (one per unique Hour).
    """
    # Load the CSV file with proper column names
    df = pd.read_csv(
        file_path,
        comment='#',
        names=['Hour', 'PersonId1', 'PersonId2', 'Intensity', 'LocationId', 'LocationType']
    )

    # Ensure numeric columns are parsed correctly
    df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
    df['PersonId1'] = pd.to_numeric(df['PersonId1'], errors='coerce')
    df['PersonId2'] = pd.to_numeric(df['PersonId2'], errors='coerce')
    df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
    df.dropna(inplace=True)

    # Determine the full set of nodes across all snapshots
    all_nodes = set(df['PersonId1']).union(set(df['PersonId2']))
    
    temporal_network = []
    hours = sorted(df['Hour'].unique())

    for hour in tqdm(hours, desc="Loading ABM network"):
        group = df[df['Hour'] == hour]
        G = nx.from_pandas_edgelist(
            group,
            'PersonId1',
            'PersonId2',
            edge_attr=['Intensity', 'LocationId', 'LocationType']
        )

        # Ensure every node appears in this snapshot
        for node in all_nodes:
            if node not in G:
                G.add_node(node)

        # Copy 'Intensity' into 'weight' attribute
        for u, v, data in G.edges(data=True):
            data["weight"] = data["Intensity"]

        temporal_network.append(G)

    if remap:
        temporal_network = remap_nodes(temporal_network,
                                        node_attributes=node_attributes,
                                        required_ageGroup=required_ageGroup)
    return temporal_network


def remap_nodes(temporal_network: List[nx.Graph],
                node_attributes: Optional[Dict[int, Dict[str, int]]] = None,
                required_ageGroup: Optional[str] = None) -> List[nx.Graph]:
    """
    Remap nodes in each snapshot to ensure consistent mapping across snapshots
    and optionally filter nodes based on metadata (e.g., required_ageGroup).

    Parameters
    ----------
    temporal_network : List[nx.Graph]
        The list of graph snapshots.
    node_attributes : dict, optional
        Mapping of node IDs to attribute dictionaries.
    required_ageGroup : str, optional
        If provided, only keep nodes with 'age_group_name' equal to required_ageGroup.

    Returns
    -------
    remapped_temporal_network : List[nx.Graph]
        The list of remapped (and optionally filtered) graph snapshots.
    """
    global_node_mapping = {}
    next_node_id = 0

    # Create global mapping for all nodes across snapshots
    for snapshot in temporal_network:
        for node in snapshot.nodes():
            if node not in global_node_mapping:
                global_node_mapping[node] = next_node_id
                next_node_id += 1

    # Determine valid nodes based on node_attributes and filtering
    if node_attributes:
        valid_nodes = set(global_node_mapping.keys())
        if required_ageGroup:
            valid_nodes = {node for node in valid_nodes
                           if node in node_attributes and
                           node_attributes[node].get("age_group_name", "") == required_ageGroup}
    else:
        valid_nodes = set(global_node_mapping.keys())

    remapped_temporal_network = []
    for snapshot in temporal_network:
        remapped_graph = nx.Graph()

        # Add valid nodes (remapped) even if isolated
        for node in valid_nodes:
            remapped_graph.add_node(global_node_mapping[node])

        # Add edges from the snapshot if both endpoints are valid
        for u, v, data in snapshot.edges(data=True):
            if u in valid_nodes and v in valid_nodes:
                remapped_u = global_node_mapping[u]
                remapped_v = global_node_mapping[v]
                if remapped_u != remapped_v:  # Prevent self-edges
                    remapped_graph.add_edge(remapped_u, remapped_v, **data)
        remapped_temporal_network.append(remapped_graph)

    return remapped_temporal_network


def load_generic_network(file_path, time_window, 
                         column_names, header = None, remap = True,
                         node_attributes = None) -> List[nx.Graph]:
    """
    Generic function to load temporal networks from a file into a list of NetworkX graphs.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.
    time_window : int
        Time window for grouping interactions into snapshots (in seconds or hours).
    column_names : list of str
        Column names for the dataset.
    remap : bool, optional
        If True, remap node IDs using `remap_nodes`.
    node_attributes : dict, optional
        Dictionary mapping node IDs to attributes (used only if remap is True).
    timestamp_unit : str, optional
        Unit of the timestamp ('seconds' or 'hours').

    Returns
    -------
    temporal_network : list of nx.Graph
        List of graph snapshots.
    """

    # Load the dataset
    df = pd.read_csv(file_path, header=header, sep=r"\s+", names=column_names)

    # Convert columns
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df['Window'] = ((df['Timestamp'] - df['Timestamp'].min()) // time_window).astype(int)
    df['PersonId1'] = pd.to_numeric(df['PersonId1'], errors='coerce')
    df['PersonId2'] = pd.to_numeric(df['PersonId2'], errors='coerce')
    df.dropna(subset=['PersonId1', 'PersonId2'], inplace=True)  # Drop rows with invalid person IDs

    # Get all unique nodes
    all_nodes = set(df['PersonId1']).union(set(df['PersonId2']))
    time_windows = range(df['Window'].min(), df['Window'].max() + 1)
    temporal_network = []

    for window in tqdm(time_windows, desc=f"Loading {file_path}"):
        group = df[df['Window'] == window]
        G = nx.from_pandas_edgelist(group, 'PersonId1', 'PersonId2')

        # Ensure all nodes appear in the snapshot
        for node in all_nodes:
            if node not in G:
                G.add_node(node)

        # Add weight of 1 for each contact
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        temporal_network.append(G)

    if remap:
        temporal_network = remap_nodes(temporal_network, node_attributes=node_attributes)

    return temporal_network


def load_network_on_dataset_name(dataset_name: str) -> List[nx.Graph]:
    """
    Load a temporal network based on the dataset name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.

    Returns
    -------
    temporal_network : list of nx.Graph
        List of graph snapshots.
    """

    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} not recognized. Please choose from the available datasets.")

    file_path, time_window, column_names, header = dataset_configs[dataset_name]
    return load_generic_network(file_path, time_window, column_names, header)


def load_network(dataset_name):
    if dataset_name == 'ABM-contacts':
        return load_abm_network('data/micro_abm_contacts.csv') 
    elif dataset_name in ['highschool-2011', 'highschool-2012', 'highschool-2013', 'hospital-contact', 'hypertext-2009', 'primaryschool', 'sfhh', 'workplace', 'UCI', 'Email']:
        return load_network_on_dataset_name(dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Please choose from the available datasets.")
