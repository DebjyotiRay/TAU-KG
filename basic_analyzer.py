import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BasicAnalyzer:
    def __init__(self, G):
        self.G = G
        
    def get_basic_stats(self):
        """Calculate comprehensive basic network statistics"""
        stats = {
            "Total Nodes": len(self.G.nodes),
            "Total Edges": len(self.G.edges),
            "Average Degree": np.mean([d for n, d in self.G.degree()]),
            "Network Density": nx.density(self.G),
            "Average Clustering Coefficient": nx.average_clustering(self.G),
            "Connected Components": nx.number_connected_components(self.G)
        }

        # Path-based metrics for largest component
        largest_cc = max(nx.connected_components(self.G), key=len)
        largest_subgraph = self.G.subgraph(largest_cc)
        
        stats["Largest Component Size"] = len(largest_cc)
        stats["Largest Component Ratio"] = len(largest_cc) / len(self.G.nodes)
        
        if nx.is_connected(self.G):
            stats["Average Path Length"] = nx.average_shortest_path_length(self.G)
            stats["Graph Diameter"] = nx.diameter(self.G)
        else:
            stats["Average Path Length"] = nx.average_shortest_path_length(largest_subgraph)
            stats["Graph Diameter"] = nx.diameter(largest_subgraph)
            stats["Note"] = "Metrics calculated on largest connected component"

        # Node type distribution
        node_types = [data["type"] for n, data in self.G.nodes(data=True)]
        type_counts = Counter(node_types)
        for node_type, count in type_counts.items():
            stats[f"{node_type.capitalize()} Count"] = count

        return stats

    def get_cluster_stats(self, cluster_name):
        """Calculate detailed statistics for a specific cluster"""
        cluster_nodes = [n for n, d in self.G.nodes(data=True)
                        if d.get("cluster") == cluster_name]

        if not cluster_nodes:
            return None

        subgraph = self.G.subgraph(cluster_nodes)
        
        stats = {
            "Nodes in Cluster": len(subgraph.nodes),
            "Edges in Cluster": len(subgraph.edges),
            "Cluster Density": nx.density(subgraph),
            "Average Clustering Coefficient": nx.average_clustering(subgraph),
            "Average Degree": np.mean([d for n, d in subgraph.degree()]),
            "Diameter": nx.diameter(subgraph) if nx.is_connected(subgraph) else np.inf,
            "Internal Edge Ratio": len(subgraph.edges) / len(self.G.edges)
        }

        # Node type distribution
        node_types = [data["type"] for n, data in subgraph.nodes(data=True)]
        type_counts = Counter(node_types)
        for node_type, count in type_counts.items():
            stats[f"{node_type.capitalize()} Count"] = count

        # Calculate key nodes using multiple centrality measures
        centrality_measures = {
            "Degree": nx.degree_centrality(subgraph),
            "Betweenness": nx.betweenness_centrality(subgraph),
            "Closeness": nx.closeness_centrality(subgraph),
            "Eigenvector": nx.eigenvector_centrality(subgraph, max_iter=1000)
        }

        stats["Key Nodes"] = {}
        for measure_name, centrality_dict in centrality_measures.items():
            sorted_nodes = sorted(centrality_dict.items(),
                                key=lambda x: x[1],
                                reverse=True)[:5]
            stats["Key Nodes"][measure_name] = sorted_nodes

        return stats

    def get_pathway_analysis(self):
        """Analyze pathway connections and influences"""
        pathway_nodes = [n for n, d in self.G.nodes(data=True)
                        if d.get("type", "") == "pathway"]
        
        pathway_stats = {}
        for pathway in pathway_nodes:
            neighbors = list(self.G.neighbors(pathway))
            if not neighbors:
                continue
                
            neighbor_types = Counter([self.G.nodes[n].get("type", "Unknown") 
                                    for n in neighbors])
            neighbor_clusters = Counter([self.G.nodes[n].get("cluster", "Unknown") 
                                      for n in neighbors])

            interaction_strengths = [d.get("weight", 0.0) 
                                   for u, v, d in self.G.edges(pathway, data=True)]
            
            if interaction_strengths:
                pathway_stats[pathway] = {
                    "Total Connections": len(neighbors),
                    "Connected Types": dict(neighbor_types),
                    "Connected Clusters": dict(neighbor_clusters),
                    "Average Interaction Strength": np.mean(interaction_strengths),
                    "Max Interaction Strength": max(interaction_strengths),
                    "Interaction Strength Distribution": np.percentile(
                        interaction_strengths, [25, 50, 75]).tolist()
                }

        return pathway_stats