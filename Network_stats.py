import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st


class NetworkAnalyzer:
    def __init__(self, nodes_data, edges_data):
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        self.node_ids = {node["id"] for node in nodes_data}
        self.G = self._create_networkx_graph()

    def _create_networkx_graph(self):
        """Create a NetworkX graph from nodes and edges data"""
        G = nx.Graph()

        # Add nodes with attributes
        for node in self.nodes_data:
            G.add_node(node["id"], **node)  # Add all node attributes

        # Add edges with attributes
        for edge in self.edges_data:
            G.add_edge(edge["source"],
                      edge["target"],
                      weight=edge["score"],
                      relation=edge["relation"])

        return G

    def get_basic_stats(self):
        """Calculate basic network statistics"""
        stats = {
            "Total Nodes": len(self.G.nodes),
            "Total Edges": len(self.G.edges),
            "Average Degree": np.mean([d for n, d in self.G.degree()]),
            "Network Density": nx.density(self.G),
            "Average Clustering Coefficient": nx.average_clustering(self.G),
            "Connected Components": nx.number_connected_components(self.G)
        }

        # Node type distribution
        node_types = [data["type"] for n, data in self.G.nodes(data=True)]
        type_counts = Counter(node_types)
        for node_type, count in type_counts.items():
            stats[f"{node_type.capitalize()} Count"] = count

        return stats

    def get_cluster_stats(self, cluster_name):
        """Calculate statistics for a specific cluster"""
        # Get nodes in cluster
        cluster_nodes = [n for n, d in self.G.nodes(data=True)
                        if d.get("cluster") == cluster_name]

        if not cluster_nodes:
            return None

        # Create subgraph for cluster
        subgraph = self.G.subgraph(cluster_nodes)

        stats = {
            "Nodes in Cluster": len(subgraph.nodes),
            "Edges in Cluster": len(subgraph.edges),
            "Cluster Density": nx.density(subgraph),
            "Average Clustering Coefficient": nx.average_clustering(subgraph),
            "Average Degree": np.mean([d for n, d in subgraph.degree()])
        }

        # Node type distribution in cluster
        node_types = [data["type"] for n, data in subgraph.nodes(data=True)]
        type_counts = Counter(node_types)
        for node_type, count in type_counts.items():
            stats[f"{node_type.capitalize()} Count"] = count

        # Calculate key nodes
        centrality_measures = {
            "Degree Centrality": nx.degree_centrality(subgraph),
            "Betweenness Centrality": nx.betweenness_centrality(subgraph),
            "Closeness Centrality": nx.closeness_centrality(subgraph)
        }

        # Find top nodes for each centrality measure
        key_nodes = {}
        for measure_name, centrality_dict in centrality_measures.items():
            sorted_nodes = sorted(centrality_dict.items(),
                                key=lambda x: x[1],
                                reverse=True)[:3]
            key_nodes[measure_name] = sorted_nodes

        stats["Key Nodes"] = key_nodes

        return stats

    def get_top_interactions(self, n=5):
        """Get top N strongest interactions in the network"""
        edges = [(u, v, d) for u, v, d in self.G.edges(data=True)]
        sorted_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
        return sorted_edges[:n]

    def get_node_importance(self, n=5):
        """Calculate and return important nodes based on centrality metrics"""
        importance_metrics = {
            "Degree": nx.degree_centrality(self.G),
            "Betweenness": nx.betweenness_centrality(self.G),
            "Closeness": nx.closeness_centrality(self.G),
            "Eigenvector": nx.eigenvector_centrality(self.G, max_iter=1000)
        }

        top_nodes = {}
        for metric, values in importance_metrics.items():
            sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
            top_nodes[metric] = sorted_nodes[:n]

        return top_nodes

    def get_pathway_analysis(self):
        """Analyze pathway connections and influences"""
        pathway_nodes = [n for n, d in self.G.nodes(data=True)
                        if d["type"] == "pathway"]

        pathway_stats = {}
        for pathway in pathway_nodes:
            neighbors = list(self.G.neighbors(pathway))
            neighbor_types = Counter([self.G.nodes[n]["type"] for n in neighbors])

            pathway_stats[pathway] = {
                "Total Connections": len(neighbors),
                "Connected Types": dict(neighbor_types),
                "Average Interaction Strength": np.mean([
                    d["weight"] for u, v, d in self.G.edges(pathway, data=True)
                ])
            }

        return pathway_stats

    def get_cluster_interactions(self):
        """Analyze interactions between clusters"""
        interactions = {}
        for edge in self.edges_data:
            source_node = next((n for n in self.nodes_data if n["id"] == edge["source"]), None)
            target_node = next((n for n in self.nodes_data if n["id"] == edge["target"]), None)
            
            if source_node and target_node:
                source_cluster = source_node["cluster"]
                target_cluster = target_node["cluster"]
                
                if source_cluster != target_cluster:
                    key = tuple(sorted([source_cluster, target_cluster]))
                    if key not in interactions:
                        interactions[key] = {"count": 0, "edges": []}
                    interactions[key]["count"] += 1
                    interactions[key]["edges"].append(edge)
        
        return interactions

    def get_pmid_distribution(self):
        """Analyze PMID distribution across nodes"""
        pmid_stats = {}
        for node in self.nodes_data:
            pmid = str(node.get("PMID", "Unknown"))
            if pmid not in pmid_stats:
                pmid_stats[pmid] = {"count": 0, "nodes": []}
            pmid_stats[pmid]["count"] += 1
            pmid_stats[pmid]["nodes"].append(node["id"])
        return pmid_stats

    def get_temporal_analysis(self):
        """Perform temporal analysis based on PMIDs"""
        pmids = sorted(set(str(node.get("PMID", "Unknown")) for node in self.nodes_data))
        pmid_timeline = {pmid: {"nodes": [], "edges": []} for pmid in pmids}
        
        # Process nodes
        for node in self.nodes_data:
            pmid = str(node.get("PMID", "Unknown"))
            pmid_timeline[pmid]["nodes"].append(node["id"])
        
        # Process edges
        for edge in self.edges_data:
            source_pmid = str(next((n.get("PMID", "Unknown") 
                                  for n in self.nodes_data if n["id"] == edge["source"]), 
                                  "Unknown"))
            pmid_timeline[source_pmid]["edges"].append(edge)
        
        return pmid_timeline

    def display_stats_streamlit(self, selected_cluster=None):
        """Display comprehensive network statistics in Streamlit"""
        if selected_cluster and selected_cluster != "All":
            self._display_cluster_stats(selected_cluster)
        else:
            self._display_overall_stats()

        # Advanced Analysis Section
        st.header("Advanced Network Analysis")
        
        # Network Health Metrics
        self._display_health_metrics()
        
        # Temporal Analysis
        self._display_temporal_analysis()
        
        # Cluster Interactions
        self._display_cluster_interactions()
        
        # Interactive Network Explorer
        self._display_network_explorer()

    def _display_overall_stats(self):
        """Display overall network statistics"""
        st.header("Network Overview")
        basic_stats = self.get_basic_stats()

        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", basic_stats["Total Nodes"])
            st.metric("Network Density", f"{basic_stats['Network Density']:.3f}")
        with col2:
            st.metric("Total Edges", basic_stats["Total Edges"])
            st.metric("Avg Clustering", f"{basic_stats['Average Clustering Coefficient']:.3f}")
        with col3:
            st.metric("Avg Degree", f"{basic_stats['Average Degree']:.2f}")
            st.metric("Components", basic_stats["Connected Components"])

        # Node Distribution
        st.subheader("Node Type Distribution")
        node_types = {k: v for k, v in basic_stats.items() if "Count" in k}
        df_types = pd.DataFrame(list(node_types.items()),
                              columns=["Type", "Count"])
        st.bar_chart(df_types.set_index("Type"))

        # Top Interactions
        st.subheader("Strongest Interactions")
        top_interactions = self.get_top_interactions()
        for u, v, d in top_interactions:
            st.write(f"**{u}** → **{v}**: {d['relation']} (strength: {d['weight']:.2f})")

        # Important Nodes
        st.subheader("Key Nodes by Centrality")
        top_nodes = self.get_node_importance()
        tabs = st.tabs(["Degree", "Betweenness", "Closeness", "Eigenvector"])
        for tab, (metric, nodes) in zip(tabs, top_nodes.items()):
            with tab:
                for node, score in nodes:
                    node_type = self.G.nodes[node]["type"]
                    st.write(f"**{node}** ({node_type}): {score:.3f}")

        # Pathway Analysis
        st.subheader("Pathway Analysis")
        pathway_stats = self.get_pathway_analysis()
        for pathway, stats in pathway_stats.items():
            with st.expander(f"🔄 {pathway}"):
                st.write(f"**Total Connections:** {stats['Total Connections']}")
                st.write("**Connected to:**")
                for type_, count in stats['Connected Types'].items():
                    st.write(f"- {type_.capitalize()}: {count}")
                st.write(f"**Average Interaction Strength:** {stats['Average Interaction Strength']:.2f}")

    def _display_cluster_stats(self, cluster_name):
        """Display cluster-specific statistics"""
        stats = self.get_cluster_stats(cluster_name)
        if not stats:
            st.warning(f"No data available for cluster: {cluster_name}")
            return

        st.subheader(f"Analysis of {cluster_name} Cluster")
        
        # Basic cluster metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", stats["Nodes in Cluster"])
        with col2:
            st.metric("Edges", stats["Edges in Cluster"])
        with col3:
            st.metric("Density", f"{stats['Cluster Density']:.3f}")

        # Node composition
        st.subheader("Cluster Composition")
        node_types = {k: v for k, v in stats.items() if "Count" in k}
        df_types = pd.DataFrame(list(node_types.items()),
                              columns=["Type", "Count"])
        st.bar_chart(df_types.set_index("Type"))

        # Key nodes analysis
        st.subheader("Key Nodes Analysis")
        tabs = st.tabs(["Degree Centrality", "Betweenness Centrality", "Closeness Centrality"])
        for tab, metric in zip(tabs, stats["Key Nodes"].keys()):
            with tab:
                for node, score in stats["Key Nodes"][metric]:
                    node_type = self.G.nodes[node]["type"]
                    st.write(f"**{node}** ({node_type}): {score:.3f}")

    def _display_health_metrics(self):
        """Display network health metrics"""
        st.subheader("Network Health Metrics")
        total_possible_edges = len(self.nodes_data) * (len(self.nodes_data) - 1) / 2
        network_density = len(self.edges_data) / total_possible_edges
        avg_degree = np.mean([d for n, d in self.G.degree()])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Network Density", f"{network_density:.2%}")
        with col2:
            st.metric("Average Node Degree", f"{avg_degree:.2f}")
        with col3:
            clustering_coef = nx.average_clustering(self.G)
            st.metric("Clustering Coefficient", f"{clustering_coef:.2%}")

    def _display_temporal_analysis(self):
        """Display temporal analysis based on PMIDs"""
        st.subheader("Temporal Analysis")
        pmid_timeline = self.get_temporal_analysis()
        
        timeline_data = []
        for pmid, data in pmid_timeline.items():
            if pmid != "Unknown":
                timeline_data.append({
                    "PMID": pmid,
                    "Nodes": len(data["nodes"]),
                    "Edges": len(data["edges"])
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            try:
                timeline_df["PMID"] = pd.to_numeric(timeline_df["PMID"])
                timeline_df = timeline_df.sort_values("PMID")
            except:
                timeline_df = timeline_df.sort_values("PMID")
            
            # Display timeline charts
            st.write("Nodes per Publication:")
            st.bar_chart(data=timeline_df.set_index("PMID")["Nodes"])
            
            st.write("Edges per Publication:")
            st.bar_chart(data=timeline_df.set_index("PMID")["Edges"])
            
            # Statistics
            st.write("Publication Statistics:")
            stats_df = pd.DataFrame({
                "Metric": ["Total Publications", "Average Nodes/Publication", "Average Edges/Publication"],
                "Value": [
                    len(timeline_df),
                    timeline_df["Nodes"].mean(),
                    timeline_df["Edges"].mean()
                ]
            })
            st.dataframe(stats_df)
        else:
            st.write("No temporal data available")

    def _display_cluster_interactions(self):
        """Display analysis of cluster interactions"""
        st.subheader("Cluster Interactions")
        interactions = self.get_cluster_interactions()
        
        if interactions:
            interaction_data = []
            for (cluster1, cluster2), data in interactions.items():
                interaction_data.append({
                    "Cluster Pair": f"{cluster1} ↔ {cluster2}",
                    "Interaction Count": data["count"],
                    "Average Score": sum(edge["score"] for edge in data["edges"]) / len(data["edges"])
                })
            
            interaction_df = pd.DataFrame(interaction_data)
            st.write("Cross-cluster Interactions:")
            st.dataframe(interaction_df)
            
            # Visualization of cluster interactions
            st.bar_chart(data=interaction_df.set_index("Cluster Pair")["Interaction Count"])

    def _display_network_explorer(self):
        """Display interactive network explorer"""
        st.subheader("Interactive Network Explorer")
        selected_node = st.selectbox(
            "Select Node to Explore",
            options=sorted(list(self.node_ids))
        )
        
        if selected_node:
            node_data = next((n for n in self.nodes_data if n["id"] == selected_node), None)
            if node_data:
                # Display node details
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Node Details:")
                    for key, value in node_data.items():
                        st.write(f"- {key}: {value}")
                
                with col2:
                    # Find connected nodes
                    connections = []
                    for edge in self.edges_data:
                        if edge["source"] == selected_node:
                            connections.append({
                                "Connected To": edge["target"],
                                "Relation": edge["relation"],
                                "Score": edge["score"]
                            })
                        elif edge["target"] == selected_node:
                            connections.append({
                                "Connected To": edge["source"],
                                "Relation": edge["relation"],
                                "Score": edge["score"]
                            })
                    
                    st.write("Connected Nodes:")
                    if connections:
                        st.dataframe(pd.DataFrame(connections))
                    else:
                        st.write("No connections found")

    def get_hub_nodes(self, top_n=10):
        """Identify hub nodes based on degree centrality"""
        centrality = nx.degree_centrality(self.G)
        return dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def get_network_health_score(self):
        """Calculate overall network health score"""
        density = nx.density(self.G)
        avg_clustering = nx.average_clustering(self.G)
        avg_degree = np.mean([d for n, d in self.G.degree()])
        
        # Normalize metrics
        max_degree = len(self.G.nodes) - 1
        norm_degree = avg_degree / max_degree
        
        # Combine metrics (equal weights)
        health_score = (density + avg_clustering + norm_degree) / 3
        
        return {
            "overall_score": health_score,
            "density": density,
            "clustering": avg_clustering,
            "normalized_degree": norm_degree
        }

    def get_community_detection(self):
        """Detect communities using various algorithms"""
        communities = {
            "louvain": nx.community.louvain_communities(self.G),
            "greedy_modularity": nx.community.greedy_modularity_communities(self.G),
            "label_propagation": nx.community.label_propagation_communities(self.G)
        }
        
        return communities

    def get_network_robustness(self):
        """Analyze network robustness through various metrics"""
        robustness = {
            "avg_shortest_path": nx.average_shortest_path_length(self.G),
            "diameter": nx.diameter(self.G),
            "radius": nx.radius(self.G),
            "assortativity": nx.degree_assortativity_coefficient(self.G)
        }
        
        # Add node connectivity if computationally feasible
        if len(self.G.nodes) <= 1000:  # Only for smaller networks
            robustness["node_connectivity"] = nx.node_connectivity(self.G)
            robustness["edge_connectivity"] = nx.edge_connectivity(self.G)
            
        return robustness
