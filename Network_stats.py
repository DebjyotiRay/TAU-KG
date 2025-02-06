import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import streamlit as st
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

class NetworkAnalyzer:
    def __init__(self, nodes_data, edges_data):
        """Initialize NetworkAnalyzer with nodes and edges data"""
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        self.node_ids = {node["id"] for node in nodes_data}
        self.G = self._create_networkx_graph()
        self.layout = None  # Will store network layout for consistent visualization

    def _create_networkx_graph(self):
        """Create a NetworkX graph from nodes and edges data"""
        G = nx.Graph()
        
        # Add nodes with all attributes
        for node in self.nodes_data:
            G.add_node(node["id"], **node)

        # Add edges with all attributes
        for edge in self.edges_data:
            G.add_edge(edge["source"],
                      edge["target"],
                      weight=edge.get("score", 1.0),
                      **edge)

        return G

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

        # Path-based metrics only if graph is connected
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

        # Find top nodes for each centrality measure
        stats["Key Nodes"] = {}
        for measure_name, centrality_dict in centrality_measures.items():
            sorted_nodes = sorted(centrality_dict.items(),
                                key=lambda x: x[1],
                                reverse=True)[:5]  # Top 5 nodes
            stats["Key Nodes"][measure_name] = sorted_nodes

        return stats

    def get_pathway_analysis(self):
        """Analyze pathway connections and influences"""
        pathway_nodes = [n for n, d in self.G.nodes(data=True)
                        if d.get("type", "") == "pathway"]
        
        if not pathway_nodes:
            return {}  # Return empty dict if no pathway nodes found

        pathway_stats = {}
        for pathway in pathway_nodes:
            neighbors = list(self.G.neighbors(pathway))
            if not neighbors:
                continue  # Skip pathways with no connections
                
            neighbor_types = Counter([self.G.nodes[n].get("type", "Unknown") 
                                    for n in neighbors])
            neighbor_clusters = Counter([self.G.nodes[n].get("cluster", "Unknown") 
                                      for n in neighbors])

            # Calculate interaction strengths
            interaction_strengths = [d.get("weight", 0.0) 
                                   for u, v, d in self.G.edges(pathway, data=True)]
            
            if interaction_strengths:  # Only add stats if there are interactions
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

    def get_temporal_analysis(self):
        """Perform detailed temporal analysis based on PMIDs"""
        pmids = sorted(set(str(node.get("PMID", "Unknown")) for node in self.nodes_data))
        pmid_timeline = {pmid: {"nodes": [], "edges": [], "types": Counter(),
                               "clusters": Counter()} for pmid in pmids}
        
        # Process nodes
        for node in self.nodes_data:
            pmid = str(node.get("PMID", "Unknown"))
            pmid_timeline[pmid]["nodes"].append(node["id"])
            pmid_timeline[pmid]["types"][node["type"]] += 1
            pmid_timeline[pmid]["clusters"][node.get("cluster", "Unknown")] += 1
        
        # Process edges
        for edge in self.edges_data:
            source_pmid = str(next((n.get("PMID", "Unknown") 
                                  for n in self.nodes_data if n["id"] == edge["source"]),
                                  "Unknown"))
            pmid_timeline[source_pmid]["edges"].append(edge)

        return pmid_timeline

    def get_community_detection(self):
        """Detect and analyze communities using multiple algorithms"""
        communities = {
            "louvain": nx.community.louvain_communities(self.G),
            "greedy_modularity": nx.community.greedy_modularity_communities(self.G),
            "label_propagation": nx.community.label_propagation_communities(self.G)
        }
        
        # Calculate modularity for each method
        modularities = {}
        for method, comm_set in communities.items():
            modularities[method] = nx.community.modularity(self.G, comm_set)
            
        return communities, modularities

    def get_network_robustness(self):
        """Analyze network robustness through comprehensive metrics"""
        robustness = {
            "avg_shortest_path": nx.average_shortest_path_length(self.G),
            "diameter": nx.diameter(self.G),
            "radius": nx.radius(self.G),
            "assortativity": nx.degree_assortativity_coefficient(self.G),
            "algebraic_connectivity": nx.algebraic_connectivity(self.G),
            "edge_density": nx.density(self.G),
            "average_clustering": nx.average_clustering(self.G)
        }
        
        if len(self.G.nodes) <= 1000:
            robustness.update({
                "node_connectivity": nx.node_connectivity(self.G),
                "edge_connectivity": nx.edge_connectivity(self.G),
                "percolation_centrality": nx.percolation_centrality(self.G)
            })
            
            # Simulate network attacks
            robustness["attack_tolerance"] = self._simulate_attacks()
            
        return robustness

    def _simulate_attacks(self, n_iterations=10):
        """Simulate targeted and random attacks on the network"""
        results = {
            "random_attack": [],
            "targeted_attack": []
        }
        
        # Original network properties
        original_size = len(nx.largest_connected_component(self.G))
        
        # Random attack
        G_random = self.G.copy()
        nodes_random = list(G_random.nodes())
        np.random.shuffle(nodes_random)
        
        # Targeted attack (highest degree nodes)
        G_targeted = self.G.copy()
        nodes_targeted = sorted(G_targeted.degree(), key=lambda x: x[1], reverse=True)
        nodes_targeted = [n[0] for n in nodes_targeted]
        
        # Remove nodes and measure impact
        for i in range(min(n_iterations, len(self.G.nodes))):
            # Random attack
            G_random.remove_node(nodes_random[i])
            size_random = len(nx.largest_connected_component(G_random)) if nx.is_connected(G_random) else 0
            results["random_attack"].append(size_random / original_size)
            
            # Targeted attack
            G_targeted.remove_node(nodes_targeted[i])
            size_targeted = len(nx.largest_connected_component(G_targeted)) if nx.is_connected(G_targeted) else 0
            results["targeted_attack"].append(size_targeted / original_size)
            
        return results

    def get_motif_analysis(self, size=3):
        """Analyze network motifs and their significance"""
        motif_stats = defaultdict(int)
        significance_scores = {}
        
        # Find all subgraphs of given size
        for nodes in combinations(self.G.nodes(), size):
            subgraph = self.G.subgraph(nodes)
            motif_type = self._classify_motif(subgraph)
            motif_stats[motif_type] += 1
            
        # Calculate motif significance through random network comparison
        random_graphs = [nx.fast_gnp_random_graph(len(self.G), nx.density(self.G)) 
                        for _ in range(10)]
        
        random_motifs = defaultdict(list)
        for random_G in random_graphs:
            for nodes in combinations(random_G.nodes(), size):
                subgraph = random_G.subgraph(nodes)
                motif_type = self._classify_motif(subgraph)
                random_motifs[motif_type].append(1)
                
        # Calculate z-scores
        for motif_type, count in motif_stats.items():
            if motif_type in random_motifs:
                random_mean = np.mean(random_motifs[motif_type])
                random_std = np.std(random_motifs[motif_type]) or 1
                significance_scores[motif_type] = (count - random_mean) / random_std
            
        return dict(motif_stats), significance_scores

    def _classify_motif(self, subgraph):
        """Classify network motifs based on their structure"""
        n_nodes = len(subgraph)
        n_edges = len(subgraph.edges())
        
        if n_nodes == 3:
            if n_edges == 2:
                return "Chain"
            elif n_edges == 3:
                return "Triangle"
        elif n_nodes == 4:
            if n_edges == 3:
                return "Star"
            elif n_edges == 4:
                return "Square"
            elif n_edges == 6:
                return "Complete"
            
        return f"Other_{n_nodes}_{n_edges}"

    def get_correlation_analysis(self):
        """Analyze correlations between different network metrics"""
        node_metrics = {
            "Degree": dict(self.G.degree()),
            "Betweenness": nx.betweenness_centrality(self.G),
            "Closeness": nx.closeness_centrality(self.G),
            "Clustering": nx.clustering(self.G)
        }
        
        correlations = {}
        for metric1, metric2 in combinations(node_metrics.keys(), 2):
            values1 = list(node_metrics[metric1].values())
            values2 = list(node_metrics[metric2].values())
            corr, p_value = pearsonr(values1, values2)
            correlations[f"{metric1}-{metric2}"] = {
                "correlation": corr,
                "p_value": p_value
            }
            
        return correlations

    def display_stats_streamlit(self, selected_cluster=None):
        """Display comprehensive network statistics in Streamlit"""
        # Create tabs for different analysis types
        tabs = st.tabs(["Basic Analysis", "Advanced Analysis", "Network Explorer"])
        
        with tabs[0]:  # Basic Analysis
            if selected_cluster and selected_cluster != "All":
                self._display_cluster_stats(selected_cluster)
            else:
                self._display_overall_stats()

        with tabs[1]:  # Advanced Analysis
            self._display_advanced_analysis()

        with tabs[2]:  # Network Explorer
            self._display_interactive_explorer()

    def _display_overall_stats(self):
        """Display overall network statistics"""
        st.header("Network Overview")
        basic_stats = self.get_basic_stats()

        # Show component information first if graph is disconnected
        if basic_stats["Connected Components"] > 1:
            st.warning(f"Network is disconnected with {basic_stats['Connected Components']} components. " +
                      f"Largest component contains {basic_stats['Largest Component Ratio']:.1%} of nodes.")

        # Basic metrics in three columns
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

        # Path-based metrics
        st.subheader("Path-based Metrics (Largest Component)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Path Length", f"{basic_stats['Average Path Length']:.2f}")
        with col2:
            st.metric("Network Diameter", basic_stats["Graph Diameter"])

        # Node Distribution
        st.subheader("Node Type Distribution")
        node_types = {k: v for k, v in basic_stats.items() if "Count" in k}
        df_types = pd.DataFrame(list(node_types.items()),
                              columns=["Type", "Count"])
        st.bar_chart(df_types.set_index("Type"))

        # Pathway Analysis (Collapsible)
        with st.expander("🔄 Pathway Analysis", expanded=False):
            pathway_stats = self.get_pathway_analysis()
            if pathway_stats:
                for pathway, stats in pathway_stats.items():
                    st.write(f"### {pathway}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Connections", stats['Total Connections'])
                    with col2:
                        st.metric("Avg Interaction Strength", 
                                f"{stats['Average Interaction Strength']:.2f}")
                    
                    # Show connected types as a bar chart
                    if stats['Connected Types']:
                        connected_types_df = pd.DataFrame(
                            list(stats['Connected Types'].items()),
                            columns=['Type', 'Count']
                        )
                        st.write("Connected Node Types:")
                        st.bar_chart(connected_types_df.set_index('Type'))
                    
                    # Show cluster distribution
                    if stats['Connected Clusters']:
                        cluster_df = pd.DataFrame(
                            list(stats['Connected Clusters'].items()),
                            columns=['Cluster', 'Count']
                        )
                        st.write("Cluster Distribution:")
                        st.bar_chart(cluster_df.set_index('Cluster'))
            else:
                st.info("No pathway data available in the network")

        # Pathway Analysis (Collapsible)
        with st.expander("🔄 Pathway Analysis", expanded=False):
            pathway_stats = self.get_pathway_analysis()
            if pathway_stats:
                for pathway, stats in pathway_stats.items():
                    st.write(f"### {pathway}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Connections", stats['Total Connections'])
                    with col2:
                        st.metric("Avg Interaction Strength", 
                                f"{stats['Average Interaction Strength']:.2f}")
                    
                    # Show connected types as a bar chart
                    connected_types_df = pd.DataFrame(
                        list(stats['Connected Types'].items()),
                        columns=['Type', 'Count']
                    )
                    st.write("Connected Node Types:")
                    st.bar_chart(connected_types_df.set_index('Type'))
                    
                    # Show cluster distribution
                    cluster_df = pd.DataFrame(
                        list(stats['Connected Clusters'].items()),
                        columns=['Cluster', 'Count']
                    )
                    st.write("Cluster Distribution:")
                    st.bar_chart(cluster_df.set_index('Cluster'))
            else:
                st.info("No pathway data available in the network")

    def _display_advanced_analysis(self):
        """Display comprehensive advanced network analysis"""
        # Sidebar for analysis selection
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Network Health", "Community Structure", "Robustness Analysis",
             "Motif Analysis", "Correlation Analysis", "Temporal Patterns"]
        )

        if analysis_type == "Network Health":
            self._display_health_analysis()
        elif analysis_type == "Community Structure":
            self._display_community_analysis()
        elif analysis_type == "Robustness Analysis":
            self._display_robustness_analysis()
        elif analysis_type == "Motif Analysis":
            self._display_motif_analysis()
        elif analysis_type == "Correlation Analysis":
            self._display_correlation_analysis()
        elif analysis_type == "Temporal Patterns":
            self._display_temporal_analysis()

    def _display_health_analysis(self):
        """Display detailed network health analysis"""
        st.header("Network Health Analysis")
        
        # Calculate health metrics
        basic_stats = self.get_basic_stats()
        
        # Overview metrics
        st.subheader("Core Health Indicators")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Network Density", f"{basic_stats['Network Density']:.3f}")
        with col2:
            st.metric("Clustering Coefficient", 
                     f"{basic_stats['Average Clustering Coefficient']:.3f}")
        with col3:
            st.metric("Average Path Length",
                     f"{basic_stats['Average Path Length']:.2f}")

        # Degree Distribution Analysis
        st.subheader("Degree Distribution")
        degrees = [d for n, d in self.G.degree()]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=degrees, nbinsx=20))
        fig.update_layout(title="Node Degree Distribution",
                         xaxis_title="Degree",
                         yaxis_title="Count")
        st.plotly_chart(fig)
        
        # Network Centralization
        centrality_metrics = {
            "Degree": nx.degree_centrality(self.G),
            "Betweenness": nx.betweenness_centrality(self.G),
            "Closeness": nx.closeness_centrality(self.G)
        }
        
        st.subheader("Centralization Analysis")
        centrality_df = pd.DataFrame({
            metric: list(values.values())
            for metric, values in centrality_metrics.items()
        })
        
        fig = go.Figure()
        for metric in centrality_df.columns:
            fig.add_trace(go.Box(y=centrality_df[metric], name=metric))
        fig.update_layout(title="Centrality Distributions",
                         yaxis_title="Centrality Value")
        st.plotly_chart(fig)

    def _display_community_analysis(self):
        """Display comprehensive community structure analysis"""
        st.header("Community Structure Analysis")
        
        # Get communities and modularity scores
        communities, modularities = self.get_community_detection()
        
        # Overview of detection methods
        st.subheader("Community Detection Results")
        for method, comm_set in communities.items():
            st.write(f"### {method.replace('_', ' ').title()}")
            
            # Basic statistics
            n_communities = len(comm_set)
            avg_size = np.mean([len(c) for c in comm_set])
            modularity = modularities[method]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Communities", n_communities)
            with col2:
                st.metric("Average Size", f"{avg_size:.1f}")
            with col3:
                st.metric("Modularity", f"{modularity:.3f}")
            
            # Size distribution
            sizes = [len(c) for c in comm_set]
            fig = go.Figure(data=[
                go.Bar(x=list(range(1, len(sizes) + 1)), y=sorted(sizes, reverse=True))
            ])
            fig.update_layout(title="Community Size Distribution",
                            xaxis_title="Community Index",
                            yaxis_title="Size")
            st.plotly_chart(fig)
            
            # Composition analysis of largest communities
            st.write("### Top Communities Composition")
            for i, comm in enumerate(sorted(comm_set, key=len, reverse=True)[:3], 1):
                with st.expander(f"Community {i} - {len(comm)} nodes"):
                    # Node type distribution
                    type_counts = Counter(self.G.nodes[n]["type"] for n in comm)
                    type_df = pd.DataFrame(list(type_counts.items()),
                                         columns=["Type", "Count"])
                    st.write("Node Type Distribution:")
                    st.bar_chart(type_df.set_index("Type"))
                    
                    # Internal vs external connections
                    subgraph = self.G.subgraph(comm)
                    internal_edges = len(subgraph.edges())
                    external_edges = sum(1 for n in comm 
                                       for neighbor in self.G.neighbors(n)
                                       if neighbor not in comm)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Internal Edges", internal_edges)
                    with col2:
                        st.metric("External Edges", external_edges)

    def _display_robustness_analysis(self):
        """Display comprehensive robustness analysis"""
        st.header("Network Robustness Analysis")
        
        # Get robustness metrics
        robustness = self.get_network_robustness()
        
        # Basic metrics
        st.subheader("Core Robustness Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Path Length", 
                     f"{robustness['avg_shortest_path']:.2f}")
        with col2:
            st.metric("Network Diameter", robustness['diameter'])
        with col3:
            st.metric("Algebraic Connectivity",
                     f"{robustness['algebraic_connectivity']:.4f}")
            
        # Advanced metrics if available
        if 'node_connectivity' in robustness:
            st.subheader("Connectivity Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Node Connectivity", robustness['node_connectivity'])
            with col2:
                st.metric("Edge Connectivity", robustness['edge_connectivity'])
        
        # Attack simulation results
        if 'attack_tolerance' in robustness:
            st.subheader("Network Attack Tolerance")
            
            # Create comparison plot
            attack_data = robustness['attack_tolerance']
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=attack_data['random_attack'],
                name='Random Attack',
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                y=attack_data['targeted_attack'],
                name='Targeted Attack',
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title="Network Degradation Under Attacks",
                xaxis_title="Number of Nodes Removed",
                yaxis_title="Largest Component Size Ratio",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig)
            
            # Analysis interpretation
            st.write("### Attack Tolerance Interpretation")
            random_resilience = np.mean(attack_data['random_attack'])
            targeted_resilience = np.mean(attack_data['targeted_attack'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Random Attack Resilience", f"{random_resilience:.2%}")
            with col2:
                st.metric("Targeted Attack Resilience", f"{targeted_resilience:.2%}")

    def _display_motif_analysis(self):
        """Display comprehensive motif analysis"""
        st.header("Network Motif Analysis")
        
        # Analyze motifs of size 3 and 4
        motifs_3, significance_3 = self.get_motif_analysis(size=3)
        motifs_4, significance_4 = self.get_motif_analysis(size=4)
        
        # Display results for 3-node motifs
        st.subheader("3-Node Motifs")
        motifs_3_df = pd.DataFrame({
            'Motif Type': list(motifs_3.keys()),
            'Count': list(motifs_3.values()),
            'Z-score': [significance_3.get(k, 0) for k in motifs_3.keys()]
        })
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Motif Counts", "Statistical Significance"))
        
        fig.add_trace(
            go.Bar(x=motifs_3_df['Motif Type'], 
                  y=motifs_3_df['Count'],
                  name="Count"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=motifs_3_df['Motif Type'],
                  y=motifs_3_df['Z-score'],
                  name="Z-score"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="3-Node Motif Analysis")
        st.plotly_chart(fig)
        
        # Display results for 4-node motifs
        st.subheader("4-Node Motifs")
        motifs_4_df = pd.DataFrame({
            'Motif Type': list(motifs_4.keys()),
            'Count': list(motifs_4.values()),
            'Z-score': [significance_4.get(k, 0) for k in motifs_4.keys()]
        })
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Motif Counts", "Statistical Significance"))
        
        fig.add_trace(
            go.Bar(x=motifs_4_df['Motif Type'],
                  y=motifs_4_df['Count'],
                  name="Count"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=motifs_4_df['Motif Type'],
                  y=motifs_4_df['Z-score'],
                  name="Z-score"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="4-Node Motif Analysis")
        st.plotly_chart(fig)
        
        # Interpretation
        st.write("### Motif Analysis Interpretation")
        # Most significant motifs
        significant_motifs = pd.concat([motifs_3_df, motifs_4_df])
        significant_motifs = significant_motifs.nlargest(3, 'Z-score')
        
        st.write("Most Significant Network Motifs:")
        for _, row in significant_motifs.iterrows():
            st.write(f"- {row['Motif Type']}: Count = {row['Count']}, "
                    f"Z-score = {row['Z-score']:.2f}")

    def _display_correlation_analysis(self):
        """Display correlation analysis between network metrics"""
        st.header("Network Metric Correlations")
        
        # Get correlation data
        correlations = self.get_correlation_analysis()
        
        # Create correlation matrix visualization
        correlation_data = []
        metrics = ["Degree", "Betweenness", "Closeness", "Clustering"]
        
        for i, metric1 in enumerate(metrics):
            row = []
            for j, metric2 in enumerate(metrics):
                if i == j:
                    row.append(1.0)
                elif i < j:
                    key = f"{metric1}-{metric2}"
                    row.append(correlations[key]["correlation"])
                else:
                    key = f"{metric2}-{metric1}"
                    row.append(correlations[key]["correlation"])
            correlation_data.append(row)
            
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data,
            x=metrics,
            y=metrics,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Correlation Matrix of Network Metrics",
            width=600,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Detailed correlation analysis
        st.subheader("Detailed Correlation Analysis")
        for pair, data in correlations.items():
            metric1, metric2 = pair.split("-")
            st.write(f"### {metric1} vs {metric2}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation", f"{data['correlation']:.3f}")
            with col2:
                st.metric("P-value", f"{data['p_value']:.3e}")

    def _display_temporal_analysis(self):
        """Display temporal analysis of the network"""
        st.header("Temporal Network Analysis")
        
        temporal_data = self.get_temporal_analysis()
        
        # Filter out Unknown PMID
        temporal_data = {k: v for k, v in temporal_data.items() if k != "Unknown"}
        
        if not temporal_data:
            st.warning("No temporal data available")
            return
            
        # Create timeline visualization
        timeline_df = pd.DataFrame([
            {
                "PMID": pmid,
                "Nodes": len(data["nodes"]),
                "Edges": len(data["edges"]),
                "Types": len(data["types"]),
                "Clusters": len(data["clusters"])
            }
            for pmid, data in temporal_data.items()
        ])
        
        # Sort by PMID
        timeline_df = timeline_df.sort_values("PMID")
        
        # Network growth over time
        st.subheader("Network Growth")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timeline_df["PMID"],
            y=timeline_df["Nodes"].cumsum(),
            name="Nodes",
            mode="lines+markers"
        ))
        
        fig.add_trace(go.Scatter(
            x=timeline_df["PMID"],
            y=timeline_df["Edges"].cumsum(),
            name="Edges",
            mode="lines+markers"
        ))
        
        fig.update_layout(
            title="Cumulative Network Growth",
            xaxis_title="PMID",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)
        
        # Node type evolution
        st.subheader("Node Type Evolution")
        type_evolution = {}
        for pmid, data in temporal_data.items():
            for node_type, count in data["types"].items():
                if node_type not in type_evolution:
                    type_evolution[node_type] = []
                type_evolution[node_type].append(count)
                
        fig = go.Figure()
        for node_type, counts in type_evolution.items():
            fig.add_trace(go.Scatter(
                x=timeline_df["PMID"],
                y=np.cumsum(counts),
                name=node_type,
                mode="lines+markers"
            ))
            
        fig.update_layout(
            title="Evolution of Node Types",
            xaxis_title="PMID",
            yaxis_title="Cumulative Count"
        )
        st.plotly_chart(fig)
        
        # Network density evolution
        densities = []
        avg_degrees = []
        clustering_coeffs = []
        
        for pmid, data in temporal_data.items():
            nodes = data["nodes"]
            edges = data["edges"]
            if nodes:
                subgraph = self.G.subgraph(nodes)
                densities.append(nx.density(subgraph))
                avg_degrees.append(np.mean([d for n, d in subgraph.degree()]))
                clustering_coeffs.append(nx.average_clustering(subgraph))
                
        # Plot network metrics evolution
        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=("Density", "Average Degree", 
                                         "Clustering Coefficient"))
        
        fig.add_trace(
            go.Scatter(x=timeline_df["PMID"], y=densities,
                      mode="lines+markers"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timeline_df["PMID"], y=avg_degrees,
                      mode="lines+markers"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=timeline_df["PMID"], y=clustering_coeffs,
                      mode="lines+markers"),
            row=1, col=3
        )
        
        fig.update_layout(height=400, title_text="Evolution of Network Metrics")
        st.plotly_chart(fig)

    def _display_interactive_explorer(self):
        """Display interactive network explorer"""
        st.header("Interactive Network Explorer")
        
        # Node selection and filtering
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Node selection
            selected_node = st.selectbox(
                "Select Node to Explore",
                options=sorted(list(self.node_ids))
            )
            
            # Filter options
            st.write("### Filter Options")
            selected_types = st.multiselect(
                "Node Types",
                options=sorted(set(data["type"] 
                                 for n, data in self.G.nodes(data=True)))
            )
            
            min_degree = st.slider(
                "Minimum Degree",
                min_value=1,
                max_value=max(dict(self.G.degree()).values()),
                value=1
            )
            
            min_weight = st.slider(
                "Minimum Edge Weight",
                min_value=0.0,
                max_value=max(d["weight"] for u, v, d in self.G.edges(data=True)),
                value=0.0
            )
        
        with col2:
            if selected_node:
                # Display node details
                node_data = self.G.nodes[selected_node]
                st.write("### Node Details")
                for key, value in node_data.items():
                    st.write(f"**{key}:** {value}")
                
                # Node neighborhood analysis
                st.write("### Neighborhood Analysis")
                
                neighbors = list(self.G.neighbors(selected_node))
                neighbor_types = Counter([self.G.nodes[n]["type"] 
                                       for n in neighbors])
                
                # Neighbor type distribution
                fig = go.Figure(data=[
                    go.Pie(labels=list(neighbor_types.keys()),
                          values=list(neighbor_types.values()))
                ])
                fig.update_layout(title="Neighbor Type Distribution")
                st.plotly_chart(fig)
                
                # Edge weight distribution
                edge_weights = [d["weight"] 
                              for u, v, d in self.G.edges(selected_node, data=True)]
                
                fig = go.Figure(data=[
                    go.Histogram(x=edge_weights, nbinsx=20)
                ])
                fig.update_layout(
                    title="Edge Weight Distribution",
                    xaxis_title="Weight",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)
                
                # Connected nodes table
                st.write("### Connected Nodes")
                connections = []
                for neighbor in neighbors:
                    edge_data = self.G.edges[selected_node, neighbor]
                    connections.append({
                        "Node": neighbor,
                        "Type": self.G.nodes[neighbor]["type"],
                        "Weight": edge_data["weight"],
                        "Relation": edge_data.get("relation", "Unknown")
                    })
                
                connections_df = pd.DataFrame(connections)
                st.dataframe(connections_df)
