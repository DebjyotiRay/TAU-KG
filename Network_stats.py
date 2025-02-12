import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import streamlit as st
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

from Network_explorer import *
from Advanced_analyzer import *
from Basic_analyzer import *
class NetworkAnalyzer:
    def __init__(self, nodes_data, edges_data):
        """Initialize NetworkAnalyzer with nodes and edges data"""
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        self.node_ids = {node["id"] for node in nodes_data}
        self.G = self._create_networkx_graph()
        
        # Initialize analyzers
        self.basic_analyzer = BasicAnalyzer(self.G)
        self.advanced_analyzer = AdvancedAnalyzer(self.G)
        self.network_explorer = NetworkExplorer(self.G)
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

    # Basic Analysis Methods
    def get_basic_stats(self):
        """Get comprehensive basic network statistics"""
        return self.basic_analyzer.get_basic_stats()

    def get_cluster_stats(self, cluster_name):
        """Get detailed statistics for a specific cluster"""
        return self.basic_analyzer.get_cluster_stats(cluster_name)

    def get_pathway_analysis(self):
        """Get pathway connections and influences analysis"""
        return self.basic_analyzer.get_pathway_analysis()

    # Advanced Analysis Methods
    def get_community_detection(self):
        """Get community detection analysis"""
        return self.advanced_analyzer.get_community_detection()

    def get_network_robustness(self):
        """Get network robustness analysis"""
        return self.advanced_analyzer.get_network_robustness()

    def get_motif_analysis(self, size=3):
        """Get network motif analysis"""
        return self.advanced_analyzer.get_motif_analysis(size)

    def get_correlation_analysis(self):
        """Get correlation analysis between network metrics"""
        return self.advanced_analyzer.get_correlation_analysis()

    def get_temporal_analysis(self):
        """Get temporal analysis based on PMIDs"""
        return self.advanced_analyzer.get_temporal_analysis()

    # Network Explorer Methods
    def get_node_details(self, node_id):
        """Get detailed information about a specific node"""
        return self.network_explorer.get_node_details(node_id)

    def get_filtered_view(self, node_types=None, min_degree=1, min_weight=0.0):
        """Get filtered view of the network"""
        return self.network_explorer.get_filtered_view(node_types, min_degree, min_weight)
    # Network Explorer Methods
    def get_cluster_interactions(self):
        return self.network_explorer.cluster_interaction_analysis()

    # def get_paper_distribution(self):
    #     return self.network_explorer.paper_distribution_analysis()
    # In Network_stats.py, modify the PMID handling in get_paper_distribution method:

    def get_paper_distribution(self):
        """Get paper distribution analysis"""
        pmid_counts = defaultdict(int)
        for node in self.nodes_data:
            pmid = str(node.get('PMID', 'Unknown'))
            pmid_counts[pmid] += 1
        
        # Convert to analyzable format
        paper_stats = []
        for pmid, count in pmid_counts.items():
            nodes = [node for node in self.nodes_data if str(node.get('PMID', 'Unknown')) == pmid]
            edges = [edge for edge in self.edges_data if str(self.G.nodes[edge['source']].get('PMID', 'Unknown')) == pmid]
            
            paper_stats.append({
                'PMID': pmid,
                'nodes': count,
                'edges': len(edges),
                'node_types': Counter(node['type'] for node in nodes)
            })
        
        return paper_stats
    def get_filtered_network(self, node_types=None, min_degree=1, min_weight=0.0):
        return self.network_explorer.get_filtered_view(node_types, min_degree, min_weight)

    def explore_node_details(self, node_id):
        return self.network_explorer.explore_node(node_id)
    def get_network_entropy(self):
        """
        Calculate network entropy metrics
        
        Returns:
            dict: Network entropy measures and visualizations
        """
        try:
            # Calculate degree entropy
            degree_counts = Counter(dict(self.G.degree()).values())
            total_degrees = sum(degree_counts.values())
            degree_probs = [count/total_degrees for count in degree_counts.values()]
            degree_entropy = -sum(p * np.log2(p) for p in degree_probs if p > 0)
            
            # Calculate type entropy
            type_counts = Counter(nx.get_node_attributes(self.G, 'type').values())
            total_types = sum(type_counts.values())
            type_probs = [count/total_types for count in type_counts.values()]
            type_entropy = -sum(p * np.log2(p) for p in type_probs if p > 0)
            
            # Calculate cluster entropy
            cluster_counts = Counter(nx.get_node_attributes(self.G, 'cluster').values())
            total_clusters = sum(cluster_counts.values())
            cluster_probs = [count/total_clusters for count in cluster_counts.values()]
            cluster_entropy = -sum(p * np.log2(p) for p in cluster_probs if p > 0)
            
            # Create visualizations
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=('Degree Distribution',
                                             'Type Distribution',
                                             'Cluster Distribution',
                                             'Entropy Comparison'))
            
            # Add distribution plots
            fig.add_trace(
                go.Bar(x=list(degree_counts.keys()),
                      y=list(degree_counts.values()),
                      name='Degree Dist'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=list(type_counts.keys()),
                      y=list(type_counts.values()),
                      name='Type Dist'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=list(cluster_counts.keys()),
                      y=list(cluster_counts.values()),
                      name='Cluster Dist'),
                row=2, col=1
            )
            
            # Add entropy comparison
            fig.add_trace(
                go.Bar(x=['Degree', 'Type', 'Cluster'],
                      y=[degree_entropy, type_entropy, cluster_entropy],
                      name='Entropy Values'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True,
                             title_text='Network Entropy Analysis')
            
            return {
                'entropy_measures': {
                    'degree_entropy': degree_entropy,
                    'type_entropy': type_entropy,
                    'cluster_entropy': cluster_entropy
                },
                'distributions': {
                    'degree': dict(degree_counts),
                    'type': dict(type_counts),
                    'cluster': dict(cluster_counts)
                },
                'visualization': fig
            }
            
        except Exception as e:
            self.logger.error(f"Network entropy calculation failed: {str(e)}")
            return None
    def get_network_overview(self):
        return self.network_explorer.get_network_summary()
    def calculate_publication_similarity(self, pub1_nodes, pub2_nodes, min_shared=1, min_weight=0.1):
        """Calculate similarity between two publications based on shared entities"""
        shared_types = set(n1['type'] for n1 in pub1_nodes) & set(n2['type'] for n2 in pub2_nodes)
        if not shared_types:
            return 0.0
            
        similarity_score = 0
        for entity_type in shared_types:
            pub1_entities = set(n['id'] for n in pub1_nodes if n['type'] == entity_type)
            pub2_entities = set(n['id'] for n in pub2_nodes if n['type'] == entity_type)
            shared_entities = pub1_entities & pub2_entities
            if len(shared_entities) >= min_shared:
                similarity_score += len(shared_entities) / max(len(pub1_entities), len(pub2_entities))
                
        return similarity_score / len(shared_types) if shared_types else 0.0
    
    def create_publication_network_plot(self, connections, publication_groups):
        """Create an interactive network visualization of publication relationships"""
        # Create nodes
        nodes = list(set(pmid for conn in connections for pmid in conn[:2]))
        node_sizes = [len(publication_groups[pmid]) for pmid in nodes]
        
        # Create the network plot using plotly
        fig = go.Figure()
        
        # Add edges (connections)
        edge_x, edge_y = [], []
        edge_colors = []
        for source, target, weight in connections:
            source_idx = nodes.index(source)
            target_idx = nodes.index(target)
            edge_x.extend([source_idx, target_idx, None])
            edge_y.extend([0, 0, None])
            edge_colors.extend([weight, weight, weight])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=edge_colors),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=list(range(len(nodes))),
            y=[0] * len(nodes),
            mode='markers+text',
            marker=dict(
                size=[s * 10 for s in node_sizes],
                color='lightblue',
                line=dict(width=2)
            ),
            text=nodes,
            hovertext=[f"PMID: {pmid}<br>Entities: {len(publication_groups[pmid])}" 
                      for pmid in nodes],
            textposition="top center"
        ))
        
        fig.update_layout(
            title="Publication Relationship Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    # Streamlit Display Methods
    def display_stats_streamlit(self, selected_cluster=None):
        """Display comprehensive network statistics in Streamlit"""
        st.set_page_config(layout="wide")
        
        # Create tabs for different analysis types
        tabs = st.tabs(["Basic Analysis", "Advanced Analysis", "Network Explorer"])
        
        with tabs[0]:  # Basic Analysis
            self._display_basic_analysis(selected_cluster)

        with tabs[1]:  # Advanced Analysis
            self._display_advanced_analysis()

        with tabs[2]:  # Network Explorer
            self._display_network_explorer()

    def _display_basic_analysis(self, selected_cluster=None):
        """Display basic network analysis in Streamlit"""
        if selected_cluster and selected_cluster != "All":
            cluster_stats = self.get_cluster_stats(selected_cluster)
            if cluster_stats:
                self._display_cluster_details(cluster_stats, selected_cluster)
        else:
            basic_stats = self.get_basic_stats()
            self._display_network_overview(basic_stats)
            self._display_paper_distribution(basic_stats)
            self._display_cluster_interactions(basic_stats)
            self._display_pathway_analysis()

    def get_pmid_distribution(self):
        """Analyze PMID distribution across nodes."""
        pmid_stats = {}
        for node in self.nodes_data:
            pmid = node.get("PMID", "Unknown")
            if pmid not in pmid_stats:
                pmid_stats[pmid] = {"count": 0, "nodes": []}
            pmid_stats[pmid]["count"] += 1
            pmid_stats[pmid]["nodes"].append(node["id"])
        return pmid_stats
    
    def get_hub_nodes(self, top_n=10):
        """Identify hub nodes based on connectivity."""
        centrality = self.get_node_centrality()
        return dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
    def _display_network_overview(self, basic_stats):
        """Display network overview statistics"""
        st.header("Network Overview")

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
        st.subheader("Path-based Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Path Length", f"{basic_stats['Average Path Length']:.2f}")
        with col2:
            st.metric("Network Diameter", basic_stats["Graph Diameter"])

    def _display_paper_distribution(self, basic_stats):
        """Display distribution of nodes and edges across papers"""
        st.header("Paper-wise Distribution")
        
        paper_stats = pd.DataFrame.from_dict(basic_stats["Paper Distribution"], 
                                           orient='index')
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Nodes per Paper", "Edges per Paper"))
        
        fig.add_trace(
            go.Bar(x=paper_stats.index, y=paper_stats["nodes"], name="Nodes"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=paper_stats.index, y=paper_stats["edges"], name="Edges"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Distribution across Papers")
        st.plotly_chart(fig)

    def _display_cluster_interactions(self, basic_stats):
        """Display cluster interaction heatmap"""
        st.header("Cluster Interactions")
        
        cluster_matrix = pd.DataFrame(basic_stats["Cluster Interactions"]).fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=cluster_matrix.values,
            x=cluster_matrix.columns,
            y=cluster_matrix.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Cluster Interaction Heatmap",
            xaxis_title="Target Cluster",
            yaxis_title="Source Cluster"
        )
        
        st.plotly_chart(fig)

    def _display_pathway_analysis(self):
        """Display pathway analysis results"""
        pathway_stats = self.get_pathway_analysis()
        
        if pathway_stats:
            st.header("Pathway Analysis")
            
            for pathway, stats in pathway_stats.items():
                with st.expander(f"Pathway: {pathway}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Connections", stats["Total Connections"])
                        st.metric("Max Interaction Strength", 
                                f"{stats['Max Interaction Strength']:.2f}")
                    
                    with col2:
                        st.metric("Average Interaction", 
                                f"{stats['Average Interaction Strength']:.2f}")
                        
                    # Connected Types Distribution
                    st.subheader("Connected Node Types")
                    type_df = pd.DataFrame(list(stats["Connected Types"].items()),
                                         columns=["Type", "Count"])
                    st.bar_chart(type_df.set_index("Type"))
                    
                    # Connected Clusters Distribution
                    st.subheader("Connected Clusters")
                    cluster_df = pd.DataFrame(list(stats["Connected Clusters"].items()),
                                            columns=["Cluster", "Count"])
                    st.bar_chart(cluster_df.set_index("Cluster"))

    def _display_advanced_analysis(self):
        """Display advanced network analysis"""
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Community Structure", "Network Robustness", "Motif Analysis",
             "Temporal Analysis", "Correlation Analysis"]
        )

        if analysis_type == "Community Structure":
            self._display_community_analysis()
        elif analysis_type == "Network Robustness":
            self._display_robustness_analysis()
        elif analysis_type == "Motif Analysis":
            self._display_motif_analysis()
        elif analysis_type == "Temporal Analysis":
            self._display_temporal_analysis()
        elif analysis_type == "Correlation Analysis":
            self._display_correlation_analysis()

    def _display_network_explorer(self):
        """Display interactive network explorer"""
        st.header("Network Explorer")
        
        # Node selection and filtering
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_node = st.selectbox(
                "Select Node to Explore",
                options=sorted(list(self.node_ids))
            )
            
            node_types = st.multiselect(
                "Filter by Node Types",
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
                node_details = self.get_node_details(selected_node)
                self._display_node_details(node_details)

            filtered_graph, filtered_edges = self.get_filtered_view(
                node_types, min_degree, min_weight)
            self._display_filtered_network(filtered_graph, filtered_edges)

    def _display_node_details(self, node_details):
        """Display detailed node information"""
        if not node_details:
            return

        st.subheader("Node Details")
        
        # Basic node information
        for key, value in node_details["Node Data"].items():
            st.write(f"**{key}:** {value}")
        
        # Connectivity metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Degree", node_details["Degree"])
            st.metric("Number of Neighbors", node_details["Neighbors"])
        with col2:
            st.metric("Local Clustering", f"{node_details['Local Clustering']:.3f}")
        
        # Centrality metrics
        st.subheader("Centrality Measures")
        centrality_df = pd.DataFrame(node_details["Centrality"], index=[0])
        st.dataframe(centrality_df)
        
        # Neighbor type distribution
        st.subheader("Neighbor Type Distribution")
        neighbor_df = pd.DataFrame(list(node_details["Neighbor Types"].items()),
                                 columns=["Type", "Count"])
        st.bar_chart(neighbor_df.set_index("Type"))
        
        # Edge weight distribution
        st.subheader("Edge Weight Distribution")
        fig = go.Figure(data=[go.Histogram(x=node_details["Edge Weights"])])
        fig.update_layout(title="Edge Weight Distribution",
                         xaxis_title="Weight",
                         yaxis_title="Count")
        st.plotly_chart(fig)

    def _display_filtered_network(self, filtered_graph, filtered_edges):
        """Display filtered network visualization"""
        st.subheader("Filtered Network View")
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", len(filtered_graph.nodes))
        with col2:
            st.metric("Edges", len(filtered_edges))
        with col3:
            st.metric("Density", f"{nx.density(filtered_graph):.3f}")
        
        # Node type distribution
        node_types = [data["type"] for n, data in filtered_graph.nodes(data=True)]
        type_counts = Counter(node_types)
        
        st.subheader("Node Type Distribution")
        type_df = pd.DataFrame(list(type_counts.items()),
                             columns=["Type", "Count"])
        st.bar_chart(type_df.set_index("Type"))
        
    def validate_graph_data(self):
        """
        Comprehensive graph data validation
        
        Returns:
            dict: Validation results with potential warnings and errors
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check node attributes
        required_node_attrs = ["id", "type", "cluster", "size"]
        for node in self.nodes_data:
            missing_attrs = [attr for attr in required_node_attrs if attr not in node]
            if missing_attrs:
                validation_results["warnings"].append(
                    f"Node {node.get('id', 'Unknown')} missing attributes: {missing_attrs}"
                )
        
        # Check edge validity
        node_ids = {node["id"] for node in self.nodes_data}
        invalid_edges = [
            edge for edge in self.edges_data 
            if edge["source"] not in node_ids or edge["target"] not in node_ids
        ]
        
        if invalid_edges:
            validation_results["warnings"].append(
                f"Found {len(invalid_edges)} edges with invalid node references"
            )
        
        # Check for isolated components
        connected_components = list(nx.connected_components(self.G))
        if len(connected_components) > 1:
            validation_results["warnings"].append(
                f"Network has {len(connected_components)} disconnected components"
            )
        
        validation_results["is_valid"] = not bool(validation_results["errors"])
        return validation_results
