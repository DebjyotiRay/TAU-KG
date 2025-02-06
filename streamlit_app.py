import streamlit as st
import json
import math
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import streamlit as st
from itertools import combinations

from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Network_stats import *
# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Biomedical Knowledge Graph")

# Load data from JSON file
def load_data(file_path="data_unique.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data["nodes"], data["edges"], data["clusters"]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Load the data
nodes_data, edges_data, clusters_data = load_data()

# Color scheme with additional types from your data
color_scheme = {
    "gene": "#1f77b4",      # Blue
    "gene group": "#aec7e8", # Light Blue
    "protein": "#2ca02c",   # Green
    "protein complex": "#98df8a",  # Light Green
    "protein fragment": "#ff9896", # Light Red
    "disease": "#d62728",   # Red
    "pathway": "#9467bd",   # Purple
    "process": "#c5b0d5",   # Light Purple
    "cell line": "#ff7f0e", # Orange
    "cell line clone": "#e377c2",  # Pink
    "cell population": "#8c564b",  # Brown
    "reagent": "#17becf",   # Cyan
    "chemical": "#bcbd22",  # Yellow-green
    "method": "#7f7f7f",    # Gray
    "organelle": "#9edae5", # Light blue
    "biological process": "#c49c94",  # Brown-red
    "phenotype": "#dbdb8d",  # Light yellow
    "type": "#c7c7c7"      # Default gray
}

# Utility functions
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_rgba(rgb, alpha):
    """Convert RGB tuple to RGBA string."""
    return f"rgba{rgb + (alpha,)}"

def safe_read_file(filename):
    """Safely read file with error handling."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def validate_data():
    """Validate data structure and content."""
    try:
        if not nodes_data or not edges_data:
            st.error("Network data is missing or empty")
            return False
        
        # Validate node references in edges
        node_ids = {node["id"] for node in nodes_data}
        invalid_edges = []
        for edge in edges_data:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                invalid_edges.append(edge)
        
        if invalid_edges:
            st.warning(f"Found {len(invalid_edges)} edges with invalid node references")
        
        # Check for required node attributes
        required_attrs = ["id", "type", "cluster", "size"]
        invalid_nodes = []
        for node in nodes_data:
            if not all(attr in node for attr in required_attrs):
                invalid_nodes.append(node["id"])
        
        if invalid_nodes:
            st.warning(f"Found nodes missing required attributes: {', '.join(invalid_nodes)}")
        
        # Validate color scheme
        node_types = {node["type"] for node in nodes_data}
        missing_colors = node_types - set(color_scheme.keys())
        if missing_colors:
            st.error(f"Missing colors for node types: {missing_colors}")
            return False
        
        return True
    except Exception as e:
        st.error(f"Error validating data: {str(e)}")
        return False

def handle_large_network():
    """Handle performance issues with large networks."""
    if len(nodes_data) > 1000 or len(edges_data) > 5000:
        st.warning("Large network detected. This may affect performance.")
        with st.expander("Performance Tips"):
            st.markdown("""
            - Consider filtering nodes by cluster
            - Reduce number of displayed edges
            - Use search instead of visual navigation
            """)

def get_node_relationships(node_ids):
    """Get all relationships for specified nodes."""
    relationships = []
    for edge in edges_data:
        if edge["source"] in node_ids or edge["target"] in node_ids:
            relationships.append({
                "source": edge["source"],
                "target": edge["target"],
                "relation": edge["relation"],
                "score": edge["score"]
            })
    return relationships

def create_network(selected_cluster=None):
    """Create and configure the network visualization."""
    try:
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
        net.force_atlas_2based()

        # Add nodes
        for node in nodes_data:
            color = color_scheme.get(node["type"], color_scheme["type"])
            base_size = node["size"] * 10
            
            if selected_cluster:
                is_in_cluster = node["cluster"] == selected_cluster
                if is_in_cluster:
                    size = base_size * 1.5
                else:
                    size = base_size
                    rgb = hex_to_rgb(color)
                    color = f"rgba{rgb + (0.15,)}"
            else:
                size = base_size

            # Enhanced tooltip
            mentions = round(math.exp(node['size']))
            title = (f"Node: {node['id']}<br>"
                    f"Cluster: {node['cluster']}<br>"
                    f"Mentions: {mentions}<br>"
                    f"PMID: {node.get('PMID', 'N/A')}")

            net.add_node(
                node["id"],
                label=node["id"],
                color=color,
                title=title,
                size=size
            )

        # Add edges
        for edge in edges_data:
            edge_color = "#666666"
            edge_width = edge["score"] * 3

            if selected_cluster:
                source_node = next((n for n in nodes_data if n["id"] == edge["source"]), None)
                target_node = next((n for n in nodes_data if n["id"] == edge["target"]), None)
                
                if source_node and target_node:
                    source_in_cluster = source_node["cluster"] == selected_cluster
                    target_in_cluster = target_node["cluster"] == selected_cluster

                    if source_in_cluster or target_in_cluster:
                        edge_color = "#000000"
                        edge_width = edge["score"] * 4
                    else:
                        edge_color = "rgba(102, 102, 102, 0.15)"
                        edge_width = edge["score"] * 2

            net.add_edge(
                edge["source"],
                edge["target"],
                title=f"Relation: {edge['relation']}<br>Score: {edge['score']:.2f}",
                width=edge_width,
                color=edge_color
            )

        # Physics options
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.015,
                    "springLength": 150,
                    "springConstant": 0.1,
                    "damping": 0.95
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "multiselect": true
            }
        }
        """)
        
        return net
    except Exception as e:
        st.error(f"Error creating network: {str(e)}")
        return None


def display_network_stats(nodes_data, edges_data, selected_cluster):
    """Display network statistics in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.title("Network Statistics")

    total_nodes = len(nodes_data)
    total_edges = len(edges_data)
    st.sidebar.write(f"Total Nodes: {total_nodes}")
    st.sidebar.write(f"Total Edges: {total_edges}")

    # Node type counting
    node_types_count = {}
    for node in nodes_data:
        if node["type"] in node_types_count:
            node_types_count[node["type"]] += 1
        else:
            node_types_count[node["type"]] = 1

    st.sidebar.write("\nNode Types:")
    for node_type, count in node_types_count.items():
        percentage = (count / total_nodes) * 100
        st.sidebar.write(f"{node_type.capitalize()}: {count} ({percentage:.1f}%)")

def add_search_functionality():
    """Add search box for finding specific nodes."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search")
    search_term = st.sidebar.text_input("Search Nodes", "").strip()
    
    if search_term:
        matching_nodes = [node["id"] for node in nodes_data 
                         if search_term.lower() in node["id"].lower()]
        if matching_nodes:
            node_to_highlight = st.sidebar.selectbox(
                "Matching nodes:", matching_nodes
            )
            if node_to_highlight:
                node_data = next((n for n in nodes_data if n["id"]== node_to_highlight), None)
                if node_data:
                    st.sidebar.markdown(f"**Type:** {node_data['type']}")
                    st.sidebar.markdown(f"**Cluster:** {node_data['cluster']}")
                    
                    # Option to add to selection
                    if st.sidebar.button("Add to Selection"):
                        st.session_state.selected_nodes.add(node_to_highlight)
                        st.rerun()

def handle_selected_nodes():
    """Handle data download for multiple selected nodes."""
    selected_nodes = st.session_state.get('selected_nodes', set())
    
    if selected_nodes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Selected Nodes")
        
        # Clear selection button
        if st.sidebar.button("Clear Selection"):
            st.session_state.selected_nodes = set()
            st.rerun()
        
        st.sidebar.markdown(f"**Selected ({len(selected_nodes)}):** {', '.join(selected_nodes)}")
        
        # Collect node and relationship data
        selected_node_data = []
        selected_relationships = []
        
        # Get node data and relationships
        for node_id in selected_nodes:
            # Get node data
            node_data = next((n for n in nodes_data if n["id"] == node_id), None)
            if node_data:
                selected_node_data.append(node_data)
            
            # Get relationships where either source or target is in selected nodes
            for edge in edges_data:
                if edge["source"] == node_id or edge["target"] == node_id:
                    if edge["source"] in selected_nodes and edge["target"] in selected_nodes:
                        selected_relationships.append(edge)
        
        # Create subgraph visualization for selected nodes
        if st.sidebar.button("Create Selected Nodes Network"):
            with st.spinner("Creating network of selected nodes..."):
                subnet = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
                subnet.force_atlas_2based()
                
                # Add selected nodes
                for node in selected_node_data:
                    color = color_scheme[node["type"]]
                    subnet.add_node(
                        node["id"],
                        label=node["id"],
                        color=color,
                        title=f"Type: {node['type']}<br>Cluster: {node['cluster']}<br>Size: {node['size']:.2f}",
                        size=node["size"] * 10
                    )
                
                # Add edges between selected nodes
                for edge in selected_relationships:
                    subnet.add_edge(
                        edge["source"],
                        edge["target"],
                        title=f"Relation: {edge['relation']}<br>Score: {edge['score']:.2f}",
                        width=edge["score"] * 3,
                        color="#666666"
                    )
                
                subnet.save_graph("selected_network.html")
                with open("selected_network.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                    
                st.sidebar.download_button(
                    label="Download Selected Network HTML",
                    data=html_content,
                    file_name="selected_nodes_network.html",
                    mime="text/html"
                )
        
        # Prepare textual data for download
        text_data = {
            "nodes": selected_node_data,
            "relationships": selected_relationships,
            "summary": {
                "total_selected_nodes": len(selected_nodes),
                "node_types": {},
                "total_relationships": len(selected_relationships)
            }
        }
        
        # Count node types
        for node in selected_node_data:
            node_type = node["type"]
            if node_type in text_data["summary"]["node_types"]:
                text_data["summary"]["node_types"][node_type] += 1
            else:
                text_data["summary"]["node_types"][node_type] = 1
        
        # Create JSON download button
        json_data = json.dumps(text_data, indent=2)
        st.sidebar.download_button(
            label="Download Selected Nodes Data (JSON)",
            data=json_data,
            file_name="selected_nodes_data.json",
            mime="application/json"
        )
        
        # Create readable text format
        text_content = f"""Selected Nodes Analysis
----------------------------
Total Nodes: {len(selected_nodes)}
Total Relationships: {len(selected_relationships)}

Node Type Distribution:
{chr(10).join(f'- {type_}: {count}' for type_, count in text_data['summary']['node_types'].items())}

Detailed Node Information:
{chr(10).join(f'- {node["id"]} ({node["type"]}): Cluster={node["cluster"]}, Size={node["size"]:.2f}' for node in selected_node_data)}

Relationships:
{chr(10).join(f'- {rel["source"]} -> {rel["target"]}: {rel["relation"]} (Score: {rel["score"]:.2f})' for rel in selected_relationships)}
"""
        
        # Create text download button
        st.sidebar.download_button(
            label="Download Selected Nodes Report (TXT)",
            data=text_content,
            file_name="selected_nodes_report.txt",
            mime="text/plain"
        )

def add_help_section():
    """Add help and instructions section."""
    with st.sidebar.expander("Help & Instructions"):
        st.markdown("""
        **Navigation Tips:**
        - Use Ctrl+Click to select multiple nodes
        - Use the search box to find specific nodes
        - Download selected node data from the sidebar
        - Clear selection to start over
                        
        **Features:**
        - Export full network or specific clusters
        - Download detailed statistics
        - Search and highlight specific nodes
        """)

def initialize_session_state():
    """Initialize session state variables."""
    if 'selected_nodes' not in st.session_state:
        st.session_state.selected_nodes = set()
    if 'last_cluster' not in st.session_state:
        st.session_state.last_cluster = "All"
def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Load and clean data
        global nodes_data, edges_data, clusters_data
        nodes_data, edges_data, clusters_data = load_data()
        
        if not nodes_data or not edges_data or not clusters_data:
            st.error("Failed to load data")
            return
            
        # Data validation reporting
        valid_node_ids = {node["id"] for node in nodes_data}
        invalid_edges = [
            edge for edge in edges_data 
            if edge["source"] not in valid_node_ids or edge["target"] not in valid_node_ids
        ]
        if invalid_edges:
            st.warning(f"Found {len(invalid_edges)} edges with invalid node references")
            with st.expander("Show invalid edges"):
                for edge in invalid_edges:
                    st.write(f"- {edge['source']} -> {edge['target']}: {edge['relation']}")
        
        # Remove invalid edges
        edges_data = [
            edge for edge in edges_data 
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]
        
        # Check network size and show performance tips if needed
        handle_large_network()
        
        # Initialize network analyzer
        analyzer = NetworkAnalyzer(nodes_data, edges_data)

        # Sidebar organization
        with st.sidebar:
            st.title("Network Navigation")
            
            # Cluster selection
            selected_cluster = st.selectbox(
                "Select Cluster to Highlight",
                ["All"] + list(clusters_data.keys())
            )

            # Help section
            add_help_section()

            # Search functionality
            add_search_functionality()

            # Handle selected nodes
            handle_selected_nodes()

            # Display network statistics
            display_network_stats(nodes_data, edges_data, selected_cluster)

            # Export options
            st.markdown("---")
            st.subheader("Export Options")

            # Export full network
            if st.button("Export Full Network"):
                with st.spinner("Preparing full network..."):
                    full_net = create_network()
                    if full_net:
                        full_net.save_graph("network_export_full.html")
                        content = safe_read_file("network_export_full.html")
                        if content:
                            st.download_button(
                                label="Download Full Network",
                                data=content,
                                file_name="full_network.html",
                                mime="text/html"
                            )

            # Export cluster
            if selected_cluster != "All":
                if st.button("Export Selected Cluster"):
                    with st.spinner(f"Preparing {selected_cluster} cluster..."):
                        cluster_net = create_network(selected_cluster)
                        if cluster_net:
                            cluster_net.save_graph("network_export_cluster.html")
                            content = safe_read_file("network_export_cluster.html")
                            if content:
                                st.download_button(
                                    label="Download Cluster Network",
                                    data=content,
                                    file_name=f"{selected_cluster.lower()}_network.html",
                                    mime="text/html"
                                )

            # Export statistics
            if st.button("Export Statistics as JSON"):
                with st.spinner("Preparing statistics..."):
                    export_stats = (
                        analyzer.get_cluster_stats(selected_cluster)
                        if selected_cluster != "All"
                        else analyzer.get_basic_stats()
                    )
                    json_stats = json.dumps(export_stats, indent=2)
                    st.download_button(
                        label="Download Statistics",
                        data=json_stats,
                        file_name="network_statistics.json",
                        mime="application/json"
                    )

        # Main content area
        main_tab, advanced_analysis_tab = st.tabs(["Network Visualization", "Detailed Analysis"])

        with main_tab:
            st.title("Biomedical Knowledge Graph Visualization")
            
            # Create and display network
            # with st.spinner("Loading network visualization..."):
            net = create_network(None if selected_cluster == "All" else selected_cluster)
            if net:
                # Add JavaScript for multi-node selection
                net.html = net.html.replace('</head>', '''
                    <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        network.on("click", function(params) {
                            if (params.nodes.length > 0 && params.event.srcEvent.ctrlKey) {
                                window.parent.postMessage({
                                    type: "nodes_selected",
                                    nodes: params.nodes
                                }, "*");
                            }
                        });
                    });
                    </script>
                    </head>
                ''')
                
                # Save and display network
                net.save_graph("network.html")
                content = safe_read_file("network.html")
                if content:
                    components.html(content, height=800)
                    st.success("Network visualization loaded successfully")

                # Display legend
                st.write("\n### Node Type Legend")
                cols = st.columns(4)
                for i, (node_type, color) in enumerate(color_scheme.items()):
                    with cols[i % 4]:
                        st.markdown(
                            f'<div style="display: flex; align-items: center;">'
                            f'<div style="width: 20px; height: 20px; background-color: {color}; '
                            f'margin-right: 10px; border-radius: 50%;"></div>'
                            f'<span style="font-weight: 500;">{node_type.capitalize()}</span></div>',
                            unsafe_allow_html=True
                        )

        # with stats_tab:
        #     # Display detailed network statistics and analysis
        #     analyzer.display_stats_streamlit(selected_cluster)
        with advanced_analysis_tab:
            st.header("Comprehensive Network Analysis")
            
            # Create multiple columns for quick overview
            overview_cols = st.columns(4)
            with overview_cols[0]:
                st.metric("Total Nodes", len(nodes_data))
            with overview_cols[1]:
                st.metric("Total Edges", len(edges_data))
            with overview_cols[2]:
                st.metric("Node Types", len(set(node['type'] for node in nodes_data)))
            with overview_cols[3]:
                st.metric("Clusters", len(clusters_data))
        
            # Tabbed interface for different analysis perspectives
            analysis_tabs = st.tabs([
                "Network Topology", 
                "Community Structure", 
                "Node Characteristics", 
                "Edge Analysis", 
                "Advanced Metrics", 
                "Temporal Dynamics", 
                "Predictive Insights"
            ])
        
            with analysis_tabs[0]:  # Network Topology
                st.subheader("Network Topology Exploration")
                
                # Topology visualization options
                topology_cols = st.columns(2)
                with topology_cols[0]:
                    # Degree distribution visualization
                    st.write("### Degree Distribution")
                    degrees = [dict(G.degree())[node] for node in G.nodes()]
                    fig_degree = go.Figure(data=[go.Histogram(x=degrees, nbinsx=30)])
                    fig_degree.update_layout(title="Node Degree Distribution")
                    st.plotly_chart(fig_degree)
                
                with topology_cols[1]:
                    # Centrality measures comparison
                    st.write("### Centrality Comparison")
                    centrality_methods = {
                        'Degree Centrality': nx.degree_centrality(G),
                        'Betweenness Centrality': nx.betweenness_centrality(G),
                        'Closeness Centrality': nx.closeness_centrality(G),
                        'Eigenvector Centrality': nx.eigenvector_centrality(G)
                    }
                    
                    # Boxplot of centrality measures
                    centrality_df = pd.DataFrame(centrality_methods)
                    fig_centrality = go.Figure()
                    for column in centrality_df.columns:
                        fig_centrality.add_trace(go.Box(y=centrality_df[column], name=column))
                    fig_centrality.update_layout(title="Centrality Measures Comparison")
                    st.plotly_chart(fig_centrality)
        
            with analysis_tabs[1]:  # Community Structure
                st.subheader("Community Detection and Analysis")
                
                # Community detection method selection
                community_method = st.selectbox(
                    "Select Community Detection Algorithm", 
                    ["Louvain", "Greedy Modularity", "Label Propagation"]
                )
                
                # Perform community detection
                if community_method == "Louvain":
                    communities = nx.community.louvain_communities(G)
                elif community_method == "Greedy Modularity":
                    communities = nx.community.greedy_modularity_communities(G)
                else:
                    communities = nx.community.label_propagation_communities(G)
                
                # Community analysis
                community_cols = st.columns(2)
                with community_cols[0]:
                    st.write("### Community Size Distribution")
                    community_sizes = [len(community) for community in communities]
                    fig_community_sizes = go.Figure(data=[go.Histogram(x=community_sizes, nbinsx=20)])
                    fig_community_sizes.update_layout(title="Community Size Distribution")
                    st.plotly_chart(fig_community_sizes)
                
                with community_cols[1]:
                    st.write("### Community Composition")
                    # Analyze node types in largest community
                    largest_community = max(communities, key=len)
                    community_node_types = Counter(
                        G.nodes[node].get('type', 'Unknown') for node in largest_community
                    )
                    
                    fig_community_types = go.Figure(data=[
                        go.Pie(
                            labels=list(community_node_types.keys()),
                            values=list(community_node_types.values())
                        )
                    ])
                    fig_community_types.update_layout(title="Node Types in Largest Community")
                    st.plotly_chart(fig_community_types)
        
            with analysis_tabs[2]:  # Node Characteristics
                st.subheader("Node Type and Cluster Analysis")
                
                # Node type distribution
                node_type_dist = Counter(node['type'] for node in nodes_data)
                fig_node_types = go.Figure(data=[
                    go.Pie(
                        labels=list(node_type_dist.keys()),
                        values=list(node_type_dist.values())
                    )
                ])
                fig_node_types.update_layout(title="Node Type Distribution")
                st.plotly_chart(fig_node_types)
                
                # Cluster analysis
                st.write("### Cluster Composition")
                cluster_type_dist = defaultdict(lambda: defaultdict(int))
                for node in nodes_data:
                    cluster_type_dist[node['cluster']][node['type']] += 1
                
                # Interactive cluster selection
                selected_cluster = st.selectbox(
                    "Select Cluster for Detailed Analysis", 
                    list(cluster_type_dist.keys())
                )
                
                if selected_cluster:
                    cluster_types = cluster_type_dist[selected_cluster]
                    fig_cluster_types = go.Figure(data=[
                        go.Bar(
                            x=list(cluster_types.keys()),
                            y=list(cluster_types.values())
                        )
                    ])
                    fig_cluster_types.update_layout(
                        title=f"Node Types in {selected_cluster} Cluster",
                        xaxis_title="Node Type",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_cluster_types)
        
            with analysis_tabs[3]:  # Edge Analysis
                st.subheader("Edge Relationship and Weight Analysis")
                
                # Edge weight distribution
                edge_weights = [edge['score'] for edge in edges_data]
                fig_edge_weights = go.Figure(data=[go.Histogram(x=edge_weights, nbinsx=30)])
                fig_edge_weights.update_layout(title="Edge Weight Distribution")
                st.plotly_chart(fig_edge_weights)
                
                # Relation type analysis
                relation_types = Counter(edge['relation'] for edge in edges_data)
                fig_relation_types = go.Figure(data=[
                    go.Pie(
                        labels=list(relation_types.keys()),
                        values=list(relation_types.values())
                    )
                ])
                fig_relation_types.update_layout(title="Edge Relation Types")
                st.plotly_chart(fig_relation_types)
        
            with analysis_tabs[4]:  # Advanced Metrics
                st.subheader("Advanced Network Metrics")
                
                # Network complexity metrics
                st.write("### Network Complexity")
                complexity_metrics = {
                    "Clustering Coefficient": nx.average_clustering(G),
                    "Network Density": nx.density(G),
                    "Average Path Length": nx.average_shortest_path_length(G),
                    "Network Diameter": nx.diameter(G)
                }
                
                for metric, value in complexity_metrics.items():
                    st.metric(metric, f"{value:.4f}")
        
            with analysis_tabs[5]:  # Temporal Dynamics
                st.subheader("Temporal Network Evolution")
                
                # Check for temporal attributes (PMID)
                temporal_nodes = [node for node in nodes_data if 'PMID' in node]
                if temporal_nodes:
                    # Group by PMID
                    pmid_groups = defaultdict(list)
                    for node in temporal_nodes:
                        pmid_groups[node.get('PMID', 'Unknown')].append(node)
                    
                    # Analyze network metrics over time
                    temporal_metrics = []
                    for pmid, nodes in pmid_groups.items():
                        subgraph = G.subgraph(nodes)
                        temporal_metrics.append({
                            'PMID': pmid,
                            'Nodes': len(subgraph.nodes()),
                            'Edges': len(subgraph.edges()),
                            'Density': nx.density(subgraph)
                        })
                    
                    # Temporal evolution visualization
                    df_temporal = pd.DataFrame(temporal_metrics)
                    
                    # Line plots for temporal metrics
                    fig_temporal = make_subplots(rows=1, cols=3, 
                                                 subplot_titles=('Nodes', 'Edges', 'Network Density'))
                    
                    fig_temporal.add_trace(
                        go.Scatter(x=df_temporal['PMID'], y=df_temporal['Nodes'], mode='lines+markers'),
                        row=1, col=1
                    )
                    fig_temporal.add_trace(
                        go.Scatter(x=df_temporal['PMID'], y=df_temporal['Edges'], mode='lines+markers'),
                        row=1, col=2
                    )
                    fig_temporal.add_trace(
                        go.Scatter(x=df_temporal['PMID'], y=df_temporal['Density'], mode='lines+markers'),
                        row=1, col=3
                    )
                    
                    fig_temporal.update_layout(height=500, title_text="Network Metrics Across Publications")
                    st.plotly_chart(fig_temporal)
                else:
                    st.warning("No temporal data available for analysis")
        
            with analysis_tabs[6]:  # Predictive Insights
                st.subheader("Network Predictive Analysis")
                
                # Link prediction methods
                st.write("### Link Prediction")
                prediction_methods = {
                    'Common Neighbors': nx.resource_allocation_index,
                    'Jaccard Coefficient': nx.jaccard_coefficient,
                    'Adamic-Adar': nx.adamic_adar_index
                }
                
                # Select prediction method
                selected_method = st.selectbox(
                    "Select Link Prediction Method", 
                    list(prediction_methods.keys())
                )
                
                # Get non-existing edges
                non_edges = list(nx.non_edges(G))
                
                # Compute predictions
                predictions = list(prediction_methods[selected_method](G, non_edges))
                
                # Sort and display top potential links
                predictions.sort(key=lambda x: x[2], reverse=True)
                top_predictions = predictions[:10]
                
                st.write("### Top 10 Potential Links")
                prediction_df = pd.DataFrame(
                    top_predictions, 
                    columns=['Source', 'Target', 'Prediction Score']
                )
                st.dataframe(prediction_df)
                
                # Visualization of prediction scores
                fig_predictions = go.Figure(data=[go.Histogram(x=[p[2] for p in top_predictions])])
                fig_predictions.update_layout(
                    title=f"Distribution of {selected_method} Prediction Scores",
                    xaxis_title="Prediction Score",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_predictions)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
