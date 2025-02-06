import streamlit as st
import json
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Network_stats import NetworkAnalyzer

# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Biomedical Knowledge Graph")

# Color scheme with visually appealing colors
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

def initialize_session_state():
    """Initialize session state variables."""
    if 'selected_nodes' not in st.session_state:
        st.session_state.selected_nodes = set()
    if 'last_cluster' not in st.session_state:
        st.session_state.last_cluster = "All"

def load_data(file_path="data_unique.json"):
    """Load and validate network data."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data["nodes"], data["edges"], data["clusters"]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def create_network(nodes_data, edges_data, selected_cluster=None, selected_nodes=None):
    """Create an interactive network visualization."""
    try:
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
        net.force_atlas_2based()

        # Add nodes with enhanced styling
        for node in nodes_data:
            color = color_scheme.get(node["type"], color_scheme["type"])
            base_size = node["size"] * 10 if "size" in node else 10
            
            # Handle node highlighting
            if selected_cluster or selected_nodes:
                is_highlighted = (not selected_cluster or node["cluster"] == selected_cluster) and \
                               (not selected_nodes or node["id"] in selected_nodes)
                if not is_highlighted:
                    rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    color = f"rgba{rgb + (0.15,)}"
                    base_size *= 0.75

            # Enhanced tooltip with more information
            tooltip = (f"Node: {node['id']}<br>"
                      f"Type: {node['type']}<br>"
                      f"Cluster: {node['cluster']}<br>"
                      f"Size: {node['size']:.2f}")
            if 'PMID' in node:
                tooltip += f"<br>PMID: {node['PMID']}"

            net.add_node(
                node["id"],
                label=node["id"],
                color=color,
                title=tooltip,
                size=base_size
            )

        # Add edges with interactive features
        for edge in edges_data:
            # Enhanced edge styling based on score
            edge_width = edge["score"] * 2
            edge_color = f"rgba(102, 102, 102, {min(edge['score'], 1.0)})"
            
            net.add_edge(
                edge["source"],
                edge["target"],
                title=f"Relation: {edge['relation']}<br>Score: {edge['score']:.2f}",
                width=edge_width,
                color=edge_color
            )

        # Configure network physics for better visualization
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
                "multiselect": true,
                "navigationButtons": true
            }
        }
        """)

        return net
    except Exception as e:
        st.error(f"Error creating network: {str(e)}")
        return None

def handle_node_selection():
    """Handle interactive node selection and analysis."""
    selected_nodes = st.session_state.get('selected_nodes', set())
    
    if selected_nodes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Selected Nodes Analysis")
        
        # Clear selection option
        if st.sidebar.button("Clear Selection"):
            st.session_state.selected_nodes = set()
            st.rerun()
        
        # Display selected nodes
        st.sidebar.markdown(f"**Selected Nodes:** {', '.join(selected_nodes)}")
        
        # Analyze selected nodes
        return True, selected_nodes
    return False, None

def display_network_metrics(analyzer, full_graph=True):
    """Display key network metrics with visualizations."""
    if full_graph:
        stats = analyzer.basic_analyzer.get_basic_stats()
    else:
        stats = analyzer.basic_analyzer.get_cluster_stats(st.session_state.last_cluster)

    # Display key metrics in an organized layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", stats["Total Nodes"])
    with col2:
        st.metric("Edges", stats["Total Edges"])
    with col3:
        st.metric("Density", f"{stats['Network Density']:.3f}")
    with col4:
        st.metric("Clustering Coefficient", f"{stats['Average Clustering Coefficient']:.3f}")

    return stats

def create_network_overview(analyzer):
    """Create comprehensive network overview visualization."""
    # Get basic network statistics
    basic_stats = analyzer.basic_analyzer.get_basic_stats()
    
    # Create overview visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Node Type Distribution', 'Edge Weight Distribution',
                       'Cluster Size Distribution', 'Degree Distribution')
    )

    # Node type distribution
    node_types = Counter(node['type'] for node in analyzer.nodes_data)
    fig.add_trace(
        go.Pie(labels=list(node_types.keys()),
               values=list(node_types.values())),
        row=1, col=1
    )

    # Edge weight distribution
    edge_weights = [edge['score'] for edge in analyzer.edges_data]
    fig.add_trace(
        go.Histogram(x=edge_weights, nbinsx=30),
        row=1, col=2
    )

    # Cluster size distribution
    cluster_sizes = Counter(node['cluster'] for node in analyzer.nodes_data)
    fig.add_trace(
        go.Bar(x=list(cluster_sizes.keys()),
               y=list(cluster_sizes.values())),
        row=2, col=1
    )

    # Degree distribution
    degrees = [d for n, d in analyzer.G.degree()]
    fig.add_trace(
        go.Histogram(x=degrees, nbinsx=30),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False)
    return fig

def main():
    """Main application function."""
    try:
        # Initialize session state and load data
        initialize_session_state()
        nodes_data, edges_data, clusters_data = load_data()
        
        if not all([nodes_data, edges_data, clusters_data]):
            st.error("Failed to load network data")
            return

        # Initialize network analyzer
        analyzer = NetworkAnalyzer(nodes_data, edges_data)

        # Sidebar controls
        st.sidebar.title("Network Controls")
        
        # Cluster selection
        selected_cluster = st.sidebar.selectbox(
            "Select Cluster",
            ["All"] + list(clusters_data.keys()),
            key="cluster_selector"
        )

        # Update session state
        st.session_state.last_cluster = selected_cluster

        # Main content area
        st.title("Biomedical Knowledge Graph Analysis")
        
        # Create tabs for different views
        main_tab, analysis_tab = st.tabs(["Network Visualization", "Detailed Analysis"])

        with main_tab:
            # Display network metrics
            st.header("Network Overview")
            display_network_metrics(analyzer, selected_cluster == "All")

            # Create and display interactive network
            with st.spinner("Loading network visualization..."):
                has_selection, selected_nodes = handle_node_selection()
                net = create_network(
                    nodes_data,
                    edges_data,
                    None if selected_cluster == "All" else selected_cluster,
                    selected_nodes if has_selection else None
                )
                
                if net:
                    net.save_graph("network.html")
                    with open("network.html", "r", encoding="utf-8") as f:
                        components.html(f.read(), height=800)

            # Display color legend
            st.write("### Node Type Legend")
            legend_cols = st.columns(4)
            for i, (node_type, color) in enumerate(color_scheme.items()):
                with legend_cols[i % 4]:
                    st.markdown(
                        f'<div style="display: flex; align-items: center;">'
                        f'<div style="width: 20px; height: 20px; background-color: {color}; '
                        f'margin-right: 10px; border-radius: 50%;"></div>'
                        f'<span>{node_type.capitalize()}</span></div>',
                        unsafe_allow_html=True
                    )

        with analysis_tab:
            # Create analysis subtabs
            basic_tab, advanced_tab, explorer_tab = st.tabs([
                "Basic Analysis",
                "Advanced Analysis",
                "Network Explorer"
            ])

            with basic_tab:
                # Display basic network analysis
                st.plotly_chart(create_network_overview(analyzer))
                
                if selected_cluster != "All":
                    cluster_stats = analyzer.basic_analyzer.get_cluster_stats(selected_cluster)
                    if cluster_stats:
                        st.subheader(f"Cluster Analysis: {selected_cluster}")
                        st.write(cluster_stats)

                # Display pathway analysis
                pathway_stats = analyzer.basic_analyzer.get_pathway_analysis()
                if pathway_stats:
                    st.subheader("Pathway Analysis")
                    st.write(pathway_stats)

            with advanced_tab:
                # Advanced analysis options
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Community Structure", "Network Resilience", "Temporal Evolution"]
                )

                if analysis_type == "Community Structure":
                    results = analyzer.advanced_analyzer.network_embedding()
                    st.plotly_chart(results['visualization'])
                    
                elif analysis_type == "Network Resilience":
                    results = analyzer.advanced_analyzer.network_resilience_analysis()
                    st.plotly_chart(results['visualization'])
                    
                elif analysis_type == "Temporal Evolution":
                    results = analyzer.advanced_analyzer.network_entropy_analysis()
                    for viz_name, viz in results['visualizations'].items():
                        st.plotly_chart(viz)

            with explorer_tab:
                # Node exploration interface
                selected_node = st.selectbox(
                    "Select Node to Explore",
                    options=[node["id"] for node in nodes_data]
                )

                if selected_node:
                    node_details = analyzer.network_explorer.advanced_node_exploration(selected_node)
                    
                    # Display node details
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(node_details['visualizations']['neighborhood_graph'])
                    with col2:
                        st.write("### Node Properties")
                        st.write(node_details['basic_info'])
                        st.write("### Centrality Metrics")
                        st.write(node_details['centrality_metrics'])

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
