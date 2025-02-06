import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
import logging

class NetworkExplorer:
    def __init__(self, G):
        """Initialize Network Explorer with graph"""
        self.G = G
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cache for expensive computations
        self._cluster_interaction_cache = None
        self._paper_distribution_cache = None

    def cluster_interaction_analysis(self):
        """
        Analyze interactions between clusters with interactive visualizations
        """
        if self._cluster_interaction_cache:
            return self._cluster_interaction_cache

        # Extract cluster information
        clusters = {}
        for node, data in self.G.nodes(data=True):
            cluster = data.get('cluster', 'Unknown')
            if cluster not in clusters:
                clusters[cluster] = {
                    'nodes': [],
                    'node_types': Counter(),
                    'connections': defaultdict(int),
                    'internal_edges': 0,
                    'external_edges': 0
                }
            clusters[cluster]['nodes'].append(node)
            clusters[cluster]['node_types'][data.get('type', 'Unknown')] += 1

        # Analyze edges
        for u, v, data in self.G.edges(data=True):
            cluster_u = self.G.nodes[u].get('cluster', 'Unknown')
            cluster_v = self.G.nodes[v].get('cluster', 'Unknown')
            
            if cluster_u == cluster_v:
                clusters[cluster_u]['internal_edges'] += 1
            else:
                clusters[cluster_u]['external_edges'] += 1
                clusters[cluster_v]['external_edges'] += 1
                clusters[cluster_u]['connections'][cluster_v] += 1
                clusters[cluster_v]['connections'][cluster_u] += 1

        # Prepare results
        results = {
            'cluster_details': {},
            'visualizations': {}
        }

        # Compile detailed metrics
        for cluster, info in clusters.items():
            results['cluster_details'][cluster] = {
                'total_nodes': len(info['nodes']),
                'node_types': dict(info['node_types']),
                'inter_cluster_connections': dict(info['connections']),
                'internal_edges': info['internal_edges'],
                'external_edges': info['external_edges'],
                'modularity': info['internal_edges'] / 
                            (info['internal_edges'] + info['external_edges'] + 1e-10)
            }

        # Create interactive dropdown visualization
        cluster_names = list(clusters.keys())
        fig_dropdown = go.Figure()

        for cluster in cluster_names:
            connections = clusters[cluster]['connections']
            if connections:  # Only add if there are connections
                fig_dropdown.add_trace(
                    go.Bar(
                        x=list(connections.keys()),
                        y=list(connections.values()),
                        name=cluster,
                        visible=(cluster == cluster_names[0])
                    )
                )

        # Create dropdown menu
        buttons = []
        for i, cluster in enumerate(cluster_names):
            visibility = [False] * len(cluster_names)
            visibility[i] = True
            buttons.append(dict(
                label=cluster,
                method='update',
                args=[
                    {'visible': visibility},
                    {'title': f'Connections from {cluster} to Other Clusters'}
                ]
            ))

        fig_dropdown.update_layout(
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            title=f'Connections from {cluster_names[0]} to Other Clusters',
            xaxis_title='Target Cluster',
            yaxis_title='Number of Connections',
            height=500,
            width=800
        )
        
        results['visualizations']['interactive_cluster_dropdown'] = fig_dropdown

        # Create heatmap
        interaction_matrix = np.zeros((len(cluster_names), len(cluster_names)))
        for i, cluster1 in enumerate(cluster_names):
            for j, cluster2 in enumerate(cluster_names):
                if cluster1 != cluster2:
                    interaction_matrix[i, j] = clusters[cluster1]['connections'].get(cluster2, 0)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=cluster_names,
            y=cluster_names,
            colorscale='Viridis',
            hoverongaps=False
        ))

        fig_heatmap.update_layout(
            title='Inter-Cluster Interaction Heatmap',
            xaxis_title='Target Cluster',
            yaxis_title='Source Cluster',
            height=600,
            width=800
        )
        
        results['visualizations']['interaction_heatmap'] = fig_heatmap

        # Node type composition by cluster
        df_composition = pd.DataFrame([
            {
                'Cluster': cluster,
                'NodeType': node_type,
                'Count': count
            }
            for cluster, info in clusters.items()
            for node_type, count in info['node_types'].items()
        ])

        fig_composition = px.bar(
            df_composition,
            x='Cluster',
            y='Count',
            color='NodeType',
            title='Node Type Composition by Cluster',
            barmode='group'
        )

        fig_composition.update_layout(
            height=500,
            width=800
        )
        
        results['visualizations']['cluster_composition'] = fig_composition

        # Cluster metrics visualization
        metrics_df = pd.DataFrame([
            {
                'Cluster': cluster,
                'Nodes': info['total_nodes'],
                'Internal Edges': info['internal_edges'],
                'External Edges': info['external_edges'],
                'Modularity': results['cluster_details'][cluster]['modularity']
            }
            for cluster, info in clusters.items()
        ])

        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Nodes per Cluster',
                'Internal vs External Edges',
                'Cluster Modularity',
                'Connection Distribution'
            )
        )

        # Nodes per cluster
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Cluster'], y=metrics_df['Nodes'], name='Nodes'),
            row=1, col=1
        )

        # Internal vs External edges
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Cluster'], y=metrics_df['Internal Edges'], 
                  name='Internal Edges'),
            row=1, col=2
        )
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Cluster'], y=metrics_df['External Edges'], 
                  name='External Edges'),
            row=1, col=2
        )

        # Modularity
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Cluster'], y=metrics_df['Modularity'], 
                  name='Modularity'),
            row=2, col=1
        )

        # Connection distribution
        connection_counts = [len(info['connections']) for info in clusters.values()]
        fig_metrics.add_trace(
            go.Histogram(x=connection_counts, name='Connections'),
            row=2, col=2
        )

        fig_metrics.update_layout(
            height=800,
            width=1000,
            showlegend=True,
            title_text='Cluster Metrics Overview'
        )
        
        results['visualizations']['cluster_metrics'] = fig_metrics

        self._cluster_interaction_cache = results
        return results

    def paper_distribution_analysis(self):
        """
        Analyze distribution of nodes and edges across papers
        """
        if self._paper_distribution_cache:
            return self._paper_distribution_cache

        # Group by PMID
        paper_stats = defaultdict(lambda: {
            'nodes': [], 
            'edges': [], 
            'node_types': Counter(),
            'clusters': Counter()
        })

        # Process nodes
        for node, data in self.G.nodes(data=True):
            pmid = data.get('PMID', 'Unknown')
            paper_stats[pmid]['nodes'].append(node)
            paper_stats[pmid]['node_types'][data.get('type', 'Unknown')] += 1
            paper_stats[pmid]['clusters'][data.get('cluster', 'Unknown')] += 1

        # Process edges
        for u, v, data in self.G.edges(data=True):
            pmid = self.G.nodes[u].get('PMID', 'Unknown')
            paper_stats[pmid]['edges'].append((u, v))

        # Prepare results
        paper_data = [
            {
                'PMID': pmid,
                'Nodes': len(stats['nodes']),
                'Edges': len(stats['edges']),
                'Node_Types': dict(stats['node_types']),
                'Clusters': dict(stats['clusters'])
            }
            for pmid, stats in paper_stats.items()
        ]

        df_papers = pd.DataFrame(paper_data)

        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Nodes per Paper',
                'Distribution of Edges per Paper',
                'Nodes vs Edges Scatter',
                'Node Types Across Papers'
            )
        )

        # Distribution of nodes per paper
        fig.add_trace(
            go.Histogram(x=df_papers['Nodes'], name='Nodes per Paper',
                        nbinsx=20),
            row=1, col=1
        )

        # Distribution of edges per paper
        fig.add_trace(
            go.Histogram(x=df_papers['Edges'], name='Edges per Paper',
                        nbinsx=20),
            row=1, col=2
        )

        # Nodes vs Edges scatter
        fig.add_trace(
            go.Scatter(
                x=df_papers['Nodes'],
                y=df_papers['Edges'],
                mode='markers',
                text=df_papers['PMID'],
                marker=dict(
                    size=10,
                    color=df_papers['Nodes'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Papers'
            ),
            row=2, col=1
        )

        # Node types across papers
        all_node_types = Counter()
        for types in df_papers['Node_Types']:
            all_node_types.update(types)

        fig.add_trace(
            go.Bar(
                x=list(all_node_types.keys()),
                y=list(all_node_types.values()),
                name='Node Types'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Paper Distribution Analysis'
        )

        results = {
            'summary': {
                'total_papers': len(paper_stats),
                'avg_nodes_per_paper': df_papers['Nodes'].mean(),
                'avg_edges_per_paper': df_papers['Edges'].mean(),
                'max_nodes_paper': df_papers['Nodes'].max(),
                'max_edges_paper': df_papers['Edges'].max()
            },
            'paper_details': paper_data,
            'visualizations': {
                'main_overview': fig
            }
        }

        self._paper_distribution_cache = results
        return results

    def get_filtered_view(self, node_types=None, min_degree=1, min_weight=0.0):
        """Get filtered view of the network based on criteria"""
        H = self.G.copy()
        
        # Apply filters
        if node_types:
            nodes_to_remove = [
                node for node, data in H.nodes(data=True)
                if data.get('type') not in node_types
            ]
            H.remove_nodes_from(nodes_to_remove)
        
        if min_degree > 1:
            nodes_to_remove = [
                node for node, degree in H.degree()
                if degree < min_degree
            ]
            H.remove_nodes_from(nodes_to_remove)
        
        if min_weight > 0:
            edges_to_remove = [
                (u, v) for u, v, d in H.edges(data=True)
                if d.get('weight', 0) < min_weight
            ]
            H.remove_edges_from(edges_to_remove)

        # Calculate metrics for filtered network
        type_dist = Counter(nx.get_node_attributes(H, 'type').values())
        cluster_dist = Counter(nx.get_node_attributes(H, 'cluster').values())
        degree_dist = [d for n, d in H.degree()]

        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Node Type Distribution',
                'Cluster Distribution',
                'Degree Distribution',
                'Node Metrics'
            )
        )

        # Node type distribution
        fig.add_trace(
            go.Bar(
                x=list(type_dist.keys()),
                y=list(type_dist.values()),
                name='Node Types'
            ),
            row=1, col=1
        )

        # Cluster distribution
        fig.add_trace(
            go.Bar(
                x=list(cluster_dist.keys()),
                y=list(cluster_dist.values()),
                name='Clusters'
            ),
            row=1, col=2
        )

        # Degree distribution
        fig.add_trace(
            go.Histogram(
                x=degree_dist,
                name='Degree Distribution',
                nbinsx=20
            ),
            row=2, col=1
        )

        # Node metrics
        betweenness = nx.betweenness_centrality(H)
        fig.add_trace(
            go.Box(
                y=list(betweenness.values()),
                name='Betweenness'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Filtered Network Analysis'
        )

        return {
            'graph': H,
            'metrics': {
                'nodes': H.number_of_nodes(),
                'edges': H.number_of_edges(),
                'density': nx.density(H),
                'avg_clustering': nx.average_clustering(H),
                'avg_degree': np.mean(degree_dist)
            },
            'distributions': {
                'node_types': dict(type_dist),
                'clusters': dict(cluster_dist),
                'degrees': degree_dist
            },
            'visualization': fig
        }
    


    def explore_node(self, node_id):
        """
        Detailed exploration of a specific node
        
        Args:
            node_id: ID of the node to explore
            
        Returns:
            dict: Comprehensive node analysis with visualizations
        """
        if node_id not in self.G:
            return None
            
        # Get node information
        node_data = self.G.nodes[node_id]
        neighbors = list(self.G.neighbors(node_id))
        
        # Calculate node metrics
        metrics = {
            'degree': self.G.degree(node_id),
            'degree_centrality': nx.degree_centrality(self.G)[node_id],
            'betweenness_centrality': nx.betweenness_centrality(self.G)[node_id],
            'closeness_centrality': nx.closeness_centrality(self.G)[node_id],
            'clustering_coefficient': nx.clustering(self.G, node_id)
        }
        
        # Analyze neighborhood
        neighbor_types = Counter(self.G.nodes[n]['type'] for n in neighbors)
        neighbor_clusters = Counter(self.G.nodes[n]['cluster'] for n in neighbors)
        
        # Get connection strengths (edge weights)
        edge_weights = [
            self.G[node_id][n].get('weight', 1.0) 
            for n in neighbors
        ]
        
        # Create subgraph visualization
        subgraph = self.G.subgraph(neighbors + [node_id])
        pos = nx.spring_layout(subgraph)
        
        # Create interactive network visualization
        edge_trace = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = subgraph[edge[0]][edge[1]].get('weight', 1.0)
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=weight, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == node_id:
                node_colors.append('red')
                node_sizes.append(20)
                node_text.append(f'Central Node: {node}<br>'
                               f'Type: {node_data["type"]}<br>'
                               f'Cluster: {node_data["cluster"]}')
            else:
                node_data = subgraph.nodes[node]
                node_colors.append('lightblue')
                node_sizes.append(15)
                node_text.append(f'Neighbor: {node}<br>'
                               f'Type: {node_data["type"]}<br>'
                               f'Cluster: {node_data["cluster"]}')
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2
            )
        )
        
        # Create network visualization
        fig_network = go.Figure(data=edge_trace + [node_trace])
        fig_network.update_layout(
            title=f'Neighborhood Network for Node {node_id}',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        # Create distribution visualizations
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Neighbor Type Distribution',
                'Neighbor Cluster Distribution',
                'Edge Weight Distribution',
                'Neighbor Degree Distribution'
            )
        )
        
        # Neighbor type distribution
        fig_dist.add_trace(
            go.Bar(
                x=list(neighbor_types.keys()),
                y=list(neighbor_types.values()),
                name='Types'
            ),
            row=1, col=1
        )
        
        # Neighbor cluster distribution
        fig_dist.add_trace(
            go.Bar(
                x=list(neighbor_clusters.keys()),
                y=list(neighbor_clusters.values()),
                name='Clusters'
            ),
            row=1, col=2
        )
        
        # Edge weight distribution
        fig_dist.add_trace(
            go.Histogram(
                x=edge_weights,
                name='Edge Weights',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Neighbor degree distribution
        neighbor_degrees = [self.G.degree(n) for n in neighbors]
        fig_dist.add_trace(
            go.Histogram(
                x=neighbor_degrees,
                name='Neighbor Degrees',
                nbinsx=20
            ),
            row=2, col=2
        )
        
        fig_dist.update_layout(
            height=800,
            showlegend=True,
            title_text=f'Neighborhood Analysis for Node {node_id}'
        )
        
        return {
            'node_data': node_data,
            'metrics': metrics,
            'neighborhood': {
                'total_neighbors': len(neighbors),
                'neighbor_types': dict(neighbor_types),
                'neighbor_clusters': dict(neighbor_clusters),
                'avg_edge_weight': np.mean(edge_weights),
                'max_edge_weight': max(edge_weights)
            },
            'visualizations': {
                'network': fig_network,
                'distributions': fig_dist
            }
        }
        
    def get_network_summary(self):
        """
        Get comprehensive network summary with visualizations
        
        Returns:
            dict: Network summary statistics and visualizations
        """
        # Calculate basic metrics
        metrics = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'avg_clustering': nx.average_clustering(self.G),
            'avg_degree': np.mean([d for n, d in self.G.degree()]),
            'is_connected': nx.is_connected(self.G),
            'number_components': nx.number_connected_components(self.G)
        }
        
        # Get largest component metrics
        if not metrics['is_connected']:
            largest_cc = max(nx.connected_components(self.G), key=len)
            largest_subgraph = self.G.subgraph(largest_cc)
            metrics.update({
                'largest_component_size': len(largest_cc),
                'largest_component_ratio': len(largest_cc) / metrics['total_nodes'],
                'largest_component_density': nx.density(largest_subgraph),
                'largest_component_diameter': nx.diameter(largest_subgraph)
            })
        
        # Get distributions
        distributions = {
            'node_types': Counter(nx.get_node_attributes(self.G, 'type').values()),
            'clusters': Counter(nx.get_node_attributes(self.G, 'cluster').values()),
            'degrees': Counter(dict(self.G.degree()).values()),
            'component_sizes': [len(c) for c in nx.connected_components(self.G)]
        }
        
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Node Type Distribution',
                'Cluster Distribution',
                'Degree Distribution',
                'Component Size Distribution'
            )
        )
        
        # Node type distribution
        fig.add_trace(
            go.Bar(
                x=list(distributions['node_types'].keys()),
                y=list(distributions['node_types'].values()),
                name='Node Types'
            ),
            row=1, col=1
        )
        
        # Cluster distribution
        fig.add_trace(
            go.Bar(
                x=list(distributions['clusters'].keys()),
                y=list(distributions['clusters'].values()),
                name='Clusters'
            ),
            row=1, col=2
        )
        
        # Degree distribution
        fig.add_trace(
            go.Histogram(
                x=list(distributions['degrees'].keys()),
                y=list(distributions['degrees'].values()),
                name='Degree Distribution',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Component size distribution
        if not metrics['is_connected']:
            fig.add_trace(
                go.Histogram(
                    x=distributions['component_sizes'],
                    name='Component Sizes',
                    nbinsx=20
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Network Summary Analysis'
        )
        
        # Create centrality comparison
        centrality_methods = {
            'Degree': nx.degree_centrality(self.G),
            'Betweenness': nx.betweenness_centrality(self.G),
            'Closeness': nx.closeness_centrality(self.G),
            'Eigenvector': nx.eigenvector_centrality(self.G, max_iter=1000)
        }
        
        # Prepare centrality data
        centrality_data = pd.DataFrame(centrality_methods)
        
        # Create centrality correlation heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=centrality_data.corr(),
            x=centrality_data.columns,
            y=centrality_data.columns,
            colorscale='RdBu'
        ))
        
        fig_corr.update_layout(
            title='Centrality Measure Correlations',
            height=500
        )
        
        return {
            'metrics': metrics,
            'distributions': distributions,
            'centrality_correlations': centrality_data.corr().to_dict(),
            'visualizations': {
                'main_overview': fig,
                'centrality_correlation': fig_corr
            }
        }
