import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from scipy import stats 
class NetworkExplorer:
    def __init__(self, G: nx.Graph):
        """
        Initialize Network Explorer with enhanced validation and caching
        
        Args:
            G (networkx.Graph): Input network graph
            
        Raises:
            ValueError: If graph is empty or invalid
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph object")
            
        if G.number_of_nodes() == 0:
            raise ValueError("Graph contains no nodes")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate required node attributes
        required_attrs = ['type', 'cluster']
        missing_attrs = defaultdict(list)
        
        for node, data in G.nodes(data=True):
            for attr in required_attrs:
                if attr not in data:
                    missing_attrs[attr].append(node)
                    
        if missing_attrs:
            self.logger.warning(
                "Nodes missing required attributes: %s",
                dict(missing_attrs)
            )
        
        self.G = G
        
        # Enhanced caching system
        self._cache = {
            'cluster_interaction': {'data': None, 'timestamp': None},
            'paper_distribution': {'data': None, 'timestamp': None},
            'basic_metrics': {'data': None, 'timestamp': None}
        }
        
        # Pre-compute frequently used attributes
        self.node_types = set(nx.get_node_attributes(G, 'type').values())
        self.clusters = set(nx.get_node_attributes(G, 'cluster').values())
        
        # Initialize basic metrics
        self._init_basic_metrics()

    def _init_basic_metrics(self):
        """Initialize and cache basic network metrics"""
        try:
            metrics = {
                'total_nodes': self.G.number_of_nodes(),
                'total_edges': self.G.number_of_edges(),
                'density': nx.density(self.G),
                'is_connected': nx.is_connected(self.G),
                'avg_clustering': nx.average_clustering(self.G),
                'timestamp': datetime.now()
            }
            
            if metrics['is_connected']:
                metrics.update({
                    'diameter': nx.diameter(self.G),
                    'avg_path_length': nx.average_shortest_path_length(self.G)
                })
            else:
                largest_cc = max(nx.connected_components(self.G), key=len)
                largest_subgraph = self.G.subgraph(largest_cc)
                metrics.update({
                    'largest_component_size': len(largest_cc),
                    'largest_component_ratio': len(largest_cc) / metrics['total_nodes'],
                    'diameter_largest_component': nx.diameter(largest_subgraph),
                    'avg_path_length_largest_component': nx.average_shortest_path_length(largest_subgraph)
                })
            
            self._cache['basic_metrics']['data'] = metrics
            self._cache['basic_metrics']['timestamp'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error initializing basic metrics: {str(e)}")
            raise

    def _check_cache(self, cache_key: str, max_age_hours: float = 1.0) -> bool:
        """
        Check if cached data is still valid
        
        Args:
            cache_key: Key to check in cache
            max_age_hours: Maximum age of cache in hours
            
        Returns:
            bool: True if cache is valid
        """
        cache_entry = self._cache.get(cache_key)
        if not cache_entry or not cache_entry['data'] or not cache_entry['timestamp']:
            return False
            
        age = datetime.now() - cache_entry['timestamp']
        return age.total_seconds() / 3600 < max_age_hours

    def cluster_interaction_analysis(self) -> Dict:
        """
        Analyze interactions between clusters with enhanced error handling
        
        Returns:
            dict: Detailed cluster interaction metrics and visualizations
            
        Raises:
            RuntimeError: If analysis fails
        """
        try:
            # Check cache
            if self._check_cache('cluster_interaction'):
                return self._cache['cluster_interaction']['data']

            # Extract cluster information with validation
            clusters = defaultdict(lambda: {
                'nodes': [],
                'node_types': Counter(),
                'connections': defaultdict(int),
                'internal_edges': 0,
                'external_edges': 0
            })
            
            # Process nodes with validation
            for node, data in self.G.nodes(data=True):
                cluster = data.get('cluster', 'Unknown')
                node_type = data.get('type', 'Unknown')
                
                clusters[cluster]['nodes'].append(node)
                clusters[cluster]['node_types'][node_type] += 1

            # Process edges with validation
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
                'visualizations': {},
                'metadata': {
                    'total_clusters': len(clusters),
                    'analysis_timestamp': datetime.now()
                }
            }

            # Calculate metrics
            for cluster, info in clusters.items():
                total_edges = info['internal_edges'] + info['external_edges']
                modularity = (info['internal_edges'] / total_edges 
                            if total_edges > 0 else 0)
                
                results['cluster_details'][cluster] = {
                    'total_nodes': len(info['nodes']),
                    'node_types': dict(info['node_types']),
                    'inter_cluster_connections': dict(info['connections']),
                    'internal_edges': info['internal_edges'],
                    'external_edges': info['external_edges'],
                    'modularity': modularity,
                    'isolation_index': (info['internal_edges'] / 
                                      (info['internal_edges'] + info['external_edges'] + 1e-10))
                }

            # Create interactive dropdown visualization
            cluster_names = list(clusters.keys())
            fig_dropdown = go.Figure()

            for cluster in cluster_names:
                connections = clusters[cluster]['connections']
                if connections:
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
            
            results['visualizations']['interactive_dropdown'] = fig_dropdown

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

            fig_composition.update_layout(height=500, width=800)
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

            fig_metrics.add_trace(
                go.Bar(x=metrics_df['Cluster'], y=metrics_df['Nodes'], 
                      name='Nodes'),
                row=1, col=1
            )

            fig_metrics.add_trace(
                go.Bar(x=metrics_df['Cluster'], 
                      y=metrics_df['Internal Edges'],
                      name='Internal Edges'),
                row=1, col=2
            )
            
            fig_metrics.add_trace(
                go.Bar(x=metrics_df['Cluster'], 
                      y=metrics_df['External Edges'],
                      name='External Edges'),
                row=1, col=2
            )

            fig_metrics.add_trace(
                go.Bar(x=metrics_df['Cluster'], 
                      y=metrics_df['Modularity'],
                      name='Modularity'),
                row=2, col=1
            )

            connection_counts = [len(info['connections']) 
                               for info in clusters.values()]
            fig_metrics.add_trace(
                go.Histogram(x=connection_counts, 
                           name='Connections'),
                row=2, col=2
            )

            fig_metrics.update_layout(
                height=800,
                width=1000,
                showlegend=True,
                title_text='Cluster Metrics Overview'
            )
            
            results['visualizations']['cluster_metrics'] = fig_metrics

            # Cache results
            self._cache['cluster_interaction']['data'] = results
            self._cache['cluster_interaction']['timestamp'] = datetime.now()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cluster interaction analysis failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
            
      
    def get_network_entropy(self) -> Dict:
        """
        Calculate network entropy metrics
        
        Returns:
            dict: Network entropy measures and analysis
        """
        try:
            # Degree entropy
            degree_counts = Counter(dict(self.G.degree()).values())
            total_degrees = sum(degree_counts.values())
            degree_probs = [count/total_degrees for count in degree_counts.values()]
            degree_entropy = -sum(p * np.log2(p) for p in degree_probs)
            
            # Type entropy
            type_counts = Counter(nx.get_node_attributes(self.G, 'type').values())
            total_types = sum(type_counts.values())
            type_probs = [count/total_types for count in type_counts.values()]
            type_entropy = -sum(p * np.log2(p) for p in type_probs)
            
            # Cluster entropy
            cluster_counts = Counter(nx.get_node_attributes(self.G, 'cluster').values())
            total_clusters = sum(cluster_counts.values())
            cluster_probs = [count/total_clusters for count in cluster_counts.values()]
            cluster_entropy = -sum(p * np.log2(p) for p in cluster_probs)
            
            return {
                'degree_entropy': degree_entropy,
                'type_entropy': type_entropy,
                'cluster_entropy': cluster_entropy,
                'normalized_degree_entropy': degree_entropy / np.log2(len(degree_counts)),
                'normalized_type_entropy': type_entropy / np.log2(len(type_counts)),
                'normalized_cluster_entropy': cluster_entropy / np.log2(len(cluster_counts))
            }
            
        except Exception as e:
            self.logger.error(f"Network entropy calculation failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")


    def paper_distribution_analysis(self) -> Dict:
        """
        Analyze distribution of nodes and edges across papers with enhanced features
        
        Returns:
            dict: Paper distribution analysis results and visualizations
        """
        try:
            if self._check_cache('paper_distribution'):
                return self._cache['paper_distribution']['data']

            # Group by PMID with validation
            paper_stats = defaultdict(lambda: {
                'nodes': [], 
                'edges': [], 
                'node_types': Counter(),
                'clusters': Counter(),
                'edge_weights': []
            })

            # Process nodes with validation
            for node, data in self.G.nodes(data=True):
                pmid = data.get('PMID', 'Unknown')
                paper_stats[pmid]['nodes'].append(node)
                paper_stats[pmid]['node_types'][data.get('type', 'Unknown')] += 1
                paper_stats[pmid]['clusters'][data.get('cluster', 'Unknown')] += 1

            # Process edges with validation
            for u, v, data in self.G.edges(data=True):
                pmid = self.G.nodes[u].get('PMID', 'Unknown')
                paper_stats[pmid]['edges'].append((u, v))
                paper_stats[pmid]['edge_weights'].append(data.get('weight', 1.0))

            # Enhanced paper metrics
            paper_data = [
                {
                    'PMID': pmid,
                    'Nodes': len(stats['nodes']),
                    'Edges': len(stats['edges']),
                    'Node_Types': dict(stats['node_types']),
                    'Clusters': dict(stats['clusters']),
                    'Avg_Edge_Weight': np.mean(stats['edge_weights']) if stats['edge_weights'] else 0,
                    'Max_Edge_Weight': max(stats['edge_weights']) if stats['edge_weights'] else 0,
                    'Density': len(stats['edges']) / (len(stats['nodes']) * (len(stats['nodes']) - 1) / 2) if len(stats['nodes']) > 1 else 0
                }
                for pmid, stats in paper_stats.items()
            ]

            df_papers = pd.DataFrame(paper_data)

            # Enhanced visualizations
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

            # Enhanced scatter plot with hover information
            fig.add_trace(
                go.Scatter(
                    x=df_papers['Nodes'],
                    y=df_papers['Edges'],
                    mode='markers',
                    text=[f"PMID: {row['PMID']}<br>"
                          f"Density: {row['Density']:.3f}<br>"
                          f"Avg Edge Weight: {row['Avg_Edge_Weight']:.2f}"
                          for _, row in df_papers.iterrows()],
                    marker=dict(
                        size=10,
                        color=df_papers['Density'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Network Density')
                    ),
                    name='Papers'
                ),
                row=2, col=1
            )

            # Node types across papers with enhanced visualization
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

            # Additional analysis metrics
            results = {
                'summary': {
                    'total_papers': len(paper_stats),
                    'avg_nodes_per_paper': df_papers['Nodes'].mean(),
                    'median_nodes_per_paper': df_papers['Nodes'].median(),
                    'avg_edges_per_paper': df_papers['Edges'].mean(),
                    'median_edges_per_paper': df_papers['Edges'].median(),
                    'avg_density': df_papers['Density'].mean(),
                    'avg_edge_weight': df_papers['Avg_Edge_Weight'].mean(),
                    'timestamp': datetime.now()
                },
                'paper_details': paper_data,
                'visualizations': {
                    'main_overview': fig
                },
                'statistics': {
                    'node_correlation': {
                        'nodes_edges_correlation': df_papers['Nodes'].corr(df_papers['Edges']),
                        'nodes_density_correlation': df_papers['Nodes'].corr(df_papers['Density'])
                    },
                    'distributions': {
                        'nodes_distribution': df_papers['Nodes'].describe().to_dict(),
                        'edges_distribution': df_papers['Edges'].describe().to_dict(),
                        'density_distribution': df_papers['Density'].describe().to_dict()
                    }
                }
            }

            # Cache results
            self._cache['paper_distribution']['data'] = results
            self._cache['paper_distribution']['timestamp'] = datetime.now()

            return results

        except Exception as e:
            self.logger.error(f"Paper distribution analysis failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def get_filtered_view(self, node_types: Optional[List[str]] = None, 
                         min_degree: int = 1, 
                         min_weight: float = 0.0) -> Dict:
        """
        Get filtered view of the network with enhanced metrics
        
        Args:
            node_types: List of node types to include
            min_degree: Minimum node degree
            min_weight: Minimum edge weight
            
        Returns:
            dict: Filtered network analysis results
        """
        try:
            H = self.G.copy()
            
            # Apply filters with validation
            if node_types:
                invalid_types = set(node_types) - self.node_types
                if invalid_types:
                    self.logger.warning(f"Invalid node types: {invalid_types}")
                
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

            # Enhanced metrics calculation
            type_dist = Counter(nx.get_node_attributes(H, 'type').values())
            cluster_dist = Counter(nx.get_node_attributes(H, 'cluster').values())
            degree_dist = [d for n, d in H.degree()]
            
            # Component analysis
            components = list(nx.connected_components(H))
            component_sizes = [len(c) for c in components]
            
            # Calculate advanced metrics
            metrics = {
                'nodes': H.number_of_nodes(),
                'edges': H.number_of_edges(),
                'density': nx.density(H),
                'avg_clustering': nx.average_clustering(H),
                'avg_degree': np.mean(degree_dist) if degree_dist else 0,
                'components': len(components),
                'largest_component_size': max(component_sizes) if component_sizes else 0,
                'isolated_nodes': len([n for n, d in H.degree() if d == 0])
            }

            # Enhanced visualizations
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

            # Component size distribution
            fig.add_trace(
                go.Histogram(
                    x=component_sizes,
                    name='Component Sizes',
                    nbinsx=20
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
                'metrics': metrics,
                'distributions': {
                    'node_types': dict(type_dist),
                    'clusters': dict(cluster_dist),
                    'degrees': degree_dist,
                    'component_sizes': component_sizes
                },
                'centrality': {
                    'degree': nx.degree_centrality(H),
                    'betweenness': nx.betweenness_centrality(H),
                    'eigenvector': nx.eigenvector_centrality(H, max_iter=1000)
                },
                'visualization': fig
            }

        except Exception as e:
            self.logger.error(f"Filtered view analysis failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def explore_node(self, node_id: str) -> Dict:
        """
        Detailed exploration of a specific node with enhanced metrics
        
        Args:
            node_id: ID of the node to explore
            
        Returns:
            dict: Comprehensive node analysis results
            
        Raises:
            ValueError: If node_id is not found
        """
        if node_id not in self.G:
            raise ValueError(f"Node {node_id} not found in network")
            
        try:
            # Get node information
            node_data = self.G.nodes[node_id]
            neighbors = list(self.G.neighbors(node_id))
            
            # Enhanced metrics calculation
            metrics = {
                'degree': self.G.degree(node_id),
                'degree_centrality': nx.degree_centrality(self.G)[node_id],
                'betweenness_centrality': nx.betweenness_centrality(self.G)[node_id],
                'closeness_centrality': nx.closeness_centrality(self.G)[node_id],
                'clustering_coefficient': nx.clustering(self.G, node_id),
                'core_number': nx.core_number(self.G)[node_id],
                'eccentricity': nx.eccentricity(self.G, node_id) if nx.is_connected(self.G) else None
            }
            
            # Enhanced neighborhood analysis
            neighborhood = {
                'neighbors': neighbors,
                'neighbor_types': Counter(self.G.nodes[n]['type'] for n in neighbors),
                'neighbor_clusters': Counter(self.G.nodes[n]['cluster'] for n in neighbors),
                'neighbor_degrees': [self.G.degree(n) for n in neighbors],
                'edge_weights': [self.G[node_id][n].get('weight', 1.0) for n in neighbors],
                'common_neighbors': {
                    n: len(list(nx.common_neighbors(self.G, node_id, n)))
                    for n in neighbors
                }
            }
            
            # Get subgraph for visualization
            subgraph = self.G.subgraph(neighbors + [node_id])
            pos = nx.spring_layout(subgraph)
            
            # Create enhanced network visualization
            edge_traces = []
            edge_weights = []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = subgraph[edge[0]][edge[1]].get('weight', 1.0)
                edge_weights.append(weight)
                
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(
                            width=weight*2,
                            color='rgba(136, 136, 136, 0.7)'
                        ),
                        hoverinfo='text',
                        text=f'Weight: {weight:.2f}',
                        mode='lines'
                    )
                )
            
            # Create enhanced node traces
            node_trace = go.Scatter(
                x=[pos[node][0] for node in subgraph.nodes()],
                y=[pos[node][1] for node in subgraph.nodes()],
                mode='markers+text',
                hoverinfo='text',
                text=[
                    f"{'Central Node: ' if node == node_id else 'Neighbor: '}{node}<br>"
                    f"Type: {subgraph.nodes[node]['type']}<br>"
                    f"Cluster: {subgraph.nodes[node]['cluster']}<br>"
                    f"Degree: {subgraph.degree(node)}"
                    for node in subgraph.nodes()
                ],
                marker=dict(
                    size=[20 if node == node_id else 15 for node in subgraph.nodes()],
                    color=['red' if node == node_id else 'lightblue' for node in subgraph.nodes()],
                    line=dict(width=2)
                )
            )
            
            # Create network visualization
            fig_network = go.Figure(data=edge_traces + [node_trace])
            fig_network.update_layout(
                title=f'Neighborhood Network for Node {node_id}',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            # Create enhanced distribution visualizations
            fig_dist = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Neighbor Type Distribution',
                    'Neighbor Cluster Distribution',
                    'Edge Weight Distribution',
                    'Common Neighbors Distribution'
                )
            )
            
            # Neighbor type distribution
            fig_dist.add_trace(
                go.Bar(
                    x=list(neighborhood['neighbor_types'].keys()),
                    y=list(neighborhood['neighbor_types'].values()),
                    name='Types'
                ),
                row=1, col=1
            )
            
            # Neighbor cluster distribution
            fig_dist.add_trace(
                go.Bar(
                    x=list(neighborhood['neighbor_clusters'].keys()),
                    y=list(neighborhood['neighbor_clusters'].values()),
                    name='Clusters'
                ),
                row=1, col=2
            )
            
            # Edge weight distribution
            fig_dist.add_trace(
                go.Histogram(
                    x=neighborhood['edge_weights'],
                    name='Edge Weights',
                    nbinsx=20
                ),
                row=2, col=1
            )
            
            # Common neighbors distribution
            fig_dist.add_trace(
                go.Histogram(
                    x=list(neighborhood['common_neighbors'].values()),
                    name='Common Neighbors',
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
                'neighborhood': neighborhood,
                'visualizations': {
                    'network': fig_network,
                    'distributions': fig_dist
                },
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Node exploration failed for {node_id}: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def get_network_summary(self) -> Dict:
        """
        Generate comprehensive network summary with enhanced metrics and visualizations
        
        Returns:
            dict: Enhanced network summary statistics and visualizations
        """
        try:
            # Check cache for basic metrics
            if not self._check_cache('basic_metrics'):
                self._init_basic_metrics()
                
            metrics = self._cache['basic_metrics']['data'].copy()
            
            # Calculate centrality metrics
            centrality_metrics = {
                'degree_centrality': nx.degree_centrality(self.G),
                'betweenness_centrality': nx.betweenness_centrality(self.G),
                'closeness_centrality': nx.closeness_centrality(self.G),
                'eigenvector_centrality': nx.eigenvector_centrality(self.G, max_iter=1000)
            }
            
            # Get degree sequence for analysis
            degree_sequence = [d for n, d in self.G.degree()]
            
            # Calculate distributions
            distributions = {
                'node_types': Counter(nx.get_node_attributes(self.G, 'type').values()),
                'clusters': Counter(nx.get_node_attributes(self.G, 'cluster').values()),
                'degrees': Counter(degree_sequence),
                'component_sizes': [len(c) for c in nx.connected_components(self.G)]
            }
            
            # Create main visualization dashboard
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
            
            # Degree distribution with power law fit
            degrees = list(distributions['degrees'].keys())
            counts = list(distributions['degrees'].values())
            fig.add_trace(
                go.Scatter(
                    x=degrees,
                    y=counts,
                    mode='markers',
                    name='Degree Distribution'
                ),
                row=2, col=1
            )
            
            # Component size distribution
            if len(distributions['component_sizes']) > 1:
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
            
            # Create centrality correlation heatmap
            centrality_df = pd.DataFrame(centrality_metrics)
            fig_corr = go.Figure(data=go.Heatmap(
                z=centrality_df.corr(),
                x=centrality_df.columns,
                y=centrality_df.columns,
                colorscale='RdBu'
            ))
            
            fig_corr.update_layout(
                title='Centrality Measure Correlations',
                height=500
            )
            
            # Calculate enhanced network metrics
            enhanced_metrics = {
                **metrics,
                'degree_statistics': {
                    'mean': np.mean(degree_sequence),
                    'median': np.median(degree_sequence),
                    'std': np.std(degree_sequence),
                    'skewness': stats.skew(degree_sequence),
                    'kurtosis': stats.kurtosis(degree_sequence)
                },
                'centrality_correlations': centrality_df.corr().to_dict(),
                'node_type_proportions': {
                    k: v/sum(distributions['node_types'].values())
                    for k, v in distributions['node_types'].items()
                },
                'cluster_proportions': {
                    k: v/sum(distributions['clusters'].values())
                    for k, v in distributions['clusters'].items()
                }
            }
            
            # Compile final results
            results = {
                'metrics': enhanced_metrics,
                'distributions': distributions,
                'centrality_metrics': centrality_metrics,
                'visualizations': {
                    'main_overview': fig,
                    'centrality_correlation': fig_corr
                },
                'metadata': {
                    'analysis_timestamp': datetime.now(),
                    'graph_name': getattr(self.G, 'name', 'Unnamed'),
                    'graph_type': type(self.G).__name__
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Network summary analysis failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
            
    def get_network_entropy(self) -> Dict:
        """
        Calculate comprehensive network entropy metrics
        
        Returns:
            dict: Network entropy measures and analysis
        """
        try:
            # Calculate degree entropy
            degree_counts = Counter(dict(self.G.degree()).values())
            total_degrees = sum(degree_counts.values())
            degree_probs = [count/total_degrees for count in degree_counts.values()]
            degree_entropy = -sum(p * np.log2(p) for p in degree_probs)
            
            # Calculate type entropy
            type_counts = Counter(nx.get_node_attributes(self.G, 'type').values())
            total_types = sum(type_counts.values())
            type_probs = [count/total_types for count in type_counts.values()]
            type_entropy = -sum(p * np.log2(p) for p in type_probs)
            
            # Calculate cluster entropy
            cluster_counts = Counter(nx.get_node_attributes(self.G, 'cluster').values())
            total_clusters = sum(cluster_counts.values())
            cluster_probs = [count/total_clusters for count in cluster_counts.values()]
            cluster_entropy = -sum(p * np.log2(p) for p in cluster_probs)
            
            # Calculate normalized entropies and prepare results
            results = {
                'entropy_measures': {
                    'degree_entropy': degree_entropy,
                    'type_entropy': type_entropy,
                    'cluster_entropy': cluster_entropy,
                    'normalized_degree_entropy': degree_entropy / np.log2(len(degree_counts)),
                    'normalized_type_entropy': type_entropy / np.log2(len(type_counts)),
                    'normalized_cluster_entropy': cluster_entropy / np.log2(len(cluster_counts))
                },
                'distribution_details': {
                    'degree_distribution': dict(degree_counts),
                    'type_distribution': dict(type_counts),
                    'cluster_distribution': dict(cluster_counts)
                },
                'metadata': {
                    'total_degrees': total_degrees,
                    'unique_degrees': len(degree_counts),
                    'unique_types': len(type_counts),
                    'unique_clusters': len(cluster_counts)
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Network entropy calculation failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
            
    def get_temporal_analysis(self) -> Dict:
        """
        Perform temporal analysis of the network based on PMID information
        
        Returns:
            dict: Temporal network analysis results and visualizations
        """
        try:
            # Extract temporal information
            pmid_data = defaultdict(lambda: {
                'nodes': [],
                'edges': [],
                'node_types': Counter(),
                'clusters': Counter()
            })
            
            # Process nodes
            for node, data in self.G.nodes(data=True):
                pmid = data.get('PMID', 'Unknown')
                pmid_data[pmid]['nodes'].append(node)
                pmid_data[pmid]['node_types'][data.get('type', 'Unknown')] += 1
                pmid_data[pmid]['clusters'][data.get('cluster', 'Unknown')] += 1
            
            # Process edges
            for u, v, data in self.G.edges(data=True):
                pmid = self.G.nodes[u].get('PMID', 'Unknown')
                pmid_data[pmid]['edges'].append((u, v))
            
            # Create temporal metrics
            temporal_metrics = []
            for pmid, data in pmid_data.items():
                metrics = {
                    'PMID': pmid,
                    'nodes': len(data['nodes']),
                    'edges': len(data['edges']),
                    'density': len(data['edges']) / (len(data['nodes']) * (len(data['nodes']) - 1) / 2) if len(data['nodes']) > 1 else 0,
                    'node_types': dict(data['node_types']),
                    'clusters': dict(data['clusters'])
                }
                temporal_metrics.append(metrics)
            
            # Convert to DataFrame for analysis
            df_temporal = pd.DataFrame(temporal_metrics)
            
            # Create visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Nodes Over Time',
                    'Edges Over Time',
                    'Network Density Over Time',
                    'Node Types Over Time'
                )
            )
            
            # Plot nodes over time
            fig.add_trace(
                go.Scatter(
                    x=df_temporal['PMID'],
                    y=df_temporal['nodes'],
                    mode='lines+markers',
                    name='Nodes'
                ),
                row=1, col=1
            )
            
            # Plot edges over time
            fig.add_trace(
                go.Scatter(
                    x=df_temporal['PMID'],
                    y=df_temporal['edges'],
                    mode='lines+markers',
                    name='Edges'
                ),
                row=1, col=2
            )
            
            # Plot density over time
            fig.add_trace(
                go.Scatter(
                    x=df_temporal['PMID'],
                    y=df_temporal['density'],
                    mode='lines+markers',
                    name='Density'
                ),
                row=2, col=1
            )
            
            # Prepare node type evolution
            node_types = set()
            for types in df_temporal['node_types']:
                node_types.update(types.keys())
            
            for node_type in node_types:
                counts = [types.get(node_type, 0) for types in df_temporal['node_types']]
                fig.add_trace(
                    go.Scatter(
                        x=df_temporal['PMID'],
                        y=counts,
                        mode='lines+markers',
                        name=node_type
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text='Temporal Network Evolution'
            )
            
            # Compile results
            results = {
                'temporal_metrics': temporal_metrics,
                'summary_statistics': {
                    'total_pmids': len(pmid_data),
                    'avg_nodes_per_pmid': df_temporal['nodes'].mean(),
                    'avg_edges_per_pmid': df_temporal['edges'].mean(),
                    'avg_density': df_temporal['density'].mean()
                },
                'trend_analysis': {
                    'node_growth_rate': np.polyfit(range(len(df_temporal)), df_temporal['nodes'], 1)[0],
                    'edge_growth_rate': np.polyfit(range(len(df_temporal)), df_temporal['edges'], 1)[0],
                    'density_trend': np.polyfit(range(len(df_temporal)), df_temporal['density'], 1)[0]
                },
                'visualizations': {
                    'temporal_evolution': fig
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
