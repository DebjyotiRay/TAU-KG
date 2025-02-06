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
        """
        Enhanced Network Explorer with comprehensive analysis capabilities
        
        Args:
            G (networkx.Graph): Input network graph
        """
        self.G = G
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cached analysis results
        self._cluster_interaction_cache = None
        self._paper_distribution_cache = None
    
    def cluster_interaction_analysis(self):
        """
        Comprehensive analysis of interactions between clusters
        
        Returns:
            dict: Detailed cluster interaction metrics and visualizations
        """
        # Check cache first
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
                    'connections': defaultdict(int)
                }
            clusters[cluster]['nodes'].append(node)
            clusters[cluster]['node_types'][data.get('type', 'Unknown')] += 1
        
        # Analyze inter-cluster connections
        for u, v, data in self.G.edges(data=True):
            cluster_u = self.G.nodes[u].get('cluster', 'Unknown')
            cluster_v = self.G.nodes[v].get('cluster', 'Unknown')
            
            if cluster_u != cluster_v:
                # Track inter-cluster connections
                clusters[cluster_u]['connections'][cluster_v] += 1
                clusters[cluster_v]['connections'][cluster_u] += 1
        
        # Prepare results
        results = {
            'cluster_details': {},
            'visualizations': {}
        }
        
        # Cluster-level metrics
        for cluster, info in clusters.items():
            results['cluster_details'][cluster] = {
                'total_nodes': len(info['nodes']),
                'node_types': dict(info['node_types']),
                'inter_cluster_connections': dict(info['connections'])
            }
        
        # Visualization: Cluster Interaction Heatmap
        cluster_names = list(clusters.keys())
        interaction_matrix = np.zeros((len(cluster_names), len(cluster_names)))
        
        for i, cluster1 in enumerate(cluster_names):
            for j, cluster2 in enumerate(cluster_names):
                if cluster1 != cluster2:
                    interaction_matrix[i, j] = clusters[cluster1]['connections'].get(cluster2, 0)
        
        # Plotly Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=cluster_names,
            y=cluster_names,
            colorscale='Viridis'
        ))
        fig_heatmap.update_layout(
            title='Inter-Cluster Interaction Heatmap',
            xaxis_title='Target Cluster',
            yaxis_title='Source Cluster',
            height=600,
            width=800
        )
        results['visualizations']['cluster_interaction_heatmap'] = fig_heatmap
        
        # Visualization: Cluster Node Type Distribution
        fig_node_types = go.Figure()
        for cluster, info in clusters.items():
            fig_node_types.add_trace(go.Bar(
                x=list(info['node_types'].keys()),
                y=list(info['node_types'].values()),
                name=cluster
            ))
        
        fig_node_types.update_layout(
            title='Node Type Distribution Across Clusters',
            xaxis_title='Node Type',
            yaxis_title='Count',
            barmode='group',
            height=600,
            width=800
        )
        results['visualizations']['node_type_distribution'] = fig_node_types
        
        # Cache and return results
        self._cluster_interaction_cache = results
        return results
    
    def paper_distribution_analysis(self):
        """
        Comprehensive analysis of node and edge distribution across papers
        
        Returns:
            dict: Detailed paper distribution metrics and visualizations
        """
        # Check cache first
        if self._paper_distribution_cache:
            return self._paper_distribution_cache
        
        # Extract PMID information
        paper_distribution = defaultdict(lambda: {
            'nodes': [],
            'edges': [],
            'node_types': Counter(),
            'unique_node_types': set()
        })
        
        # Process nodes
        for node, data in self.G.nodes(data=True):
            pmid = str(data.get('PMID', 'Unknown'))
            paper_distribution[pmid]['nodes'].append(node)
            paper_distribution[pmid]['node_types'][data.get('type', 'Unknown')] += 1
            paper_distribution[pmid]['unique_node_types'].add(data.get('type', 'Unknown'))
        
        # Process edges
        for u, v, data in self.G.edges(data=True):
            # Find PMID of source node
            pmid = str(self.G.nodes[u].get('PMID', 'Unknown'))
            paper_distribution[pmid]['edges'].append((u, v))
        
        # Prepare results
        results = {
            'paper_details': {},
            'visualizations': {}
        }
        
        # Aggregate paper-level metrics
        papers_by_size = []
        for pmid, info in paper_distribution.items():
            paper_info = {
                'PMID': pmid,
                'total_nodes': len(info['nodes']),
                'total_edges': len(info['edges']),
                'node_types': dict(info['node_types']),
                'unique_node_types': list(info['unique_node_types'])
            }
            papers_by_size.append(paper_info)
            results['paper_details'][pmid] = paper_info
        
        # Convert to DataFrame for easier manipulation
        df_papers = pd.DataFrame(papers_by_size)
        
        # Visualization: Nodes and Edges per Paper
        fig_paper_stats = make_subplots(rows=1, cols=2, 
                                        subplot_titles=('Nodes per Paper', 'Edges per Paper'))
        
        # Nodes per paper histogram
        fig_paper_stats.add_trace(
            go.Histogram(x=df_papers['total_nodes'], name='Nodes'),
            row=1, col=1
        )
        
        # Edges per paper histogram
        fig_paper_stats.add_trace(
            go.Histogram(x=df_papers['total_edges'], name='Edges'),
            row=1, col=2
        )
        
        fig_paper_stats.update_layout(
            title='Distribution of Nodes and Edges Across Papers',
            height=600,
            width=800
        )
        results['visualizations']['nodes_edges_distribution'] = fig_paper_stats
        
        # Visualization: Node Type Composition
        # Aggregate node type information
        node_type_composition = defaultdict(int)
        for pmid, info in paper_distribution.items():
            for node_type, count in info['node_types'].items():
                node_type_composition[node_type] += count
        
        fig_node_type_pie = go.Figure(data=[go.Pie(
            labels=list(node_type_composition.keys()),
            values=list(node_type_composition.values()),
            hole=.3
        )])
        fig_node_type_pie.update_layout(
            title='Overall Node Type Composition',
            height=600,
            width=800
        )
        results['visualizations']['node_type_composition'] = fig_node_type_pie
        
        # Scatter plot: Nodes vs Edges per Paper
        fig_nodes_vs_edges = go.Figure(data=go.Scatter(
            x=df_papers['total_nodes'],
            y=df_papers['total_edges'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_papers['total_nodes'],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"PMID: {pmid}" for pmid in df_papers['PMID']]
        ))
        fig_nodes_vs_edges.update_layout(
            title='Nodes vs Edges per Paper',
            xaxis_title='Number of Nodes',
            yaxis_title='Number of Edges',
            height=600,
            width=800
        )
        results['visualizations']['nodes_vs_edges_scatter'] = fig_nodes_vs_edges
        
        # Advanced statistical analysis
        results['statistics'] = {
            'total_papers': len(df_papers),
            'avg_nodes_per_paper': df_papers['total_nodes'].mean(),
            'median_nodes_per_paper': df_papers['total_nodes'].median(),
            'avg_edges_per_paper': df_papers['total_edges'].mean(),
            'median_edges_per_paper': df_papers['total_edges'].median(),
            'node_type_distribution': dict(node_type_composition)
        }
        
        # Cache and return results
        self._paper_distribution_cache = results
        return results
    
    def advanced_node_exploration(self, node_id=None):
        """
        Comprehensive node exploration with advanced insights
        
        Args:
            node_id (str, optional): Specific node to explore in depth
        
        Returns:
            dict: Detailed node exploration results
        """
        # If no specific node is provided, return overview
        if node_id is None:
            return self._global_node_overview()
        
        # Detailed node analysis
        if node_id not in self.G:
            raise ValueError(f"Node {node_id} not found in the network")
        
        # Node-level metrics
        node_data = self.G.nodes[node_id]
        node_metrics = {
            'basic_info': dict(node_data),
            'degree': self.G.degree(node_id),
            'neighbors': list(self.G.neighbors(node_id)),
            'centrality_metrics': {
                'degree_centrality': nx.degree_centrality(self.G)[node_id],
                'betweenness_centrality': nx.betweenness_centrality(self.G)[node_id],
                'closeness_centrality': nx.closeness_centrality(self.G)[node_id],
                'eigenvector_centrality': nx.eigenvector_centrality(self.G)[node_id]
            }
        }
        
        # Neighborhood analysis
        neighborhood = self.G.subgraph(
            list(self.G.neighbors(node_id)) + [node_id]
        )
        
        node_metrics['neighborhood'] = {
            'total_neighbors': len(node_metrics['neighbors']),
            'neighborhood_density': nx.density(neighborhood),
            'neighbor_types': Counter(
                self.G.nodes[n].get('type', 'Unknown') for n in node_metrics['neighbors']
            ),
            'neighbor_clusters': Counter(
                self.G.nodes[n].get('cluster', 'Unknown') for n in node_metrics['neighbors']
            )
        }
        
        # Visualization of neighborhood
        pos = nx.spring_layout(neighborhood)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in neighborhood.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in neighborhood.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Highlight the central node
            if node == node_id:
                node_color.append('red')
                node_text.append(f"Central Node: {node}")
            else:
                node_color.append('blue')
                node_text.append(f"Neighbor: {node}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=10
            )
        )
        
        # Create network graph
        fig_neighborhood = go.Figure(data=[edge_trace, node_trace])
        fig_neighborhood.update_layout(
            title=f'Neighborhood of Node {node_id}',
            showlegend=False,
            height=600,
            width=800
        )
        
        node_metrics['visualizations'] = {
            'neighborhood_graph': fig_neighborhood
        }
        
        return node_metrics
    
    def _global_node_overview(self):
        """
        Provide a global overview of nodes in the network
        
        Returns:
            dict: Global node overview metrics
        """
        # Global node type distribution
        node_type_dist = Counter(
            self.G.nodes[node].get('type', 'Unknown') for node in self.G.nodes()
        )
        
        # Node degree distribution
        degrees = [d for n, d in self.G.degree()]
        
        # Visualizations
        # Node Type Distribution Pie Chart
        fig_node_types = go.Figure(data=[go.Pie(
            labels=list(node_type_dist.keys()),
            values=list(node_type_dist.values()),
            hole=.3
        )])
        fig_node_types.update_layout(
            title='Global Node Type Distribution',
            height=600,
            width=800
        )
        
        # Degree Distribution Histogram
        fig_degree_dist = go.Figure(data=[go.Histogram(
            x=degrees,
            nbinsx=50
        )])
        fig_degree_dist.update_layout(
            title='Node Degree Distribution',
            xaxis_title='Degree',
            yaxis_title='Frequency',
            height=600,
            width=800
        )
        
        return {
            'node_type_distribution': dict(node_type_dist),
            'degree_distribution': {
                'mean': np.mean(degrees),
                'median': np.median(degrees),
                'min': min(degrees),
                'max': max(degrees)
            },
            'visualizations': {
                'node_type_pie': fig_node_types,
                'degree_distribution': fig_degree_dist
            }
        }
    def enhanced_cluster_interaction_visualization(self):
        """
        Create an enhanced, interactive visualization of cluster interactions
        
        Returns:
            dict: Interactive cluster interaction visualizations
        """
        # Reuse the cluster interaction analysis method
        cluster_analysis = self.cluster_interaction_analysis()
        
        # Prepare data for dropdown visualization
        clusters = list(cluster_analysis['cluster_details'].keys())
        
        # Create interactive dropdown-based visualization
        fig_dropdown = go.Figure()
        
        # Prepare data for each cluster
        for cluster in clusters:
            # Get connections for this cluster
            connections = cluster_analysis['cluster_details'][cluster]['inter_cluster_connections']
            
            # Prepare connection data
            connection_data = []
            for target_cluster, connection_count in connections.items():
                connection_data.append({
                    'Source Cluster': cluster,
                    'Target Cluster': target_cluster,
                    'Connection Count': connection_count
                })
            
            # Convert to DataFrame for easier plotting
            df_connections = pd.DataFrame(connection_data)
            
            # Add trace for this cluster (initially invisible)
            fig_dropdown.add_trace(
                go.Bar(
                    x=df_connections['Target Cluster'],
                    y=df_connections['Connection Count'],
                    name=cluster,
                    visible=(cluster == clusters[0])  # Only first cluster visible by default
                )
            )
        
        # Create dropdown menu
        dropdown_buttons = []
        for i, cluster in enumerate(clusters):
            visibility = [False] * len(clusters)
            visibility[i] = True
            
            dropdown_buttons.append(
                dict(
                    method='update',
                    label=cluster,
                    args=[{'visible': visibility},
                        {'title': f'Inter-Cluster Connections for {cluster}'}]
                )
            )
        
        # Update layout with dropdown
        fig_dropdown.update_layout(
            updatemenus=[{
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.5,
                'xanchor': 'center',
                'y': 1.15,
                'yanchor': 'top'
            }],
            title=f'Inter-Cluster Connections for {clusters[0]}',
            xaxis_title='Target Cluster',
            yaxis_title='Connection Count',
            height=600,
            width=800
        )
        
        # Combine visualizations
        cluster_analysis['visualizations']['interactive_dropdown'] = fig_dropdown
        
        return cluster_analysis

    def additional_network_exploration_methods(self):
        """
        Additional advanced network exploration techniques
        
        Returns:
            dict: Various advanced network exploration analyses
        """
        # Network Topology Analysis
        def network_topology_analysis():
            """Analyze network topology characteristics"""
            # Compute various topological metrics
            topology_metrics = {
                'Average Clustering Coefficient': nx.average_clustering(self.G),
                'Global Efficiency': nx.global_efficiency(self.G),
                'Average Path Length': nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else float('inf'),
                'Network Diameter': nx.diameter(self.G) if nx.is_connected(self.G) else float('inf'),
                'Is Connected': nx.is_connected(self.G),
                'Number of Connected Components': nx.number_connected_components(self.G)
            }
            
            # Visualize component sizes
            components = list(nx.connected_components(self.G))
            component_sizes = [len(comp) for comp in components]
            
            fig_component_sizes = go.Figure(data=[go.Histogram(
                x=component_sizes,
                nbinsx=20
            )])
            fig_component_sizes.update_layout(
                title='Connected Component Size Distribution',
                xaxis_title='Component Size',
                yaxis_title='Frequency',
                height=600,
                width=800
            )
            
            return {
                'metrics': topology_metrics,
                'visualizations': {
                    'component_size_distribution': fig_component_sizes
                }
            }
        
        # Network Centrality Landscape
        def network_centrality_landscape():
            """
            Comprehensive centrality analysis
            
            Returns:
                dict: Detailed centrality metrics and visualizations
            """
            # Compute various centrality measures
            centrality_metrics = {
                'Degree Centrality': nx.degree_centrality(self.G),
                'Betweenness Centrality': nx.betweenness_centrality(self.G),
                'Closeness Centrality': nx.closeness_centrality(self.G),
                'Eigenvector Centrality': nx.eigenvector_centrality(self.G),
                'PageRank': nx.pagerank(self.G)
            }
            
            # Prepare visualization data
            centrality_df = pd.DataFrame(centrality_metrics)
            
            # Correlation heatmap of centrality measures
            fig_centrality_corr = go.Figure(data=go.Heatmap(
                z=centrality_df.corr().values,
                x=centrality_df.columns,
                y=centrality_df.columns,
                colorscale='RdBu_r'
            ))
            fig_centrality_corr.update_layout(
                title='Centrality Measures Correlation',
                height=600,
                width=800
            )
            
            # Scatter matrix of centrality measures
            fig_scatter_matrix = px.scatter_matrix(
                centrality_df, 
                dimensions=list(centrality_metrics.keys()),
                title='Centrality Measures Scatter Matrix'
            )
            fig_scatter_matrix.update_layout(height=1000, width=1000)
            
            return {
                'metrics': centrality_metrics,
                'visualizations': {
                    'centrality_correlation_heatmap': fig_centrality_corr,
                    'centrality_scatter_matrix': fig_scatter_matrix
                }
            }
        
        # Network Resilience Analysis
        def network_resilience_analysis():
            """
            Analyze network resilience through node removal simulations
            
            Returns:
                dict: Network resilience metrics and visualizations
            """
            # Simulate node removals
            def simulate_node_removal(remove_strategy):
                """
                Simulate network degradation by removing nodes
                
                Args:
                    remove_strategy (str): Strategy for node removal
                
                Returns:
                    list: Largest component sizes after each removal
                """
                G_copy = self.G.copy()
                largest_component_sizes = [len(max(nx.connected_components(G_copy), key=len))]
                
                if remove_strategy == 'random':
                    nodes = list(G_copy.nodes())
                    np.random.shuffle(nodes)
                elif remove_strategy == 'highest_degree':
                    nodes = sorted(G_copy.degree(), key=lambda x: x[1], reverse=True)
                    nodes = [n[0] for n in nodes]
                
                for node in nodes[:min(len(nodes), 100)]:
                    G_copy.remove_node(node)
                    largest_component_sizes.append(
                        len(max(nx.connected_components(G_copy), key=len))
                    )
                
                return largest_component_sizes
            
            # Simulate different removal strategies
            random_removal = simulate_node_removal('random')
            targeted_removal = simulate_node_removal('highest_degree')
            
            # Visualization
            fig_resilience = go.Figure()
            fig_resilience.add_trace(go.Scatter(
                y=random_removal,
                mode='lines+markers',
                name='Random Node Removal'
            ))
            fig_resilience.add_trace(go.Scatter(
                y=targeted_removal,
                mode='lines+markers',
                name='Highest Degree Node Removal'
            ))
            
            fig_resilience.update_layout(
                title='Network Resilience under Node Removal',
                xaxis_title='Number of Nodes Removed',
                yaxis_title='Largest Component Size',
                height=600,
                width=800
            )
            
            return {
                'metrics': {
                    'random_removal_degradation': random_removal,
                    'targeted_removal_degradation': targeted_removal
                },
                'visualizations': {
                    'network_resilience': fig_resilience
                }
            }
        
        # Temporal Evolution (if temporal data is available)
        def temporal_network_evolution():
            """
            Analyze network evolution over time
            
            Returns:
                dict: Temporal network evolution metrics
            """
            # Check for temporal attributes (PMID)
            temporal_nodes = [
                node for node, data in self.G.nodes(data=True) 
                if 'PMID' in data
            ]
            
            if not temporal_nodes:
                return {"error": "No temporal data available"}
            
            # Group nodes by PMID
            pmid_groups = defaultdict(list)
            for node, data in self.G.nodes(data=True):
                if 'PMID' in data:
                    pmid_groups[data['PMID']].append(node)
            
            # Analyze network metrics over time
            temporal_metrics = []
            sorted_pmids = sorted(pmid_groups.keys())
            
            for pmid in sorted_pmids:
                subgraph = self.G.subgraph(pmid_groups[pmid])
                temporal_metrics.append({
                    'PMID': pmid,
                    'Nodes': len(subgraph.nodes()),
                    'Edges': len(subgraph.edges()),
                    'Density': nx.density(subgraph),
                    'Avg Clustering': nx.average_clustering(subgraph)
                })
            
            # Convert to DataFrame
            df_temporal = pd.DataFrame(temporal_metrics)
            
            # Visualization of network metrics over time
            fig_temporal = make_subplots(rows=2, cols=2, 
                                        subplot_titles=('Nodes', 'Edges', 'Network Density', 'Clustering Coefficient'))
            
            # Nodes over time
            fig_temporal.add_trace(
                go.Scatter(x=df_temporal['PMID'], y=df_temporal['Nodes'], mode='lines+markers'),
                row=1, col=1
            )
            
            # Edges over time
            fig_temporal.add_trace(
                go.Scatter(x=df_temporal['PMID'], y=df_temporal['Edges'], mode='lines+markers'),
                row=1, col=2
            )
            
            # Density over time
            fig_temporal.add_trace(
                go.Scatter(x=df_temporal['PMID'], y=df_temporal['Density'], mode='lines+markers'),
                row=2, col=1
            )
            
            # Clustering coefficient over time
            fig_temporal.add_trace(
                go.Scatter(x=df_temporal['PMID'], y=df_temporal['Avg Clustering'], mode='lines+markers'),
                row=2, col=2
            )
            
            fig_temporal.update_layout(
                title='Network Evolution Over Time',
                height=800,
                width=1000
            )
            
            return {
                'metrics': temporal_metrics,
                'visualizations': {
                    'temporal_evolution': fig_temporal
                }
            }
        
        # Combine all analyses
        return {
            'topology_analysis': network_topology_analysis(),
            'centrality_landscape': network_centrality_landscape(),
            'resilience_analysis': network_resilience_analysis(),
            'temporal_evolution': temporal_network_evolution()
        }

