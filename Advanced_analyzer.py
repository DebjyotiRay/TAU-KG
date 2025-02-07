import networkx as nx
import numpy as np
import pandas as pd
import logging
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, Counter

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# Node embedding libraries
try:
    import node2vec
    import gensim
except ImportError:
    node2vec = None
    gensim = None

class NetworkAdvancedExtensions:
    def __init__(self, G):
        """
        Advanced network analysis extensions
        
        Args:
            G (networkx.Graph): Input network graph
        """
        self.G = G
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cached results to improve performance
        self._embedding_cache = {}
        self._centrality_cache = {}

    def network_embedding(self, method='node2vec', dimensions=2):
        """
        Generate network node embeddings with visualization
        
        Args:
            method (str): Embedding technique
            dimensions (int): Embedding dimensions
        
        Returns:
            dict: Embedding results with visualization
        """
        if method in self._embedding_cache:
            return self._embedding_cache[method]
        
        try:
            if method == 'node2vec':
                if node2vec is None:
                    self.logger.warning("node2vec not installed. Falling back to basic embedding.")
                    return self._fallback_embedding(dimensions)
                
                # Node2Vec embedding
                graph = node2vec.Graph(self.G, is_directed=False, p=1, q=1)
                graph.preprocess_transition_probs()
                walks = graph.simulate_walks(num_walks=10, walk_length=80)
                
                # Train Word2Vec model
                model = gensim.models.Word2Vec(walks, vector_size=dimensions, window=10, 
                                               min_count=0, sg=1, workers=4, epochs=1)
                
                # Get node embeddings
                embeddings = {node: model.wv[str(node)] for node in self.G.nodes()}
            
            elif method == 'tsne':
                # Centrality-based embedding
                centrality_methods = {
                    'degree': dict(self.G.degree()),
                    'betweenness': nx.betweenness_centrality(self.G),
                    'closeness': nx.closeness_centrality(self.G),
                    'eigenvector': nx.eigenvector_centrality(self.G)
                }
                
                # Prepare feature matrix
                features = np.array([
                    [centrality_methods['degree'][node],
                     centrality_methods['betweenness'][node],
                     centrality_methods['closeness'][node],
                     centrality_methods['eigenvector'][node]]
                    for node in self.G.nodes()
                ])
                
                # Normalize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Apply t-SNE
                tsne = TSNE(n_components=dimensions, random_state=42)
                embeddings_array = tsne.fit_transform(features_scaled)
                
                # Convert to dictionary
                embeddings = {node: embeddings_array[i] 
                             for i, node in enumerate(self.G.nodes())}
            
            elif method == 'pca':
                # Similar to t-SNE, but using PCA
                centrality_methods = {
                    'degree': dict(self.G.degree()),
                    'betweenness': nx.betweenness_centrality(self.G),
                    'closeness': nx.closeness_centrality(self.G),
                    'eigenvector': nx.eigenvector_centrality(self.G)
                }
                
                # Prepare feature matrix
                features = np.array([
                    [centrality_methods['degree'][node],
                     centrality_methods['betweenness'][node],
                     centrality_methods['closeness'][node],
                     centrality_methods['eigenvector'][node]]
                    for node in self.G.nodes()
                ])
                
                # Normalize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Apply PCA
                pca = PCA(n_components=dimensions)
                embeddings_array = pca.fit_transform(features_scaled)
                
                # Convert to dictionary
                embeddings = {node: embeddings_array[i] 
                             for i, node in enumerate(self.G.nodes())}
            
            else:
                raise ValueError(f"Unsupported embedding method: {method}")
            
            # Prepare visualization data
            x = [emb[0] for emb in embeddings.values()]
            y = [emb[1] if dimensions > 1 else 0 for emb in embeddings.values()]
            
            # Color nodes by type if available
            try:
                node_types = [self.G.nodes[node].get('type', 'Unknown') 
                             for node in self.G.nodes()]
                unique_types = list(set(node_types))
                color_map = {t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                             for i, t in enumerate(unique_types)}
                node_colors = [color_map[t] for t in node_types]
            except:
                node_colors = 'blue'
            
            # Create Plotly scatter plot
            fig = go.Figure(data=go.Scatter(
                x=x, 
                y=y, 
                mode='markers',
                marker=dict(
                    size=10,
                    color=node_colors,
                    opacity=0.7
                ),
                text=[f"Node: {node}<br>Type: {self.G.nodes[node].get('type', 'Unknown')}" 
                      for node in self.G.nodes()]
            ))
            
            fig.update_layout(
                title=f"{method.upper()} Network Embedding",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2" if dimensions > 1 else "Constant",
                height=600,
                width=800
            )
            
            # Cache and return results
            result = {
                'embeddings': embeddings,
                'visualization': fig,
                'method': method
            }
            
            self._embedding_cache[method] = result
            return result
        
        except Exception as e:
            self.logger.error(f"Network embedding failed: {e}")
            return self._fallback_embedding(dimensions)
    
    def _fallback_embedding(self, dimensions=2):
        """
        Fallback embedding method using basic networkx metrics
        
        Args:
            dimensions (int): Number of embedding dimensions
        
        Returns:
            dict: Simple embedding results
        """
        # Use basic network metrics as embedding
        degree_centrality = dict(self.G.degree())
        betweenness = nx.betweenness_centrality(self.G)
        closeness = nx.closeness_centrality(self.G)
        
        # Prepare feature matrix
        features = np.array([
            [degree_centrality[node], 
             betweenness[node], 
             closeness[node]]
            for node in self.G.nodes()
        ])
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Reduce dimensions
        if dimensions == 2:
            pca = PCA(n_components=2)
            embeddings_array = pca.fit_transform(features_scaled)
        else:
            embeddings_array = features_scaled[:, :dimensions]
        
        # Convert to dictionary
        embeddings = {node: embeddings_array[i] 
                     for i, node in enumerate(self.G.nodes())}
        
        # Visualization
        x = [emb[0] for emb in embeddings.values()]
        y = [emb[1] if dimensions > 1 else 0 for emb in embeddings.values()]
        
        fig = go.Figure(data=go.Scatter(
            x=x, 
            y=y, 
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.7),
            text=[f"Node: {node}" for node in self.G.nodes()]
        ))
        
        fig.update_layout(
            title="Fallback Network Embedding",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2" if dimensions > 1 else "Constant",
            height=600,
            width=800
        )
        
        return {
            'embeddings': embeddings,
            'visualization': fig,
            'method': 'fallback'
        }
    
    def advanced_centrality_analysis(self):
        """
        Comprehensive centrality analysis with Plotly visualizations
        
        Returns:
            dict: Advanced centrality metrics and visualizations
        """
        # Check cache
        if self._centrality_cache:
            return self._centrality_cache
        
        # Compute various centrality measures
        centrality_methods = {
            'Degree': dict(self.G.degree()),
            'Betweenness': nx.betweenness_centrality(self.G),
            'Closeness': nx.closeness_centrality(self.G),
            'Eigenvector': nx.eigenvector_centrality(self.G),
            'PageRank': nx.pagerank(self.G)
        }
        
        # Prepare results dictionary
        results = {}
        
        # Boxplot of centrality distributions
        centrality_df = pd.DataFrame(centrality_methods)
        
        # Create boxplot
        fig_boxplot = go.Figure()
        for column in centrality_df.columns:
            fig_boxplot.add_trace(go.Box(y=centrality_df[column], name=column))
        
        fig_boxplot.update_layout(
            title='Centrality Measures Distribution',
            yaxis_title='Centrality Value',
            height=600,
            width=800
        )
        results['centrality_boxplot'] = fig_boxplot
        
        # Scatter matrix of centrality measures
        fig_scatter_matrix = px.scatter_matrix(
            centrality_df, 
            dimensions=list(centrality_methods.keys()),
            title='Centrality Measures Correlation'
        )
        fig_scatter_matrix.update_layout(height=1000, width=1000)
        results['centrality_scatter_matrix'] = fig_scatter_matrix
        
        # Top nodes by each centrality measure
        top_nodes = {}
        for method, centrality in centrality_methods.items():
            top_nodes[method] = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        results['top_nodes'] = top_nodes
        
        # Correlation heatmap
        corr_matrix = centrality_df.corr()
        fig_correlation = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r'
        ))
        fig_correlation.update_layout(
            title='Centrality Measures Correlation',
            height=600,
            width=800
        )
        results['correlation_heatmap'] = fig_correlation
        
        # Cache results
        self._centrality_cache = results
        return results
    
    def network_resilience_analysis(self):
        """
        Comprehensive network resilience analysis
        
        Returns:
            dict: Network resilience metrics and visualizations
        """
        # Attack simulation methods
        attack_methods = {
            'Random Removal': self._random_node_removal,
            'Targeted Removal (Degree)': self._targeted_node_removal_degree,
            'Targeted Removal (Betweenness)': self._targeted_node_removal_betweenness
        }
        
        # Results storage
        results = {}
        
        # Simulation of network attacks
        for method_name, attack_method in attack_methods.items():
            # Perform attack simulation
            attack_results = attack_method()
            results[method_name] = attack_results
        
        # Create combined visualization
        fig = go.Figure()
        
        for method_name, attack_data in results.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(attack_data['largest_component_ratio']))),
                y=attack_data['largest_component_ratio'],
                mode='lines+markers',
                name=method_name
            ))
        
        fig.update_layout(
            title='Network Resilience Under Different Attack Strategies',
            xaxis_title='Nodes Removed',
            yaxis_title='Largest Component Ratio',
            height=600,
            width=800
        )
        
        results['visualization'] = fig
        
        return results
    
    def _random_node_removal(self, iterations=100):
        """
        Simulate random node removal
        
        Args:
            iterations (int): Number of nodes to remove
        
        Returns:
            dict: Random node removal results
        """
        G_copy = self.G.copy()
        original_size = len(G_copy)
        largest_component_ratio = [1.0]  # Start with full network
        
        nodes = list(G_copy.nodes())
        np.random.shuffle(nodes)
        
        for i in range(min(iterations, len(nodes))):
            G_copy.remove_node(nodes[i])
            largest_component = max(nx.connected_components(G_copy), key=len)
            largest_component_size = len(largest_component)
            largest_component_ratio.append(largest_component_size / original_size)
        
        return {
            'largest_component_ratio': largest_component_ratio
        }
    
    def _targeted_node_removal_degree(self, iterations=100):
        """
        Simulate targeted node removal based on degree centrality
        
        Args:
            iterations (int): Number of nodes to remove
        
        Returns:
            dict: Targeted node removal results
        """
        G_copy = self.G.copy()
        original_size = len(G_copy)
        largest_component_ratio = [1.0]  # Start with full network
        
        # Sort nodes by degree centrality
        nodes_by_degree = sorted(G_copy.degree(), key=lambda x: x[1], reverse=True)
        
        for i in range(min(iterations, len(nodes_by_degree))):
            node = nodes_by_degree[i][0]
            G_copy.remove_node(node)
            
            largest_component = max(nx.connected_components(G_copy), key=len)
            largest_component_size = len(largest_component)
            largest_component_ratio.append(largest_component_size / original_size)
        
        return {
            'largest_component_ratio': largest_component_ratio
        }
    
    def _targeted_node_removal_betweenness(self, iterations=100):
        """
        Simulate targeted node removal based on betweenness centrality
        
        Args:
            iterations (int): Number of nodes to remove
        
        Returns:
            dict: Targeted node removal results
        """
        G_copy = self.G.copy()
        original_size = len(G_copy)
        largest_component_ratio = [1.0]  # Start with full network
        
        # Compute betweenness centrality
        betweenness = nx.betweenness_centrality(G_copy)
        
        # Sort nodes by betweenness centrality
        nodes_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(min(iterations, len(nodes_by_betweenness))):
            node = nodes_by_betweenness[i][0]
            G_copy.remove_node(node)
            
            largest_component = max(nx.connected_components(G_copy), key=len)
            largest_component_size = len(largest_component)
            largest_component_ratio.append(largest_component_size / original_size)
        
        return {
            'largest_component_ratio': largest_component_ratio
        }
    
    def community_structure_analysis(self):
        """
        Advanced community structure analysis with Plotly visualizations
        
        Returns:
            dict: Community structure metrics and visualizations
        """
        # Multiple community detection algorithms
        community_methods = {
            'Louvain': nx.community.louvain_communities,
            'Greedy Modularity': nx.community.greedy_modularity_communities,
            'Label Propagation': nx.community.label_propagation_communities
        }
        
        # Results storage
        results = {
            'communities': {},
            'modularity_scores': {},
            'visualizations': {}
        }
        
        # Analyze communities
        for method_name, community_func in community_methods.items():
            try:
                # Detect communities
                communities = community_func(self.G)
                
                # Compute modularity
                modularity = nx.community.modularity(self.G, communities)
                
                # Store results
                results['communities'][method_name] = communities
                results['modularity_scores'][method_name] = modularity
                
                # Prepare community size visualization
                community_sizes = [len(community) for community in communities]
                
                # Plotly histogram of community sizes
                fig_community_sizes = go.Figure(data=[go.Histogram(
                    x=community_sizes,
                    nbinsx=20,
                    name=method_name
                )])
                fig_community_sizes.update_layout(
                    title=f'Community Size Distribution - {method_name}',
                    xaxis_title='Community Size',
                    yaxis_title='Frequency',
                    height=400,
                    width=600
                )
                
                results['visualizations'][method_name] = {
                    'community_sizes': fig_community_sizes
                }
                
                # Community composition analysis
                if len(communities) > 0:
                    # Analyze largest community
                    largest_community = max(communities, key=len)
                    
                    # Node type distribution in the largest community
                    try:
                        type_counts = Counter(
                            self.G.nodes[node].get('type', 'Unknown') 
                            for node in largest_community
                        )
                        
                        # Pie chart of node type distribution
                        fig_type_dist = go.Figure(data=[go.Pie(
                            labels=list(type_counts.keys()),
                            values=list(type_counts.values()),
                            name='Node Type Distribution'
                        )])
                        fig_type_dist.update_layout(
                            title=f'Node Type Distribution in Largest Community - {method_name}',
                            height=400,
                            width=600
                        )
                        
                        results['visualizations'][method_name]['type_distribution'] = fig_type_dist
                    except Exception as e:
                        self.logger.warning(f"Could not generate type distribution: {e}")
            
            except Exception as e:
                self.logger.error(f"Community detection failed for {method_name}: {e}")
        
        return results
    
    def network_entropy_analysis(self):
        """
        Compute network entropy and complexity metrics
        """
        # Degree distribution entropy
        degrees = [d for n, d in self.G.degree()]
        degree_dist = np.array(degrees) / sum(degrees)
        degree_entropy = -np.sum(degree_dist * np.log2(degree_dist + 1e-10))
    
        # Clustering coefficient analysis
        clustering_coeffs = np.array(list(nx.clustering(self.G).values()))
        if len(clustering_coeffs) > 0:
            clustering_entropy = -np.sum(
                clustering_coeffs * np.log2(clustering_coeffs + 1e-10)
            )
        else:
            clustering_entropy = 0
        
        # Centrality entropy
        centrality_methods = {
            'Degree': nx.degree_centrality(self.G),
            'Betweenness': nx.betweenness_centrality(self.G),
            'Closeness': nx.closeness_centrality(self.G),
            'Eigenvector': nx.eigenvector_centrality(self.G)
        }
        
        centrality_entropies = {}
        for method, centrality in centrality_methods.items():
            values = np.array(list(centrality.values()))
            dist = values / np.sum(values)
            centrality_entropies[method] = -np.sum(dist * np.log2(dist + 1e-10))
        
        # Visualization of entropy distributions
        fig_entropy_dist = go.Figure()
        
        # Degree distribution entropy
        fig_entropy_dist.add_trace(go.Histogram(
            x=degrees,
            name='Degree Distribution',
            opacity=0.75
        ))
        
        # Clustering coefficient distribution
        fig_entropy_dist.add_trace(go.Histogram(
            x=clustering_coeffs,
            name='Clustering Coefficients',
            opacity=0.75
        ))
        
        fig_entropy_dist.update_layout(
            title='Network Entropy Distributions',
            xaxis_title='Value',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400,
            width=600
        )
        
        # Centrality entropy comparison
        fig_centrality_entropy = go.Figure(data=[
            go.Bar(
                x=list(centrality_entropies.keys()),
                y=list(centrality_entropies.values())
            )
        ])
        fig_centrality_entropy.update_layout(
            title='Centrality Entropy Comparison',
            xaxis_title='Centrality Method',
            yaxis_title='Entropy',
            height=400,
            width=600
        )
        
        return {
            'degree_entropy': degree_entropy,
            'clustering_entropy': clustering_entropy,
            'centrality_entropies': centrality_entropies,
            'visualizations': {
                'entropy_distributions': fig_entropy_dist,
                'centrality_entropy': fig_centrality_entropy
            }
        }
    
    def link_prediction_analysis(self):
        """
        Perform link prediction analysis
        
        Returns:
            dict: Link prediction metrics and visualizations
        """
        # Various link prediction methods
        prediction_methods = {
            'Common Neighbors': nx.resource_allocation_index,
            'Jaccard Coefficient': nx.jaccard_coefficient,
            'Adamic-Adar': nx.adamic_adar_index,
            'Preferential Attachment': nx.preferential_attachment
        }
        
        # Results storage
        results = {
            'predictions': {},
            'visualizations': {}
        }
        
        # Compute link predictions
        for method_name, prediction_func in prediction_methods.items():
            try:
                # Get all possible node pairs
                node_pairs = list(nx.non_edges(self.G))
                
                # Compute prediction scores
                predictions = list(prediction_func(self.G, node_pairs))
                
                # Sort predictions by score
                predictions.sort(key=lambda x: x[2], reverse=True)
                
                # Store top predictions
                results['predictions'][method_name] = predictions[:100]
                
                # Visualization of prediction scores
                scores = [pred[2] for pred in predictions]
                
                # Histogram of prediction scores
                fig_prediction_scores = go.Figure(data=[go.Histogram(
                    x=scores,
                    nbinsx=50,
                    name=method_name
                )])
                fig_prediction_scores.update_layout(
                    title=f'Link Prediction Scores - {method_name}',
                    xaxis_title='Prediction Score',
                    yaxis_title='Frequency',
                    height=400,
                    width=600
                )
                
                results['visualizations'][method_name] = fig_prediction_scores
            
            except Exception as e:
                self.logger.error(f"Link prediction failed for {method_name}: {e}")
        
        return results
