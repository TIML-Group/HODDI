# Import libraries
import utils  
import numpy as np  
from tqdm import tqdm  
import torch
import torch.nn as nn
import torch_geometric.nn as hnn  
import torch.nn.functional as F  
from torch_geometric.nn import GCNConv, GATConv, global_max_pool, global_mean_pool  

class GCNNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_GMP=True):
        """
        Initialize a GCNNet module.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_GMP (bool): Whether to use global max pooling. Default is True.
        """

        super(GCNNet, self).__init__()
        self.use_GMP = use_GMP  
        self.conv1 = GCNConv(in_channels, 128)  
        self.batch_conv1 = nn.BatchNorm1d(128)  
        self.conv2 = GCNConv(128, out_channels)  
        self.batch_conv2 = nn.BatchNorm1d(out_channels)  
        self.reset_para()  
        self.act = nn.ReLU()  
        
    def reset_para(self):

        """
        Reset all parameters of the module using Xavier uniform distribution
        """
        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.Linear)):  
                nn.init.xavier_uniform_(m.weight)  
                if m.bias is not None:
                    nn.init.zeros_(m.bias) 
        return

    def forward(self, input_feature, input_adj, ibatch):

        """
        Define the forward propagation of GCNNet.

        Parameters:
            input_feature (Tensor): Input node features.
            input_adj (Tensor): Input adjacency matrix.
            ibatch (Tensor): Input batch index.

        Returns:
            Tensor: Output node features after GCN convolution and pooling.
        """
        x_feature = self.conv1(input_feature, input_adj)  
        x_feature = self.batch_conv1(self.act(x_feature))  
        x_feature = self.conv2(x_feature, input_adj)  
        x_feature = self.act(x_feature)  
        x_feature = self.batch_conv2(x_feature) 
        if self.use_GMP:  
            x_feature = global_max_pool(x_feature, ibatch)  
        else:
            x_feature = global_mean_pool(x_feature, ibatch)  
        return x_feature  

class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, heads=6, use_GMP=True):

        """
        Initialize a GATNet module.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            heads (int): Number of attention heads. Default is 6.
            use_GMP (bool): Whether to use global max pooling. Default is True.
        """
        
        super(GATNet, self).__init__()
        self.use_GMP = use_GMP  
        self.conv1 = GATConv(in_channels, 128, heads=heads)  
        self.batch_conv1 = nn.BatchNorm1d(128 * heads)  
        self.conv2 = GATConv(128 * heads, out_channels, heads=heads)  
        self.batch_conv2 = nn.BatchNorm1d(out_channels * heads) 
        self.reset_para()  
        self.act = nn.ReLU()  
        self.gcn_linear = nn.Linear(out_channels * heads, out_channels)  

    def reset_para(self):
        """
        Reset the parameters of convolutional and linear layers in the module.
        """

        for m in self.modules():  
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, input_feature, input_adj, ibatch):

        """
        Define the forward propagation of GATNet.

        Parameters:
            input_feature (Tensor): Input node features.
            input_adj (Tensor): Input adjacency matrix.
            ibatch (Tensor): Input batch index.

        Returns:
            Tensor: Output node features after GAT convolution and pooling.
        """
        
        x_feature = self.conv1(input_feature, input_adj) 
        x_feature = self.batch_conv1(self.act(x_feature))  
        x_feature = self.conv2(x_feature, input_adj)  
        x_feature = self.act(x_feature)  
        x_feature = self.batch_conv2(x_feature)  
        x_feature = self.gcn_linear(x_feature)  
        if self.use_GMP:  
            x_feature = global_max_pool(x_feature, ibatch)  
        else:
            x_feature = global_mean_pool(x_feature, ibatch)  
        return x_feature  


def run_gcn_nodes(model):
    
    """
    Run the forward propagation of GATNet and return the output node features.
    
    Parameters:
        model (nn.Module): An instance of GATNet.
    
    Returns:
        Tensor: Output node features after GAT convolution and pooling.
    """
    return model.forward()


def generate_hyperedge_features(node_features, incidence_matrix, aggregation="mean"):

    """
    Generate hyperedge features from node features and incidence matrix using the specified aggregation method.

    Args:
        node_features (Tensor): Node features, shape (num_nodes, node_dim)
        incidence_matrix (Tensor): Incidence matrix, shape (num_nodes, num_edges)
        aggregation (str, optional): Aggregation method, can be "mean" or "sum". Default is "mean".

    Returns:
        Tensor: Hyperedge features, shape (num_edges, node_dim)
    """
    
    num_nodes, node_dim = node_features.shape  
    num_edges = incidence_matrix.shape[1]  

    # Sparse matrix processing
    if incidence_matrix.is_sparse:  
        incidence_matrix = incidence_matrix.coalesce()  
        if aggregation == "mean":
            degree = torch.sparse.sum(incidence_matrix, dim=0).to_dense()  
            degree[degree == 0] = 1  
        weighted_node_features = torch.sparse.mm(incidence_matrix.T, node_features)  
        if aggregation == "mean":
            hyperedge_features = weighted_node_features / degree.unsqueeze(1)  
        else:
            hyperedge_features = weighted_node_features  
    else:
        weighted_node_features = torch.mm(incidence_matrix.T, node_features)  
        degree = torch.sum(incidence_matrix, dim=0, keepdim=True)  
        degree[degree == 0] = 1  
        if aggregation == "mean":
            hyperedge_features = weighted_node_features / degree.T 
        else:
            hyperedge_features = weighted_node_features  

    return hyperedge_features  

class NodeToEdgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):

        """
        Initialize the NodeToEdgeAttention module.

        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output dimension.
        """

        super(NodeToEdgeAttention, self).__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim) 
        self.attention_weight = nn.Linear(hidden_dim, 1) 
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  
        self.batch_norm = nn.BatchNorm1d(output_dim)  
        self.expand_output = nn.Linear(hidden_dim, output_dim) 

    def forward(self, node_features, incidence_matrix):

        """
        Forward pass for the NodeToEdgeAttention module.

        Args:
            node_features (Tensor): Input node features with shape (n, node_dim).
            incidence_matrix (Tensor): Incidence matrix with shape (n, num_edges).

        Returns:
            Tensor: Updated edge features with shape (num_edges, output_dim).
        """

        # Get the number of hyperedges
        num_edges = incidence_matrix.shape[1]  

        # Calculate the projected node features
        transformed_node_features = self.node_transform(node_features)  # (n, hidden_dim)

        # Get the hyperedge features
        masked_node_features = torch.mm(incidence_matrix.T, transformed_node_features)  # (num_edges, hidden_dim)

        # Calculate the attention scores
        attention_scores = self.leaky_relu(self.attention_weight(masked_node_features))  # (num_edges, 1)

        # Normalize the attention scores
        attention_coeffs = torch.nn.functional.softmax(attention_scores.to_dense(), dim=0)
        weighted_edge_features = attention_coeffs * masked_node_features  # (num_edges, hidden_dim)

        # Get final hyperedge features
        updated_edge_features = self.expand_output(weighted_edge_features)  # (num_edges, output_dim)

        # Normalize the edge features
        updated_edge_features = utils.min_max_normalize_cuda(updated_edge_features)

        return updated_edge_features  

class EdgeToNodeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):

        """
        Initialize the EdgeToNodeAttention module.

        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output dimension.
        """
        super(EdgeToNodeAttention, self).__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)  
        self.edge_transform = nn.Linear(edge_dim, hidden_dim) 
        self.attention_weight = nn.Linear(hidden_dim, 1)  
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  
        self.expand_output = nn.Linear(hidden_dim, output_dim)  
        self.batch_norm = nn.BatchNorm1d(output_dim)  

    def forward(self, node_features, incidence_matrix, edge_features):

        """
        Forward pass for the EdgeToNodeAttention module.

        Args:
            node_features (Tensor): Input node features with shape (n, node_dim).
            incidence_matrix (Tensor): Incidence matrix with shape (n, num_edges).
            edge_features (Tensor): Input edge features with shape (num_edges, edge_dim).

        Returns:
            Tensor: Updated node features with shape (n, output_dim).
        """
        # Get the number of nodes and edges
        num_nodes, node_dim = node_features.shape  
        num_edges, edge_dim = edge_features.shape  

        # Calculate the projected node features
        transformed_node_features = self.node_transform(node_features)  # (num_nodes, hidden_dim)
        transformed_edge_features = self.edge_transform(edge_features)  # (num_edges, hidden_dim)

        # Calculate the aggregated edge features
        aggregated_edge_features = torch.mm(incidence_matrix, transformed_edge_features) / (
            incidence_matrix.sum(dim=1, keepdim=True) + 1e-8
        )  

        # Calculate the attention scores
        attention_input = transformed_node_features + aggregated_edge_features  # (num_nodes, hidden_dim)
        attention_scores = self.leaky_relu(self.attention_weight(attention_input))  # (num_nodes, 1)

        # Normlize the attention scores
        attention_coeffs = torch.sigmoid(attention_scores)  
        weighted_edge_features = attention_coeffs * aggregated_edge_features  

        # Calculate the updated node features
        updated_node_features = weighted_edge_features + transformed_node_features  
        updated_node_features = self.expand_output(updated_node_features)  

        # Normalize the node features
        updated_node_features = utils.min_max_normalize_cuda(updated_node_features)

        return updated_node_features


class MultiHeadNodeToEdgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.3):

        """
        Initialize the MultiHeadNodeToEdgeAttention module.

        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.3.
        """
        super(MultiHeadNodeToEdgeAttention, self).__init__()
        self.num_heads = num_heads  
        self.attentions = nn.ModuleList([NodeToEdgeAttention(node_dim, edge_dim, hidden_dim, edge_dim) for _ in range(num_heads)]) 
        self.relu = nn.ReLU()  
        self.output_transform = nn.Linear(num_heads * edge_dim, output_dim)  
        self.dropout = nn.Dropout(dropout_rate) 
    
    def forward(self, node_features, incidence_matrix):

        """
        Forward pass for the MultiHeadNodeToEdgeAttention module.

        Args:
            node_features (Tensor): Input node features with shape (n, node_dim).
            incidence_matrix (Tensor): Incidence matrix with shape (n, num_edges).

        Returns:
            Tensor: Updated node features with shape (n, output_dim).
        """
        multi_head_outputs = [attention(node_features, incidence_matrix) for attention in self.attentions]  
        concatenated_output = torch.cat(multi_head_outputs, dim=-1)  
        concatenated_output = self.relu(concatenated_output)  
        concatenated_output = self.dropout(concatenated_output)  
        return self.output_transform(concatenated_output)  

class MultiHeadEdgeToNodeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=4, output_dim=128, dropout_rate=0.3):

        """
        Initialize the MultiHeadEdgeToNodeAttention module.

        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            hidden_dim (int): Hidden layer dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            output_dim (int, optional): Output dimension. Defaults to 128.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.3.
        """
        super(MultiHeadEdgeToNodeAttention, self).__init__()
        self.num_heads = num_heads 
        self.hidden_dim = hidden_dim  
        self.attention_heads = nn.ModuleList([
            EdgeToNodeAttention(node_dim, edge_dim, hidden_dim, output_dim) for _ in range(num_heads)
        ]) 
        self.relu = nn.ReLU()  
        self.output_transform = nn.Linear(num_heads * output_dim, output_dim)  
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, node_features, incidence_matrix, edge_features):

        """
        Forward pass for the MultiHeadEdgeToNodeAttention module.

        Args:
            node_features (Tensor): Input node features with shape (n, node_dim).
            incidence_matrix (Tensor): Incidence matrix with shape (n, num_edges).
            edge_features (Tensor): Input edge features with shape (num_edges, edge_dim).

        Returns:
            Tensor: Updated node features with shape (n, output_dim).
        """
        multi_head_outputs = node_features  
        for head in self.attention_heads: 
            multi_head_outputs = head(multi_head_outputs, incidence_matrix, edge_features) 
            multi_head_outputs = self.relu(multi_head_outputs)  
        
        output = self.dropout(multi_head_outputs)  
        return output  

class CLOSEgaps(nn.Module):
    def __init__(self, algorithm, input_num, input_feature_num, emb_dim, conv_dim, head=3, p=0.1, L=1,
                 use_attention=True, extra_feature=None, reaction_feature=None, enable_hygnn=False, incidence_matrix_pos=None):

        """
        Initialize the CLOSEgaps module.

        Args:
            algorithm (str): The algorithm type to be used.
            input_num (int): Number of input features.
            input_feature_num (int): Dimension of the input feature vector.
            emb_dim (int): Embedding dimension.
            conv_dim (int): Convolution dimension.
            head (int, optional): Number of attention heads. Defaults to 3.
            p (float, optional): Dropout probability. Defaults to 0.1.
            L (int, optional): Number of hypergraph convolution layers. Defaults to 1.
            use_attention (bool, optional): Whether to use attention mechanism. Defaults to True.
            extra_feature (Tensor, optional): Additional node features. Defaults to None.
            reaction_feature (Tensor, optional): Reaction features for linear transformation. Defaults to None.
            enable_hygnn (bool, optional): Whether to enable HyGNN. Defaults to False.
            incidence_matrix_pos (Tensor, optional): Incidence matrix position. Defaults to None.
        """

        super(CLOSEgaps, self).__init__()
        self.algorithm = algorithm  
        self.emb_dim = emb_dim  
        self.conv_dim = conv_dim 
        self.p = p  
        self.input_num = input_num 
        self.head = head  
        self.hyper_conv_L = L  
        self.linear_encoder = nn.Linear(input_feature_num, emb_dim) 
        self.similarity_liner = nn.Linear(input_num, emb_dim)  
        self.max_pool = hnn.global_max_pool  
        self.extra_feature = extra_feature
        self.incidence_matrix_pos = incidence_matrix_pos

        self.in_channel = emb_dim
        if extra_feature is not None:
            self.extra_feature = extra_feature  
            self.in_channel = emb_dim
            self.in_channel = 2 * emb_dim

            self.reaction_feature = reaction_feature 
            if reaction_feature is not None:    
                self.linear_reaction_feature = nn.Linear(self.reaction_feature.shape[1], emb_dim)  

            self.enable_hygnn = enable_hygnn  

            if self.algorithm in ["mol2vec", "smiles2vec"]:
                self.pre_linear = nn.Linear(extra_feature.shape[1], emb_dim) 
                
            if self.algorithm == "gcn_ind":
                self.gcn = GCNNet(extra_feature.dataset.num_features, emb_dim)  

            if self.algorithm == "gcn_ind_attr":
                self.gcn = GATNet(extra_feature.dataset.num_features, emb_dim) 

            # if self.algorithm == "deep_gcn":
                # self.gcn = DeepGCNNet(extra_feature.dataset.num_features, emb_dim)  

            # if self.algorithm == "gae":
                # self.gcn = GAE_GCNNet(extra_feature.dataset.num_features, emb_dim)  

            if self.enable_hygnn:  
                self.linear_node_to_edge = nn.Linear(emb_dim, self.in_channel)  
                self.node_to_edge_attr = MultiHeadNodeToEdgeAttention(edge_dim=emb_dim, node_dim=emb_dim,
                                                                    hidden_dim=emb_dim * 2, output_dim=self.in_channel, num_heads=2)
                self.edge_to_node_attr = MultiHeadEdgeToNodeAttention(node_dim=emb_dim, edge_dim=emb_dim * 2,
                                                                    hidden_dim=emb_dim * 2, output_dim=emb_dim, num_heads=2)

        self.relu = nn.ReLU()  
        self.hypergraph_conv = hnn.HypergraphConv(self.in_channel, conv_dim, heads=head, use_attention=use_attention, dropout=p)  
        if L > 1:
            self.hypergraph_conv_list = nn.ModuleList()
            for l in range(L - 1):
                self.hypergraph_conv_list.append(
                    hnn.HypergraphConv(head * conv_dim, conv_dim, heads=head, use_attention=use_attention, dropout=p))

        if use_attention:
            self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
            if L > 1:
                self.hyperedge_attr_list = nn.ModuleList()
                for l in range(L - 1):
                    self.hyperedge_attr_list.append(nn.Linear(input_num, head * conv_dim))
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features, incidence_matrix):
        incidence_matrix_T = incidence_matrix.T  
        input_nodes_features = self.relu(self.linear_encoder(input_features))
        row, col = torch.where(incidence_matrix_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix_T)

        
        if self.extra_feature is not None:
            if self.algorithm == "similarity":
                extra_feature = self.similarity_liner(self.extra_feature)

            if self.algorithm in ["fp", "xor"]:
                extra_feature = self.extra_feature

            if self.algorithm in ["mol2vec", "smiles2vec"]:
                extra_feature = self.pre_linear(self.extra_feature)

            if self.algorithm in ["gcn_ind", "gcn_ind_attr", "deep_gcn", "gae"]:
                extra_feature = []
                for feature in self.extra_feature:
                    gcn_result = self.gcn(feature.x, feature.edge_index, feature.batch)
                    extra_feature.append(gcn_result)
                # extra_feature = torch.tensor(np.vstack([x.detach().cpu().numpy() for x in extra_feature]), dtype=torch.float32, device=device)
            
                extra_feature = torch.empty((len(self.extra_feature), self.emb_dim), dtype=torch.float32, device='cuda')
                for i, feature in enumerate(self.extra_feature):
                    gcn_result = self.gcn(feature.x, feature.edge_index, feature.batch)
                    extra_feature[i] = gcn_result

            if self.enable_hygnn and self.incidence_matrix_pos is not None:
                if self.reaction_feature is None:
                    updated_node_features = self.edge_to_node_attr(extra_feature, self.incidence_matrix_pos, updated_edge_features)
                    extra_feature = extra_feature + updated_node_features
                    hyperedge_attr = updated_edge_features
                    pass
                else:
                    edge_feature = self.linear_reaction_feature(self.reaction_feature)
                    edge_feature = self.linear_node_to_edge(edge_feature)
                    updated_node_features = self.edge_to_node_attr(extra_feature, self.incidence_matrix_pos, edge_feature)
                    updated_edge_features = self.node_to_edge_attr(updated_node_features, self.incidence_matrix_pos)
                    updated_node_features = self.edge_to_node_attr(extra_feature, self.incidence_matrix_pos, updated_edge_features)
                    extra_feature = extra_feature + updated_node_features
                    hyperedge_attr = updated_edge_features
            
            extra_feature = self.relu(extra_feature)
            input_nodes_features = torch.cat((extra_feature, input_nodes_features), dim=1)

        input_nodes_features = self.hypergraph_conv(input_nodes_features, edges, hyperedge_attr=hyperedge_attr)
        if self.hyper_conv_L > 1:
            for l in range(self.hyper_conv_L - 1):
                layer_hyperedge_attr = self.hyperedge_attr_list[l](incidence_matrix_T)
                input_nodes_features = self.hypergraph_conv_list[l](input_nodes_features, edges,
                                                                    hyperedge_attr=layer_hyperedge_attr)
                input_nodes_features = self.relu(input_nodes_features)

        hyperedge_feature = torch.mm(incidence_matrix_T, input_nodes_features)
        return self.hyperedge_linear(hyperedge_feature)

    def predict(self, input_fetures, incidence_matrix):
        """
        Predict the probability distribution of the output given input features and incidence matrix.

        Args:
            input_fetures (Tensor): Input features, shape (num_nodes, input_dim)
            incidence_matrix (Tensor): Incidence matrix, shape (num_edges, num_nodes)

        Returns:
            Tensor: Predicted probability distribution, shape (num_edges, num_classes)
        """
        return self.softmax(self.forward(input_fetures, incidence_matrix))
