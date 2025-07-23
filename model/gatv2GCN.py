import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool as gap, global_max_pool as gmp, GCNConv

class GraphConvolutionalBlock(nn.Module):
    """
    This block processes the drug's molecular graph.
    It uses a GATv2 layer followed by a GCN layer and a final linear layer
    to ensure the output dimension is 128.
    """
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphConvolutionalBlock, self).__init__()
        self.gatv2_layer1 = GATv2Conv(input_dim, output_dim, heads=10, dropout=dropout_rate)
        self.gatv2_layer2 = GCNConv(output_dim * 10, output_dim * 10)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # Add a final linear layer to control the output dimension.
        # The output from gmp+gap is (output_dim * 10 * 2).
        self.fc_out = nn.Linear(output_dim * 10 * 2, 128)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gatv2_layer1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gatv2_layer2(x, edge_index)
        x = self.activation(x)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # Pass through the final linear layer to get a fixed-size output of 128.
        x = self.fc_out(x)
        return x

class ProteinSequenceBlock(nn.Module):
    """
    This block processes the protein's amino acid sequence.
    It uses a BiLSTM followed by a 1D Convolutional layer.
    """
    def __init__(self, xt_features, embed_dim, n_filters, output_dim):
        super(ProteinSequenceBlock, self).__init__()
        self.embedding = nn.Embedding(xt_features + 1, embed_dim)
        
        # The BiLSTM outputs 128 features (64 forward + 64 backward).
        lstm_output_features = 64 * 2
        
        # The convolution's in_channels must match the LSTM's output features.
        self.conv_protein = nn.Conv1d(in_channels=lstm_output_features, out_channels=n_filters, kernel_size=8)
        
        # Using batch_first=True makes handling dimensions much easier.
        self.bilstm = nn.LSTM(embed_dim, 64, 1, dropout=0.2, bidirectional=True, batch_first=True)
        
        # The input size of the fully connected layer must be calculated correctly.
        # After convolution, the length will be (1000 - kernel_size + 1) = 993.
        # So the flattened size is n_filters * 993.
        self.fc_protein = nn.Linear(n_filters * (1000 - 8 + 1), output_dim)

    def forward(self, target):
        embedded_xt = self.embedding(target)
        lstm_out, _ = self.bilstm(embedded_xt)
        
        # Permute the dimensions for the Conv1d layer: (batch, features, seq_len)
        conv_input = lstm_out.permute(0, 2, 1)
        
        conv_xt = self.conv_protein(conv_input)
        
        # Flatten the output of the convolution to a single vector per sample.
        xt = conv_xt.view(conv_xt.size(0), -1)
        
        return self.fc_protein(xt)

class CombinedDenseBlock(nn.Module):
    """
    This block takes the combined drug and protein features and makes the final prediction.
    """
    def __init__(self, input_dim, output_size, dropout_rate):
        super(CombinedDenseBlock, self).__init__()
        self.fc_combined1 = nn.Linear(input_dim, 1024)
        self.fc_combined2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc_combined1(x)))
        x = self.dropout(self.activation(self.fc_combined2(x)))
        return self.output_layer(x)

class GATv2GCNModel(nn.Module):
    """
    The main model class that assembles all the blocks.
    """
    def __init__(self, output_size=1, xd_features=78, xt_features=25,
                 filter_count=32, embedding_dim=128, dense_output_dim=128, dropout_rate=0.2):
        super(GATv2GCNModel, self).__init__()
        
        # Instantiate the graph block. dense_output_dim (128) is passed as the 'output_dim'
        # for the GATv2 and GCN layers before the final linear layer.
        self.graph_block = GraphConvolutionalBlock(xd_features, dense_output_dim, dropout_rate)
        
        # Instantiate the protein block. It's set to output 128 features.
        self.protein_block = ProteinSequenceBlock(xt_features, embedding_dim, filter_count, 128)
        
        # The combined block correctly takes 128 (from graph) + 128 (from protein) = 256 as input.
        self.combined_block = CombinedDenseBlock(256, output_size, dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        graph_features = self.graph_block(x, edge_index, batch)
        protein_features = self.protein_block(target)

        combined_features = torch.cat((graph_features, protein_features), 1)
        return self.combined_block(combined_features)
