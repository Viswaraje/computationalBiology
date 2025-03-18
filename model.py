import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool

from torch.nn import Parameter
from my_utiils import *
from torch_geometric.nn import GATConv  # Import GATConv
from base_model.SGATConv import SGATConv  # Import SGATConv
from my_utiils import *
EPS = 1e-15

class NodeRepresentation(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], use_relu=True, use_bn=True,
                 use_GMP=True, use_mutation=True, use_gexpr=True, use_methylation=True):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)
        # -------drug layers
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_methylation = use_methylation
        
        # ---- Use SGATConv instead of SGConv for drug features
        self.conv1 = SGATConv(gcn_layer, units_list[0])
        
        # Fix 1: Register batch norm layers properly
        self.batch_conv1 = nn.BatchNorm1d(units_list[0])
        
        # Fix 2: Use ModuleList for graph_conv and graph_bn
        self.graph_conv = nn.ModuleList()
        self.graph_bn = nn.ModuleList()
        for i in range(len(units_list) - 1):
            self.graph_conv.append(SGATConv(units_list[i], units_list[i + 1]))
            self.graph_bn.append(nn.BatchNorm1d(units_list[i + 1]))
        
        self.conv_end = SGATConv(units_list[-1], output)
        self.batch_end = nn.BatchNorm1d(output)
        
        # --------cell line layers (three omics)
        # -------gexp_layer
        self.fc_gexp1 = nn.Linear(dim_gexp, 256)
        self.batch_gexp1 = nn.BatchNorm1d(256)
        self.fc_gexp2 = nn.Linear(256, output)
        # -------methy_layer
        self.fc_methy1 = nn.Linear(dim_methy, 256)
        self.batch_methy1 = nn.BatchNorm1d(256)
        self.fc_methy2 = nn.Linear(256, output)
        # -------mut_layer
        self.cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.fla_mut = nn.Flatten()
        self.fc_mut = nn.Linear(2010, output)
        # ------Concatenate_three omics
        self.fcat = nn.Linear(300, output)
        self.batchc = nn.BatchNorm1d(100)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data):
        # -----drug representation
        # Debug prints for input shapes
        print(f"Drug feature shape: {drug_feature.shape}")
        print(f"Drug adj shape: {drug_adj.shape}")
        print(f"ibatch shape: {ibatch.shape}")
        
        x_drug = self.conv1(drug_feature, drug_adj)
        print(f"x_drug shape after conv1: {x_drug.shape}")
        
        x_drug = F.relu(x_drug)
        
        # Fix 3: Apply batch norm correctly
        if self.use_bn:
            # Determine if we're dealing with a batch or single sample
            if len(x_drug.shape) == 2:
                # Already in [num_nodes, features] format, apply batch norm
                x_drug = self.batch_conv1(x_drug)
            else:
                # Flatten and then apply batch norm
                orig_shape = x_drug.shape
                x_drug = x_drug.view(-1, orig_shape[-1])
                x_drug = self.batch_conv1(x_drug)
                x_drug = x_drug.view(*orig_shape)
        
        print(f"x_drug shape after batch norm: {x_drug.shape}")
        
        for i in range(len(self.units_list) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.relu(x_drug)
            
            if self.use_bn:
                # Apply batch norm with same shape handling
                if len(x_drug.shape) == 2:
                    x_drug = self.graph_bn[i](x_drug)
                else:
                    orig_shape = x_drug.shape
                    x_drug = x_drug.view(-1, orig_shape[-1])
                    x_drug = self.graph_bn[i](x_drug)
                    x_drug = x_drug.view(*orig_shape)
        
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.relu(x_drug)
        
        if self.use_bn:
            # Apply final batch norm with same shape handling
            if len(x_drug.shape) == 2:
                x_drug = self.batch_end(x_drug)
            else:
                orig_shape = x_drug.shape
                x_drug = x_drug.view(-1, orig_shape[-1])
                x_drug = self.batch_end(x_drug)
                x_drug = x_drug.view(*orig_shape)
        
        # Debug print before pooling
        print(f"Before pooling - x_drug shape: {x_drug.shape}, ibatch shape: {ibatch.shape}, ibatch dtype: {ibatch.dtype}")
        
        # FIX: Properly reshape tensors for global pooling
        if self.use_GMP:
            # Ensure x_drug is in the right format [num_nodes, features]
            if len(x_drug.shape) == 3:  # If shaped [batch, nodes, features]
                x_drug = x_drug.reshape(-1, x_drug.size(-1))
            
            # Ensure ibatch is a 1D tensor with appropriate batch indices
            if len(ibatch.shape) > 1:
                ibatch = ibatch.view(-1)
                
            # Make sure ibatch is long tensor type as required by scatter operations
            ibatch = ibatch.long()
            
            # Apply global max pooling
            x_drug = gmp(x_drug, ibatch)
        else:
            # Same reshape logic for mean pooling
            if len(x_drug.shape) == 3:
                x_drug = x_drug.reshape(-1, x_drug.size(-1))
                
            if len(ibatch.shape) > 1:
                ibatch = ibatch.view(-1)
                
            ibatch = ibatch.long()
            x_drug = global_mean_pool(x_drug, ibatch)
        
        print(f"x_drug shape after pooling: {x_drug.shape}")
            
        # -----cell line representation
        # -----mutation representation
        if self.use_mutation:
            x_mutation = torch.tanh(self.cov1(mutation_data))
            x_mutation = F.max_pool2d(x_mutation, (1, 5))
            x_mutation = F.relu(self.cov2(x_mutation))
            x_mutation = F.max_pool2d(x_mutation, (1, 10))
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = F.relu(self.fc_mut(x_mutation))
            # x_mutation = torch.dropout(x_mutation, 0.1, train=False)

        # ----gexpr representation
        if self.use_gexpr:
            x_gexpr = torch.tanh(self.fc_gexp1(gexpr_data))
            x_gexpr = self.batch_gexp1(x_gexpr)
            # x_gexpr = torch.dropout(x_gexpr,0.1, train=False)
            x_gexpr = F.relu(self.fc_gexp2(x_gexpr))

        # ----methylation representation
        if self.use_methylation:
            x_methylation = torch.tanh(self.fc_methy1(methylation_data))
            x_methylation = self.batch_methy1(x_methylation)
            # x_methylation = torch.dropout(x_methylation, 0.1, train=False)
            x_methylation = F.relu(self.fc_methy2(x_methylation))

        # ------Concatenate representations of three omics
        if self.use_gexpr==False and self.use_mutation==True and self.use_methylation==True:
            x_cell = torch.cat((x_mutation, x_methylation), 1)
        elif self.use_mutation==False and self.use_gexpr==True and self.use_methylation==True:
            x_cell = torch.cat((x_gexpr, x_methylation), 1)
        elif self.use_methylation==False and self.use_mutation==True and self.use_gexpr==True:
            x_cell = torch.cat((x_mutation, x_gexpr), 1)
        elif self.use_mutation==True and self.use_gexpr==True and self.use_methylation==True:
            x_cell = torch.cat((x_mutation, x_gexpr, x_methylation), 1)
        else:
            # Handle case where less than 2 features are used
            if self.use_mutation:
                x_cell = x_mutation
            elif self.use_gexpr:
                x_cell = x_gexpr
            elif self.use_methylation:
                x_cell = x_methylation
            else:
                raise ValueError("At least one of mutation, gexpr, or methylation must be used")
        
        x_cell = F.relu(self.fcat(x_cell))
        
        print(f"x_cell shape: {x_cell.shape}")
        print(f"x_drug shape before cat: {x_drug.shape}")
        
        # Combine representations of cell line and drug
        x_all = torch.cat((x_cell, x_drug), 0)
        
        print(f"x_all shape before batchc: {x_all.shape}")
        
        x_all = self.batchc(x_all)
        
        print(f"x_all shape after batchc: {x_all.shape}")
        
        return x_all

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.prelu1 = nn.PReLU(hidden_channels * heads)  # Adjusted for multiple heads

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        return x

class Summary(nn.Module):
    def __init__(self, ino, inn):
        super(Summary, self).__init__()
        self.fc1 = nn.Linear(ino + inn, 1)

    def forward(self, xo, xn):
        m = self.fc1(torch.cat((xo, xn), 1))
        m = torch.tanh(torch.squeeze(m))
        m = torch.exp(m) / (torch.exp(m)).sum()
        x = torch.matmul(m, xn)
        return x

class GraphCDR(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index):
        super(GraphCDR, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(100, 10)
        self.fd = nn.Linear(100, 10)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        # ---CDR_graph_edge and corrupted CDR_graph_edge
        if not isinstance(edge, torch.Tensor):
            edge = torch.tensor(edge)  # Convert NumPy array to PyTorch tensor

        pos_edge = edge[edge[:, 2] == 1, :2].T.contiguous().long()
        neg_edge = edge[edge[:, 2] == -1, :2].T.contiguous().long()
        
        # ---cell+drug node attributes
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        
        # ---cell+drug embedding from the CDR graph and the corrupted CDR graph
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        
        # ---graph-level embedding (summary)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        
        # ---embedding at layer l
        cellpos = pos_z[:self.index, ]
        drugpos = pos_z[self.index:, ]
        
        # ---embedding at layer 0
        cellfea = self.fc(feature[:self.index, ])
        drugfea = self.fd(feature[self.index:, ])
        cellfea = torch.sigmoid(cellfea)
        drugfea = torch.sigmoid(drugfea)
        
        # ---concatenate embeddings at different layers (0 and l)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        
        # ---inner product
        pos_adj = torch.matmul(cellpos, drugpos.t())
        pos_adj = self.act(pos_adj)
        
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value


    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(torch.clamp(self.discriminate(pos_z, summary, sigmoid=True), min=EPS)).mean()
        neg_loss = -torch.log(torch.clamp(1 - self.discriminate(neg_z, summary, sigmoid=True), min=EPS)).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
