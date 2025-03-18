import torch
import time
import argparse
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from model import *
from data_process import process
from data_load import dataload

parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--alph', type=float, default=0.30, help='Weight for DGI positive loss')
parser.add_argument('--beta', type=float, default=0.30, help='Weight for DGI negative loss')
parser.add_argument('--epoch', type=int, default=350, help='Number of epochs')
parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden layer size')
parser.add_argument('--output_channels', type=int, default=100, help='Output feature size')
args = parser.parse_args()

start_time = time.time()

# Data file paths
Drug_info_file = '../data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file = '../data/Drug/drug_threshold.txt'
Drug_feature_file = '../data/Drug/drug_graph_feat'
Cell_line_info_file = '../data/Celline/Cell_lines_annotations.txt'
Genomic_mutation_file = '../data/Celline/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '../data/Celline/GDSC_IC50.csv'
Gene_expression_file = '../data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'

# Load data
drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = dataload(
    Drug_info_file, IC50_threds_file, Drug_feature_file, Cell_line_info_file,
    Genomic_mutation_file, Cancer_response_exp_file, Gene_expression_file, Methylation_file
)

# Process data
drug_set, cellline_set, train_edge, label_pos, train_mask, test_mask, atom_shape = process(
    drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs
)

# Define Model
model = GraphCDR(
    hidden_channels=args.hidden_channels,
    encoder=Encoder(args.output_channels, args.hidden_channels),
    summary=Summary(args.output_channels, args.hidden_channels),
    feat=NodeRepresentation(atom_shape, gexpr_feature.shape[-1], methylation_feature.shape[-1], args.output_channels),
    index=nb_celllines
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
loss_fn = nn.BCELoss()


# Metric Calculation
def compute_metrics(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)  # Convert probabilities to binary (threshold = 0.5)
    auc = roc_auc_score(y_true, y_pred_probs)
    aupr = average_precision_score(y_true, y_pred_probs)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return auc, aupr, f1, acc


# Training Function
def train():
    model.train()
    total_loss = 0
    for drug, cell in zip(drug_set, cellline_set):
        optimizer.zero_grad()
        pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
            drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
        )

        dgi_pos = model.loss(pos_z, neg_z, summary_pos)
        dgi_neg = model.loss(neg_z, pos_z, summary_neg)
        pos_loss = loss_fn(pos_adj[train_mask], label_pos[train_mask])

        loss = (1 - args.alph - args.beta) * pos_loss + args.alph * dgi_pos + args.beta * dgi_neg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Training Loss: {round(total_loss, 4)}')


# Testing Function
def test():
    model.eval()
    with torch.no_grad():
        for drug, cell in zip(drug_set, cellline_set):
            _, _, _, _, pre_adj = model(
                drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
            )
            test_loss = loss_fn(pre_adj[test_mask], label_pos[test_mask])
        
        y_pred_probs = pre_adj[test_mask].cpu().numpy()
        y_true = label_pos[test_mask].cpu().numpy()
        
        AUC, AUPR, F1, ACC = compute_metrics(y_true, y_pred_probs)
        
        print(f'Test Loss: {round(test_loss.item(), 4)}')
        print(f'Test AUC: {round(AUC, 4)} | Test AUPR: {round(AUPR, 4)} | '
              f'Test F1: {round(F1, 4)} | Test ACC: {round(ACC, 4)}')

    return AUC, AUPR, F1, ACC


# Main Training Loop
final_AUC, final_AUPR, final_F1, final_ACC = 0, 0, 0, 0
for epoch in range(args.epoch):
    print(f'\nEpoch {epoch}')
    train()
    AUC, AUPR, F1, ACC = test()
    
    if AUC > final_AUC:  # Track best model performance
        final_AUC, final_AUPR, final_F1, final_ACC = AUC, AUPR, F1, ACC

elapsed_time = time.time() - start_time
print('---------------------------------------')
print(f'Elapsed Time: {round(elapsed_time, 4)} sec')
print(f'Final AUC: {round(final_AUC, 4)} | Final AUPR: {round(final_AUPR, 4)} | '
      f'Final F1: {round(final_F1, 4)} | Final ACC: {round(final_ACC, 4)}')
print('---------------------------------------')
