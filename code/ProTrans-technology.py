# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import numpy as np
import pandas as pd
import anndata as ad
import scvi
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.parallel import DataParallel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from encode import encode_protein, encode_gene

        
def encode_cell_all(mrna_data, file_embedding):
    if os.path.exists(file_embedding):
        embedding_data = np.load(file_embedding)
        embedding = embedding_data['embedding']
        imputed = embedding_data['imputed']
        print('Successfully loaded ' + str(embedding.shape[0]) + ' cell embeddings from ' + file_embedding + '.')
        return embedding, imputed
    else:
        adata = ad.AnnData(X=mrna_data.values, dtype=np.float32)
        adata.var['gene'] = np.array(mrna_data.columns)
        adata.obs['cell'] = np.array(mrna_data.index)
        scvi.model.SCVI.setup_anndata(adata)
        vae = scvi.model.SCVI(adata, n_latent = 100)
        vae.train()
        adata.obsm["X_scVI"] = vae.get_latent_representation()
        adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

        cell = mrna_data.index.tolist()
        cell_embedding = np.array(adata.obsm["X_scVI"])
        imputed = np.array(adata.obsm["X_normalized_scVI"])
        np.savez(file_embedding, cell=cell, embedding=cell_embedding, imputed = imputed)
        print('We got '+ str(len(cell_embedding))+' cells with 100-dimentional embedding and save in ' + file_embedding +'.')
        return cell_embedding, imputed

class mmTrans(nn.Module):
    def __init__(self, head=10, head_dim=50):
        super(mmTrans, self).__init__()
        self.head = head
        self.head_dim = head_dim
        self.query_linear = nn.Linear(1124, 500)
        self.key_linear = nn.Linear(10100, 500)
        self.linear_1 = nn.Linear(head, 10)
        self.linear_2 = nn.Linear(10, 1)

    def forward(self, query, key, value):
        query = self.query_linear(query)
        query = query.reshape(query.size(0), query.size(1), self.head, 50)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, query.size(1), query.size(3))

        key = self.key_linear(key)
        key = key.reshape(key.size(0), key.size(1), self.head, 50)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, key.size(1), key.size(3))

        value = value.unsqueeze(2)
        value = value.repeat(1, 1, self.head)
        value = value.reshape(value.size(0), value.size(1), self.head, 1)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, value.size(1), value.size(3))

        protein_value, _ = self.protein_rna_attention(query, key, value)
        protein_value = protein_value.reshape(self.head, -1, protein_value.size(1), protein_value.size(2))
        protein_value = protein_value.permute(1, 2, 0, 3).contiguous().view(protein_value.size(1), protein_value.size(2), -1)
        protein_value = self.linear_1(protein_value)
        protein_value = F.relu(protein_value)
        protein_value = self.linear_2(protein_value)
        protein_value = F.relu(protein_value)
        protein_value = protein_value.squeeze(2)
        return protein_value

    def protein_rna_attention(self, query, key, value):
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / np.power(self.head_dim, 0.5)
        attn = torch.softmax(attn, dim=2)
        output = torch.bmm(attn, value)
        return output, attn


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='ProTrans-technology.py')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--transpose', type=bool, default=False)
    parser.add_argument('--data_dir', type=str, default='../dataset/GSE200417')
    parser.add_argument('--out_dir', type=str, default='../result/GSE200417')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = mmTrans()
    if torch.cuda.device_count() > 1: 
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    print('We are using device: '+str(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Load train and test RNA Data
    mrna_train = pd.read_csv(args.data_dir+'/CITE/rna.csv', sep = ',', header=0, index_col= 0 )  # (cell, gene)
    protein_train =  pd.read_csv(args.data_dir+'/CITE/protein.csv', sep = ',', header=0, index_col= 0) # (cell, protein)
    mrna_test = pd.read_csv(args.data_dir+'/DOGMA/rna.csv', sep = ',', header=0, index_col= 0 )  # (cell, gene)
    protein_test =  pd.read_csv(args.data_dir+'/DOGMA/protein.csv', sep = ',', header=0, index_col= 0) # (cell, protein)
    proteins = []
    for col in protein_test.columns:
        proteins.append(col[:col.rfind('-')])
    protein_test.columns = proteins
    if mrna_test.shape[0] != protein_test.shape[0]:
        mrna_test = mrna_test.loc[np.intersect1d(np.array(mrna_test.index),np.array(protein_test.index)),:]
        protein_test = protein_test.loc[np.intersect1d(np.array(mrna_test.index),np.array(protein_test.index)),:]

    mrna_data = pd.concat([mrna_train, mrna_test], axis=0)
    print('Raw mRNA data shape:'+str(mrna_data.shape))
    obs = pd.DataFrame({'cell':np.array(mrna_data.index)})
    var = pd.DataFrame({'gene':np.array(mrna_data.columns)})
    adata_mrna = ad.AnnData(X=mrna_data.values, obs=obs, var=var, dtype=np.float32)
    cell_train = np.array(mrna_test.index)
    cell_test = np.array(mrna_train.index)

    sc.pp.filter_genes(adata_mrna, min_cells = 10)
    sc.pp.normalize_total(adata_mrna, target_sum = 10000)
    sc.pp.log1p(adata_mrna)
    sc.pp.highly_variable_genes(adata_mrna,  n_top_genes=5000)
    adata_mrna = adata_mrna[:, adata_mrna.var['highly_variable']]
    mrna_data = pd.DataFrame(adata_mrna.X, columns=adata_mrna.var['gene'].values, index=adata_mrna.obs['cell'].values)
    gene_list, gene_embedding = encode_gene(mrna_data, args.data_dir + '/gene_embedding.npz')
    mrna_data_all = mrna_data[gene_list]
    cells = np.array(mrna_data_all.index)

    # Load train and test Protein Data
    protein_data = pd.concat([protein_train, protein_test], axis=0)
    print('Raw protein data shape:'+str(protein_data.shape))
    if protein_data.shape[0] > len(cells):
        protein_data = protein_data.loc[cells,:]
    adata_protein = ad.AnnData(X=protein_data.values, obs=pd.DataFrame({'cell':np.array(protein_data.index)}), var=pd.DataFrame({'protein':np.array(protein_data.columns)}), dtype=np.float32)  # (3158, 19253) (cell, gene)
    sc.pp.normalize_total(adata_protein, target_sum = 10000)
    sc.pp.log1p(adata_protein)
    protein_data = pd.DataFrame(adata_protein.X, columns=adata_protein.var['protein'].values, index=adata_protein.obs['cell'].values)
    protein_list, protein_embedding = encode_protein(protein_data, args.data_dir + '/protein_embedding.npz')
    protein_data_all = protein_data[protein_list]

    assert mrna_data_all.index.tolist() == protein_data_all.index.tolist()
    cell_all = np.array(mrna_data_all.index)
    print('After cells alignment: '+str(mrna_data_all.shape)+' '+str(protein_data_all.shape), flush = True)
    
    # Generate cell embedding
    cell_embedding_all, mrna_data_imputed= encode_cell_all(mrna_data_all, args.data_dir + '/cell_embedding.npz')  # (3158, 100)

    print('Testing model on from CITE to DOGMA...')
    df_cell = pd.DataFrame({'k':np.arange(0, mrna_data_all.shape[0], 1), 'v':cell_all})
    cell_embedding_test_index = df_cell[df_cell['v'].isin(cell_test)]['k'].values
    cell_embedding_train_index = df_cell[~df_cell['v'].isin(cell_test)]['k'].values
    print('{} cells for training, {} cells for testing'.format(len(cell_embedding_train_index), len(cell_embedding_test_index)), flush = True)

    dataset_test = data.TensorDataset(torch.tensor(mrna_data_all.loc[cell_test].values, dtype=torch.float32),
                                    torch.tensor(protein_data_all.loc[cell_test].values, dtype=torch.float32),
                                    torch.tensor(cell_embedding_all[cell_embedding_test_index,:], dtype=torch.float32),
                                    torch.tensor(cell_embedding_test_index, dtype=torch.int32))
    dataset = data.TensorDataset(torch.tensor(mrna_data_all.loc[cell_train].values, dtype=torch.float32),
                                    torch.tensor(protein_data_all.loc[cell_train].values, dtype=torch.float32),
                                    torch.tensor(cell_embedding_all[cell_embedding_train_index,:], dtype=torch.float32),
                                    torch.tensor(cell_embedding_train_index, dtype=torch.int32))

    # Define non-cell-specific Query and Key Vector
    query = torch.tensor(protein_embedding, dtype=torch.float32)
    query = query.unsqueeze(0).repeat(args.batch_size, 1, 1)
    query = query.to(device)
    key = torch.tensor(gene_embedding, dtype=torch.float32)
    key = key.unsqueeze(0).repeat(args.batch_size, 1, 1)
    key = key.to(device)

    ############################## Train #################################
    data_loader_train = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Train Model
    print('Training model...')
    model.train()
    best_loss = np.inf
    not_improved_count = 0
    for epoch in range(args.epochs):
        loss_epoch = 0
        for i, (mrna, protein, cell_encoding, _) in enumerate(data_loader_train):
            cell_encoding = cell_encoding.unsqueeze(1)
            query_cell_encoding = cell_encoding.repeat(1, query.size(1), 1).to(device)
            query_cell_encoding = torch.cat((query, query_cell_encoding), dim=2)

            key_cell_encoding = cell_encoding.repeat(1, key.size(1), 1).to(device)
            key_cell_encoding = torch.cat((key, key_cell_encoding), dim=2)
            
            mrna = mrna.to(device)
            protein = protein.to(device) 
            optimizer.zero_grad()
            prediction = model(query_cell_encoding, key_cell_encoding, mrna)

            loss = criterion(prediction, protein)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            not_improved_count = 0
            torch.save(model, args.out_dir + '/best_model.pth')
        else:
            not_improved_count += 1
            if not_improved_count >= args.patience:
                print("Not improved for " + str(not_improved_count) +
                      " epochs, early stopping at Epoch " + str(epoch) + ".", flush=True)
                break

        print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') +
              ' Epoch: {}, Training loss: {:.4f}.'.format(epoch, loss_epoch), flush=True)
    
    ############################## Test #################################
    # Load Test Data
    # torch.cuda.empty_cache()
    test_data_loader = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # Test Model
    print('Testing model...')
    model = torch.load(args.out_dir + '/best_model.pth')
    model.eval()
    with torch.no_grad():
        prediction_all = []
        truthvalue_all = []
        cell_pre = []
        for i, (mrna, protein, cell_encoding, cell_index) in enumerate(test_data_loader):
            batch_size_test = mrna.size(0)
            query = query[:batch_size_test, :, :]
            key = key[:batch_size_test, :, :]
            cell_encoding = cell_encoding.unsqueeze(1)
            query_cell_encoding = cell_encoding.repeat(1, query.size(1), 1).to(device)
            query_cell_encoding = torch.cat((query, query_cell_encoding), dim=2)
            key_cell_encoding = cell_encoding.repeat(1, key.size(1), 1).to(device)
            key_cell_encoding = torch.cat((key, key_cell_encoding), dim=2)
            mrna = mrna.to(device)
            prediction = model(query_cell_encoding, key_cell_encoding, mrna)

            prediction_all.append(prediction.cpu().numpy())
            truthvalue_all.append(protein.cpu().numpy())
            cell_pre.extend(cell_all[cell_index.cpu().numpy().reshape(-1)])

        prediction_all = np.concatenate(prediction_all, axis=0)
        truthvalue_all = np.concatenate(truthvalue_all, axis=0)
    assert np.array_equal(np.array(cell_test), np.array(cell_pre))
    df_prediction_all = pd.DataFrame(prediction_all, columns=list(protein_list), index = list(cell_pre)).T
    df_prediction_all = df_prediction_all.fillna(value=0)
    df_prediction_all.to_csv(args.out_dir + '/prediction.csv')
    df_truthvalue_all = pd.DataFrame(truthvalue_all, columns=list(protein_list), index = list(cell_pre)).T
    df_truthvalue_all = df_truthvalue_all.fillna(value=0)
    df_truthvalue_all.to_csv(args.out_dir + '/truthvalue.csv')

    # Calculate the metrics with raw prediction
    df_metrics = pd.DataFrame(index=list(protein_list),
                            columns=['MSE', 'MAE', 'Correlation'])
    df_metrics = df_metrics.fillna(value=0)
    df_prediction_all = df_prediction_all.T
    prediction_all = np.nan_to_num(df_prediction_all.values)

    for i in range(len(protein_list)):
        prediction = prediction_all[:,i]
        truthvalue = truthvalue_all[:,i]
        mae = mean_absolute_error(truthvalue, prediction)
        mse = mean_squared_error(truthvalue, prediction)
        cosine = cosine_similarity(truthvalue.reshape(1, -1), prediction.reshape(1, -1))[0][0]
        df_metrics.loc[protein_list[i]] = np.array([mse, mae, cosine])

    df_metrics.loc['statistics'] = df_metrics.mean(axis=0)
    df_metrics.to_csv(args.out_dir + '/evaluate.csv', sep=',', index=True, header=True)
    print('MSE: {:.6f}, MAE: {:.6f}, Correlation: {:.6f}'.format(
        df_metrics.loc['statistics', 'MSE'], df_metrics.loc['statistics', 'MAE'], df_metrics.loc['statistics', 'Correlation']))
