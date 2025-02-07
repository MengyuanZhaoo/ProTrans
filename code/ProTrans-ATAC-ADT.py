# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import numpy as np
import pandas as pd
import anndata as ad
import scvi
import scipy
import scanpy as sc
import episcanpy.api as epi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.parallel import DataParallel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from encode import encode_protein
        
def encode_cell_all(mrna_data, file_embedding):
    if os.path.exists(file_embedding):
        embedding_data = np.load(file_embedding)
        embedding = embedding_data['embedding'] # (3158, 100)
        imputed = embedding_data['imputed'] # (cell, gene)
        print('Successfully loaded ' + str(embedding.shape[0]) + ' cell embeddings from ' + file_embedding + '.')
        return embedding, imputed
    else:
        adata = ad.AnnData(X=mrna_data.values, dtype=np.float32) # (cell, gene)
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


def TFIDF(count_mat): 
    """
    TF-IDF transformation for matrix.

    Parameters
    ----------
    count_mat
        numpy matrix with cells as rows and peak as columns, cell * peak.

    Returns
    ----------
    tfidf_mat
        matrix after TF-IDF transformation.

    divide_title
        matrix divided in TF-IDF transformation process".

    multiply_title
        matrix multiplied in TF-IDF transformation process".

    """

    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    nfreqs = 1.0 * count_mat / divide_title
    multiply_title = np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1]))
    tfidf_mat = scipy.sparse.csr_matrix(np.multiply(nfreqs, multiply_title)).T
    return tfidf_mat, divide_title, multiply_title

def ATAC_data_preprocessing(
    ATAC_data,
    binary_data = True,
    filter_features = True,
    fpeaks = 0.005,
    tfidf = True,
    normalize = True,
    save_data = False,
    file_path = None,
    logging_path = None
):
    """
    Preprocessing for ATAC data, we choose binarize, peaks filtering, TF-IDF transformation and scale transformation, using scanpy.
    
    Parameters
    ----------
    ATAC_data: Anndata
        ATAC anndata for processing.
        
    binary_data: bool
        choose binarized ATAC data or not, default True.
        
    filter_features: bool
        choose use peaks filtering or not, default True.
        
    fpeaks: float
        filter out the peaks expressed less than fpeaks*n_cells, if don't filter peaks set it None, default 0.005.
        
    tfidf: bool
        choose using TF-IDF transform or not, default True.
    
    normalize: bool
        choose set data to [0, 1] or not, default True.
        
    save_data: bool
        choose save the processed data or not, default False.
        
    file_path: str
        the path for saving processed data, only used if save_data is True, default None.
   
    logging_path: str
        the path for output process logging, if not save, set it None, default None.

    Returns
    ---------
    ATAC_data_processed: Anndata 
        ATAC data with binarization, peaks filtering, TF-IDF transformation and scale transformation preprocessed.

    divide_title: numpy matrix
        matrix divided in TF-IDF transformation process".

    multiply_title: numpy matrix
        matrix multiplied in TF-IDF transformation process".
        
    max_temp: float
        max scale factor divided in process".
        
    """
    ATAC_data_processed = ATAC_data.copy()
    divide_title, multiply_title, max_temp = None, None, None

    if binary_data:
        epi.pp.binarize(ATAC_data_processed)
        
    if filter_features:
        epi.pp.filter_features(ATAC_data_processed, min_cells=np.ceil(fpeaks*ATAC_data.shape[0]))

    if tfidf:
        count_mat = ATAC_data_processed.X.copy()
        count_mat = count_mat.astype(np.float32)
        ATAC_data_processed.X, divide_title, multiply_title = TFIDF(count_mat)
    
    if normalize:
        max_temp = np.max(ATAC_data_processed.X)
        ATAC_data_processed.X = ATAC_data_processed.X / max_temp
    
    if save_data:
        if filter_features:
            ATAC_data.write_h5ad(file_path + '/binarize_' + str(binary_data) + '_filter_' + str(filter_features) + '_fpeaks_' + str(fpeaks)  + '_tfidf_' + str(tfidf) + '_normalize_' + str(normalize) + '_ATAC_processed_data.h5ad')
        else:
            ATAC_data.write_h5ad(file_path + '/binarize_' + str(binary_data) + '_filter_' + str(filter_features) + '_tfidf_' + str(tfidf) + '_normalize_' + str(normalize) + '_ATAC_processed_data.h5ad')

    return ATAC_data_processed, divide_title, multiply_title, max_temp


class ProTrans(nn.Module):
    def __init__(self, head=10, head_dim=100):
        super(ProTrans, self).__init__()
        self.head = head
        self.head_dim = head_dim
        self.query_linear = nn.Linear(1124, 1000)
        self.key_linear = nn.Linear(2100, 1000)
        self.linear_1 = nn.Linear(head, 10)
        self.linear_2 = nn.Linear(10, 1)

    def forward(self, query, key, value):
        query = self.query_linear(query)
        query = query.reshape(query.size(0), query.size(1), self.head, self.head_dim)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, query.size(1), query.size(3))

        key = self.key_linear(key)
        key = key.reshape(key.size(0), key.size(1), self.head, self.head_dim)
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
    parser = argparse.ArgumentParser(description='ProTrans-ATAC-ADT.py')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preprocess_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../dataset/REAP-PBMC')
    parser.add_argument('--out_dir', type=str, default='../result/REAP-PBMC')
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
    model = ProTrans()
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    print('We are using device: '+str(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Load ATAC and Protein Data
    atac = pd.read_csv(args.data_dir+'/ATAC-ADT/atac.csv', sep = ',', header=0, index_col= 0 )  # (cell, peak)
    protein =  pd.read_csv(args.data_dir+'/ATAC-ADT/protein.csv', sep = ',', header=0, index_col= 0) # (cell, protein)
    # filter peaks with chr
    atac = atac[[col for col in atac.columns if col.startswith('chr')]]
    # align cell number
    cells = np.intersect1d(np.array(atac.index),np.array(protein.index))
    atac = atac.loc[cells,:]
    protein = protein.loc[cells,:]
    print('Raw atac data shape:'+str(atac.shape))
    obs = pd.DataFrame({'cell':np.array(atac.index)})
    var = pd.DataFrame({'peak':np.array(atac.columns)})
    adata_atac = ad.AnnData(X=atac.values, obs=obs, var=var)
    adata_atac, _, _, _ = ATAC_data_preprocessing(adata_atac)
    sc.pp.highly_variable_genes(adata_atac,  n_top_genes=5000)
    adata_atac = adata_atac[:, adata_atac.var['highly_variable']]
    atac_data_all = pd.DataFrame(adata_atac.X.toarray(), columns=adata_atac.var['peak'].values, index=adata_atac.obs['cell'].values)

    print('Raw protein data shape:'+str(protein.shape))
    if protein.shape[0] > len(cells):
        protein = protein.loc[cells,:]
    adata_protein = ad.AnnData(X=protein.values, obs=pd.DataFrame({'cell':np.array(protein.index)}), var=pd.DataFrame({'protein':np.array(protein.columns)}))
    sc.pp.normalize_total(adata_protein, target_sum = 10000)
    sc.pp.log1p(adata_protein)
    protein_data = pd.DataFrame(adata_protein.X, columns=adata_protein.var['protein'].values, index=adata_protein.obs['cell'].values)
    protein_list, protein_embedding = encode_protein(protein_data, args.data_dir + '/ATAC-ADT/protein_embedding.npz')
    protein_data_all = protein_data[protein_list]

    # Save preprocessed atac and protein data
    atac_data_all.to_csv(args.data_dir + '/ATAC-ADT/atac_preprocess.csv', sep=',', index=True, header=True)
    protein_data_all.to_csv(args.data_dir + '/ATAC-ADT/protein_preprocess.csv', sep=',', index=True, header=True)

    # filter highly viriable peaks
    embedding_data = np.load(args.data_dir + '/atac_emb.npz', allow_pickle=True)
    peaks = embedding_data['peak']
    embedding = embedding_data['embedding']
    df_peak_emb = pd.DataFrame(embedding, index=peaks)
    atac_list = atac_data_all.columns.values
    atac_embedding = embedding[np.isin(peaks, atac_list)]
    print('Successfully loaded ' + str(len(atac_list)) + ' peak embeddings.')
    assert atac_data_all.index.tolist() == protein_data_all.index.tolist()
    cell_all = np.array(atac_data_all.index.values)
    print('After cells alignment: '+str(atac_data_all.shape)+' '+str(protein_data_all.shape), flush = True)
    
    # Generate cell embedding
    cell_embedding_all, atac_data_imputed= encode_cell_all(atac_data_all, args.data_dir + '/ATAC-ADT/cell_embedding.npz')
 
    # Random split train:test = 6:4
    print('Testing model on random training:test = 6:4')
    dataset_all = data.TensorDataset(torch.tensor(atac_data_all.values, dtype=torch.float32),
                                    torch.tensor(protein_data_all.values, dtype=torch.float32),
                                    torch.tensor(cell_embedding_all, dtype=torch.float32),
                                    torch.tensor(np.arange(0, cell_embedding_all.shape[0]).reshape(-1, 1), dtype=torch.int32))
    
    cell_num_test = int(cell_embedding_all.shape[0]*0.4)
    cell_num_train = cell_embedding_all.shape[0] - cell_num_test
    dataset, dataset_test = data.random_split(dataset = dataset_all,lengths = [cell_num_train, cell_num_test])
    print('{} cells for training, {} cells for testing'.format(cell_num_train, cell_num_test), flush = True)

    # Define non-cell-specific Query and Key Vector
    query = torch.tensor(protein_embedding, dtype=torch.float32)
    query = query.unsqueeze(0).repeat(args.batch_size, 1, 1)
    query = query.to(device)
    key = torch.tensor(atac_embedding, dtype=torch.float32)
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
    torch.cuda.empty_cache()
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
    df_prediction_all = pd.DataFrame(prediction_all, columns=list(protein_list), index = list(cell_pre)).T
    df_prediction_all = df_prediction_all.fillna(value=0)
    df_prediction_all.to_csv(args.out_dir + '/prediction.csv')
    df_truthvalue_all = pd.DataFrame(truthvalue_all, columns=list(protein_list), index = list(cell_pre)).T
    df_truthvalue_all = df_truthvalue_all.fillna(value=0)
    df_truthvalue_all.to_csv(args.out_dir + '/truthvalue.csv')
