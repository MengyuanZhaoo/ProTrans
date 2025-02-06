# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.utils.data as data


def encode_gene(df_exp, file_embeding): # (cell, gene)
    if os.path.exists(file_embeding):
        embedding_data = np.load(file_embeding)
        gene = embedding_data['gene']
        embedding = embedding_data['embedding']
        print('Successfully loaded ' + str(len(gene)) + ' gene embeddings from ' + file_embeding + '.')
        return gene, embedding
    else:
        embedding_data_all = np.load('../dataset/gene_embedding/dna2vec_1w.npz')
        gene_all = embedding_data_all['gene']
        embedding_all = embedding_data_all['embedding']

        df_gene = df_exp.T # (gene, cell)
        gene_temp = np.array(df_gene.index)
        gene = []
        embedding = []
        for i in range(len(gene_temp)):
            if gene_temp[i] in gene_all:
                gene.append(gene_temp[i])
                embedding.append(embedding_all[gene_all == gene_temp[i]][0])
        
        print('We got '+ str(len(gene))+' genes with ' +str(len(embedding[0])) + '-dimentional embedding.')
        np.savez(file_embeding, gene=gene, embedding=embedding)
        print('Gene embeddings have been saved in' + file_embeding +'.')
        return gene, embedding



def encode_protein(df_protein, file_embedding): # (cell, protein)
    if os.path.exists(file_embedding):
        embedding_data = np.load(file_embedding)
        protein = embedding_data['protein']
        embedding = embedding_data['embedding']
        print('Successfully loaded ' + str(len(protein)) + ' protein embeddings from ' + file_embedding + '.')
        return protein, embedding
    else:
        file_antigen = '../dataset/protein_embedding/uniprot_protein_antigen_information_human.csv'
        file_sequence = '../dataset/protein_embedding/uniprot_protein_information_human.csv'

        protein_temp = np.array(df_protein.columns)
        antigen_temp = [antigen.split('_')[0] for antigen in protein_temp]
        antigen_temp = [antigen.split('-')[0] for antigen in antigen_temp]
        antigen_temp = [antigen.split('(')[0] for antigen in antigen_temp]
        antigen_temp = [antigen.split('/')[0] for antigen in antigen_temp]
        
        df_antigen = pd.read_csv(file_antigen, sep=',', header= 0)
        antigen_protein_dict = dict(zip(df_antigen['antigen_name'], df_antigen['Protein_name_prime']))
        df_sequence = pd.read_csv(file_sequence, sep=',', header= 0)
        protein_entry_dict = dict(zip(df_sequence['Protein_name_prime'], df_sequence['Entry']))

        protein = []
        embedding = []
        # ProtT5
        with h5py.File('../dataset/protein_embedding/embedding_ProtT5.h5', "r") as file:
            print("Number of entries in ProtT5 file: {}".format(len(file.keys())))
            for i in range(len(protein_temp)):
                if protein_temp[i] not in protein and antigen_temp[i] in antigen_protein_dict.keys() and protein_entry_dict[antigen_protein_dict[antigen_temp[i]]] in file.keys():
                    protein.append(protein_temp[i])
                    embedding.append(np.array(file[protein_entry_dict[antigen_protein_dict[antigen_temp[i]]]]))
                elif protein_temp[i] not in protein and antigen_temp[i] in protein_entry_dict.keys() and protein_entry_dict[antigen_temp[i]] in file.keys():
                    protein.append(protein_temp[i])
                    embedding.append(np.array(file[protein_entry_dict[antigen_temp[i]]]))
        print('We got '+ str(len(protein))+' proteins with '+str(len(embedding[0]))+'-dimentional embedding.')
        print('Protein embeddings have been saved in ' + file_embedding +'.')
        np.savez(file_embedding, protein=protein, embedding=embedding)
        return np.array(protein), np.array(embedding)

