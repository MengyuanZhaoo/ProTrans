# scProTrans
A sequence knowledge-guided deep learning method for single-cell multi-omics translation (scProTrans)
# Ovreview
Proteins, as direct executors of cellular biological functions, are central to understanding cellular life activities, disease mechanisms, and therapeutic strategies. Despite their importance, proteomics data remain scarce compared to the abundance of single-cell RNA sequencing (scRNA-seq) data, primarily due to experimental limitations and high costs. Advances of multi-omics sequencing technologies frame the pathway between transcriptomics and proteomics. A promising strategy involves leveraging multi-omics datasets to train models that translate scRNA-seq data into proteomics profiles, thereby constructing comprehensive multi-omics profiles. Here, we introduce ProTrans, a sequence knowledge-guided deep learning framework that bridges transcriptomics and proteomics by deciphering gene-protein relationships from CITE-seq datasets. ProTrans integrates gene, protein, and cell encoding to uncover cell-specific associations and enable zero-shot translation through sequence-to-embedding-to-profile learning. Extensive evaluations across 15 multi-omics datasets demonstrate that ProTrans surpasses state-of-the-art methods in proteomics translation and enhances downstream analyses, including cell clustering, subtype identification, and biomarker discovery. Additionally, ProTrans is extended to tri-omics scenarios by refactoring encoders, demonstrating its flexibility and scalability. Significantly, ProTrans not only elucidates cell-specific gene-protein relationships but also predicts protein profiles that are challenging to capture experimentally.

![Figure1](https://github.com/user-attachments/assets/5a078b33-06d7-435e-a04c-dfa50b860bda)

# Requirements
```Bash
anndata==0.11.3
h5py==3.10.0
numpy==2.2.2
pandas==2.2.3
scanpy==1.10.4
scikit_learn==1.4.2
scipy==1.15.1
scvi==0.6.8
torch==2.0.0
```

# Datasets
All the original datasets can be downloaded from [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122), [GSE100866](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866), [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378), [GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639), [GSE156473](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156473), [GSE200417](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200417), [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013), [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583). We have released the pretrained gene and protein sequence embeddings with [link1](https://drive.google.com/file/d/1LbC_xtLxgTd3cqdjKmJkzDIhW1GDaUqP/view?usp=sharing) and [link2](https://drive.google.com/file/d/1KgDbkDumm-nAA4SADiOCTz0JVWlpaxtx/view?usp=sharing).

# Usage

## Detailed explanation of parameters
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data_dir` | `str` | `./data` | Directory containing the input data. |
| `out_dir` | `str` | `./result` | Directory to save the output results. |
| `mode` | `str` | `''` | Mode of translation (cell type name, batch name, 'zeroshot', 'all' or ''). |
| `batch_size` | `int` | `48` | Number of samples per batch during training. |
| `epochs` | `int` | `200` | Total number of epochs to train the model. |
| `lr` | `float` | `0.001` | Learning rate for the optimizer. |
| `patience` | `int` | `50` | Number of epochs to wait for improvement before early stopping. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `preprocessed` | `bool` | `False` | Whether to use preprocessed data. |
| `transpose` | `bool` | `False` | Whether to transpose the input data. If the columns of input data are cells, set transpose to True. |
| `attention` | `bool` | `False` | Whether to save the gene-protein relationship matrix. |

Users specify the mode parameter to perform translation tasks under different scenarios.
Setting mode as cell type name (e.g. Mono), ProTrans will use the Mono cells as test set and the other cells as training set.
Setting mode as batch name (e.g. Batch1), ProTrans will use the cells belonging to Batch1 as test set and the other cells as training set.
Setting mode as 'zeroshot', ProTrans will randomly divides all proteins into training and test sets according to 6 to 4.
Setting mode as 'all', ProTrans will use all cells to train model.
Setting mode as '', ProTrans will randomly divides all cells into training and test sets according to 6 to 4. 

Users need to provide rna.csv and protein.csv of raw expression reads to train or evaluate ProTrans. The annotation file is optional which is used to enable omics translation across cell types or batches. Taking GSE164378 as example, the directory and specific instructions for input files are as follows:
```
 |-- dataset
        |-- GSE164378
              |-- rna.csv           # (cell, gene)
              |-- protein.csv       # (cell, protein)
              |-- annotation.csv    # (cell, annotation)    
```
All output files will be saved in out_dir.

## The proteomics translation for intra-datasets
- Run with raw RNA and protein expression file
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result
```
- Run with preprocessed RNA and protein expression file
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True
```
## The proteomics translation across cell types
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode Mono
```
## The proteomics translation across batches
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode Batch1
```
## The proteomics translation while saving gene-protein relationship
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode all --attention True
```
##  The proteomics translation with zeroshot machanism
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result-zeroshot --preprocessed True --mode zeroshot
```
## The proteomics translation across technologies
Taking GSE200417 as example, the directory and specific instructions for input files are as follows:
```
 |-- dataset
        |-- GSE200417
            |-- CITE
                |-- rna.csv          # (cell, gene)
                |-- protein.csv      # (cell, protein)      
            |-- DOGMA
                |-- rna.csv          # (cell, gene)
                |-- protein.csv      # (cell, protein)
```
Run the command as follows:
```Bash
python ProTrans-technology.py --data_dir ../dataset/GSE200417 --out_dir ./result
```
## Extending ProTrans to tri-omics translation
Taking GSM5123953 as example, users follow gen_atac.ipynb to convert the h5 file into atac.csv (cell*peak). Next, referring to atac2seq.ipynb to extract sequences corresponding to peaks, then follow seq2emb.ipynb to generate atac_emb.npz used in translation process. In addition, users need to unzip dataset/dna2vec/pre-trained DNA-8mers.7z to pre-trained DNA-8mers.txt.

the directory and specific instructions for input files are as follows:
```
 |-- dataset
        |-- GSM5123953
            |-- ATAC-ADT
                |-- atac.csv          # (cell, peak)
                |-- protein.csv       # (cell, protein)      
            |-- ATAC-RNA
                |-- atac.csv          # (cell, peak)
                |-- rna.csv           # (cell, protein)
```
- The proteomics translation based on epigenomics
```Bash
python ProTrans-ATAC-ADT.py --data_dir ../dataset/GSM5123953 --out_dir ./result
```

- The transcriptomics translation based on epigenomics
```Bash
python ProTrans-ATAC-RNA.py --data_dir ../dataset/GSM5123953 --out_dir ./result
```

