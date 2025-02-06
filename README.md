# ProTrans
A sequence knowledge-guided deep learning method for single-cell zero-shot translation (ProTrans)
# Ovreview

# Requirements
anndata==0.11.3
h5py==3.10.0
numpy==2.2.2
pandas==2.2.3
scanpy==1.10.4
scikit_learn==1.4.2
scipy==1.15.1
scvi==0.6.8
torch==2.0.0

# Datasets
All the original datasets can be downloaded from [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122), [GSE100866](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866), [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378), [GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639), [GSE156473](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156473), [GSE200417](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200417), [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013), [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583).

# Usage


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
| `transpose` | `bool` | `False` | A boolean flag determining whether to transpose the input data, defaulting to `False`. |
| `attention` | `bool` | `False` | Whether to save the gene-protein relationship matrix. |

Setting mode as cell type name (e.g. Mono), ProTrans will use the Mono cells as test set and the other cells as training set.
Setting mode as batch name (e.g. Batch1), ProTrans will use the cells belonging to Batch1 as test set and the other cells as training set.
Setting mode as 'zeroshot', ProTrans will randomly divides all proteins into training and test sets according to 6 to 4.
Setting mode as 'all', ProTrans will use all cells to train model.
Setting mode as '', ProTrans will randomly divides all cells into training and test sets according to 6 to 4. 
1. The proteomics translation for intra-datasets
- Run with raw RNA and protein expression file
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result
```
- Run with preprocessed RNA and protein expression file
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True
```
2. The proteomics translation across cell types
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode Mono
```
- Run with raw RNA and protein expression file
3. The proteomics translation across batches
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode Batch1
```
4. The proteomics translation across technologies
```Bash
python ProTrans-technology.py --data_dir ../dataset/GSE200417 --out_dir ./result
```
5.  The proteomics translation while saving gene-protein relationship
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode all --attention True
```
6.  The proteomics translation with zeroshot machanism
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result-zeroshot --preprocessed True --mode zeroshot
```
