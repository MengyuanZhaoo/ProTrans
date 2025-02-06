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
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode Batch
```
4. The proteomics translation across technologies
```Bash
python ProTrans.py --data_dir ../dataset/GSE200417 --out_dir ./result
```
5.  The proteomics translation while saving gene-protein relationship
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode all --attention True
```
6.  The proteomics translation with zeroshot machanism
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result-zeroshot --preprocessed True --epochs 5 --mode zeroshot
```
