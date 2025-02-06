# ProTrans
A sequence knowledge-guided deep learning method for single-cell zero-shot translation (ProTrans)

## Datasets
All the original datasets can be downloaded from [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122), [GSE100866](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866), [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378), [GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639), [GSE156473](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156473), [GSE200417](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200417), [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013), [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583).


1. The proteomics translation for intra-datasets
- Run with raw RNA and protein expression file
```Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result
```
- Run with preprocessed RNA and protein expression file
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True'''
2. The proteomics translation across cell types
'''Bash
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode Mono
'''
3. The proteomics translation across batches

4. The proteomics translation across technologies

5.  The proteomics translation while saving gene-protein relationship
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result --preprocessed True --mode all --attention True
python ProTrans.py --data_dir ../dataset/GSE164378 --out_dir ./result-zeroshot --preprocessed True --epochs 5 --mode zeroshot
