# A deep learning model for data-driven discovery of functional connectivity


https://doi.org/10.3390/a14030075


Usman Mahmood, Zengin Fu, Vince D. Calhoun, Sergey M. Plis





#### Dependencies:
* PyTorch
* Scikit-Learn
* torch-geometric

```bash
conda install pytorch torchvision -c pytorch
conda install sklearn
```

### Installation 
Refer to 

https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

for torch-geometric installation

```bash
# PyTorch
conda install pytorch torchvision -c pytorch
git clone https://github.com/UsmanMahmood27/BrainGNN.git
cd BrainGNN
pip install -e .
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric
```


