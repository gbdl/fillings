# Machine Learning Gravity Compactifications on Negatively Curved Manifolds

This repository implements the code for the paper 'Machine Learning Gravity Compactifications on Negatively Curved Manifolds', [arxiv:2501.00093](https://arxiv.org/abs/2501.00093). 

To perform a filling with $k = 5$, on a single cusp of the manifold $M_3$, run

```
python3 training.py
```



Running the command above will first perform the pre-training on the piece-wise metric, and then continue training to enforce the Einstein conditions and the boundary conditions once the pre-training loss is below a certain threshold (0.03 by default).  After the pre-training phase, the training phase will run until manually stopped. The best models are constantly updated and saved in the folder `saved-models`, and the notebook `analysis-notebook.ipynb`can be used to evaluate the quality of the Riemannian metric obtained. The training can also be resumed. 

The various option can be accessed as `python3 training.py --help`.

**This code is not optimized for performance!** 
The training and pre-training phase can last several hours, depending on your machine. By default the code runs on CPUs even when GPUs are available, this be changed by using the flag `--cpu false`. 

For convenience this repository includes pre-trained and trained networks in the folder `saved-models`. They can be analyzed using the notebook `analysis-notebook.ipynb`.


## Citation
If you find this code useful, please cite

```
@article{DeLuca:2024njg,
    author = "De Luca, G. Bruno",
    title = "{Machine Learning Gravity Compactifications on Negatively Curved Manifolds}",
    eprint = "2501.00093",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    month = "12",
    year = "2024"
}
```