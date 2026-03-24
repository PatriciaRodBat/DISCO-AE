# MASTER THESIS of Patricia Rodríguez Batista. Based on the DISCO-AE: Autoencoder for Diverse 3D Shape Collections (DISCO).

This file intends to be explanatory of the Master Thesis repository files, that can be found at [DISCO-AE branch](https://github.com/PatriciaRodBat/DISCO-AE.git) and physically on a CD attached to each physical copy of the master thesis.

## Packages required 
**(better seen on requirements_PatRodBat.txt, an updated version)**
- pytorch (Tested with Pytorch 1.13 and CUDA 11.6)
- tqdm
- scikit-learn
- scipy
- igl 
- matplotlib
- plotly
- meshplot
- robust-laplacian 
- potpourri3d 
- trimesh

**For experiments reproduction it is indispensable that the original DISCO-AE repository is taken into account, together with its original README file. Some of the main instructions follow.**

## Load Data
Download data and p2p maps: 

   ```sh
    ./00_get_data.sh gallop
    ./00_get_data.sh FAUST
    ./00_get_data.sh TRUCK
   ```

## Get Point-to-Point Maps 
To train the first stage in our pipeline, and extract the point-to-point maps, run the following command:

   ```sh
   cd stage1
   python 01_train_FM.py --config faust
   # once the training ends, run the following command to extract the point-to-point maps
   # weights are saved in the same folder as the data under the name "saved_models_DatasetName"
   python get_p2p.py --config faust --weights path/to/weights
   ```

To speed up the p2p map extraction calculate only the maps necessary to define the FMN for the shape collection. 
We recommend calculating p2p maps from every shape to the corresponding template shape and connecting the template shapes to each other.
To reduce the runtime even further, reduce the number of ZoomOut interations.
For best results, we recommend rotating the shapes properly (facing the same direction), and for the car components, normalizing using L1 distance and mean-centering.

## DISCO-autoencoder

Supervised DISCO-AE for GALLOP and FAUST dataset. The config file (faust-extra) trains the "unknown poses" experiment setup. The config file (faust-inter) trains the "unknown individuals" experiment setup. 
The paper explains the experiment setups in detail.

   ```sh
    python 02_train_network.py --config gallop
    python 02_train_network.py --config faust-extra
    python 02_train_network.py --config faust-inter
    python 02_train_network.py --config TRUCK_pall 
   ```

Unsupervised DISCO-AE for selected GALLOP and FAUST setups.

   ```sh
    python 02_train_network.py --config horse-unsup --loss_rec 10
    python 02_train_network.py --config faust-unsup-inter --loss_rec 10
   ```


| **Layer**          | **Output Shape**   | **Trainable** |
|--------------------|--------------------|---------------|
| Input              | (n, 3)             |               |
| DiffusionNet       | (n, nfeature)      | X             |
| ProjToLimitShape   | **(nB, nfeature)** |               |
| ProjToTemplateMesh | (m, nfeature)      |               |
| Append TemplateShape 3D-Coord | (m, nfeature+3)      |               |
| DiffusionNet       | (m, 3)             | X             |

## Citation of the original DISCO-AE:
@InProceedings{Hahner2024,
    author    = {Hahner, Sara and Attaiki, Souhaib and Garcke, Jochen and Ovsjanikov, Maks},
    title     = {Unsupervised Representation Learning for Diverse Deformable Shape Collections},
    booktitle = {Proceedings of the International Conference on 3D Vision (3DV 2024)},
    year      = {2024},
}

## ORIGINAL IMPLEMENTATIONS
Some files were modified three times in order to fulfill the three ideas to develop of the master thesis: PCA-based second reduction algorithm, Selection of a particular set of eigenvectors and MLP-based fusion for combination of shape and template features.

**The naming pattern of these variations goes as follows: xxx_PCA.py, xxx_eigvct.py, xxx_TF.py**

In the particular case of the model architecture, **model.py**, there are two versions attached: **model_TF.py** and **model_TF2.py**, corresponding to the first and second MLP-fusion approaches, further explained on the thesis.

Some files do not have one version per approach, but were still slightly modified. These were named **xxx_PatRodBat.py** for distintion purposes. In particular:
-*geometry.py*, originally within the diffusion_net folder, was modified adding a sanity line so that a division was not over an extremely small number, since it was bugging the GALLOP dataset MLP-fusion implementation.
-*dataloading.py* and *utils.py*, originally within the utils folder.

Some other files were directly added to the repository, instead of modified.
-To plot and compare surface meshes and their errors regarding different approaches: **PLOTS3D_true.ipynb** and **PLOTS3D_compare.ipynb**, where the first one solely shows ground truth meshes.
-To run experiments, **02_train_net_jobscript.sh**.
-To faster run experiments with differences in seed, approach, etc, these *.yalm* files were added to the config folder, although their functioning changes nothing really, resulting in files of very low interest: **gallop_pca_seed1/2/3**, **gallop_mlp_seed1/2/3**, **gallop_model_seed1/2/3**.
-Together with these files, **run_6.sh** was designed to be able to run them simultaneously.
