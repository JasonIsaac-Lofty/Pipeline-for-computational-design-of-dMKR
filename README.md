# Overall design pipeline of dMKR
Yunhao Li, PhD student (Department of Chemistry, Zhejiang University)<br>

e_mail: 12237050@zju.edu.cn

#### This pipeline and design protocols are adapted from three articles below, if you want to use this pipeline, please also cite:
*1. Yeh AH, Norn C, Kipnis Y, Tischer D, Pellock SJ, Evans D, et al. De novo design of luciferases using deep learning. Nature 2023, 614(7949): 774-780.*

*2. Krishna R, Wang J, Ahern W, Sturmfels P, Venkatesh P, Kalvet I, et al. Generalized biomolecular modeling and design with RoseTTAFold All-Atom. Science 2024, 384(6693): eadl2528.*

*3. An L, Said M, Tran L, Majumder S, Goreshnik I, Lee GR, et al. Binding and sensing diverse small molecules using shape-complementary pseudocycles. Science 2024, 385(6706): 276-282.*

#### The design process can be divided into five parts:
    1) Ligand conformation search;

    2) De novo protein scaffolds generation;

    3) Ligand docking and metal adding;

    4) Multiple rounds of sequence design;
    
    5) Final structure validation.

#### Remember to replace all file paths to fit your computer environment.
#### All contigs and flags files used in this pipeline are stored at corresponding step folders.

## Softwares
Please install the following software, see the respective **Github** or **Official** pages for installation instructions

### 1. RFDiffusionAA: 
https://github.com/baker-laboratory/rf_diffusion_all_atom

### 2. ProteinMPNN & LigandMPNN
https://github.com/dauparas/ProteinMPNN<br>
https://github.com/dauparas/LigandMPNN

### 3. AlphaFold2
https://github.com/google-deepmind/alphafold

### 4. RIFDock
https://github.com/rifdock/rifdock

### 5. Rosetta & pyRosetta
https://rosettacommons.org/software/download/<br>
https://github.com/RosettaCommons/rosetta

### 6. Metal3D
https://github.com/lcbc-epfl/metal-site-prediction

### 7. ColabFold/LocalColabFold
https://github.com/sokrypton/ColabFold<br>
https://github.com/YoshitakaMo/localcolabfold

### 8. Conformer-Rotamer Ensemble Sampling Tool(CREST) v2.12
https://crest-lab.github.io/crest-docs/

### Addition scripts source
https://github.com/ikalvet/heme_binder_diffusion
