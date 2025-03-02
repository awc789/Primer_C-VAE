# Primer C-VAE: An interpretable deep learning primer design method to detect emerging virus variants
CORMSIS External Summer Project
（Master Graduation Project）

Tutor: 

Dr. Alain Zemkoho (University of Southampton, UK)

Dr. Emmanuel Kagning-Tsinda (Tohoku University, Japan)

2021-06 ---– 2022-06

Updated on 2025-03


## SARS-CoV-2 virus Gene Sequence Data

## Dataset:
- GISAID (https://www.gisaid.org/)
- NCBI (https://www.ncbi.nlm.nih.gov/)
- ~~NGDC (https://big.ac.cn/ncov/?lang=en)~~

The detail of the SARS-CoV-2 sequence data used in this project can be seen in the following file:

**`SARS_CoV_2_Gene_sequence_info.md`**

#### Overall
- ##### For SARS-CoV-2 virus (Homo Sapiens Host):
    - The gene sequence data files downloaded from the **GISAID** database. If the classification of the variant virus has been completed, please move them to **`./Dataset/Variant_virus`** with correct variant types. Or please using the [Pangolin](https://cov-lineages.org/resources/pangolin.html) to make sure which type of variants that the gene sequence belongs to and then move to **`./Dataset/Variant_virus`** with correct variant types.
    - The gene sequence data files downloaded from the **NCBI** database need to use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to make sure which type of variants that the gene sequence belongs to and then move to **`./Dataset/Variant_virus`** with correct variant types.
- ##### For SARS-CoV-2 virus (Non-Homo Sapiens Host):  
    -  Please move files to **`./Dataset/other_virus/other_virus_seq`**
- ##### For other taxa (Homo Sapiens Host):
    -  Please move files to **`./Dataset/other_virus/other_virus_seq`**

<font color='black'><td><tr><table>
ATTENTION: 

- The GISAID Dataset requires registration and login to download the gene sequnence data.
- The NCBI Dataset will not provide classification of virus variants. If needs to download the data from NCBI Dataset, it will need to use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to implement the dynamic nomenclature of SARS-CoV-2 lineages.
- The NGDC Dataset is alos a popular dataset for SARS-CoV-2 virus. But at the moment working on this project the network always shows that it cannot download the data from the NGDC Database.
</td><tr></table></font>

![](https://github.com/cov-lineages/pangolin/raw/master/docs/logo.png)


## E.coli and S. flexneri Gene Sequence Data

## Dataset:
- NCBI (https://www.ncbi.nlm.nih.gov/)


## Overall Pipeline and Primer C-VAE architecture

![Overall Pipeline](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Overall_Pipeline.png?raw=true)

The Primer C-VAE methodology comprises four interconnected computational stages. \textbf{Stage I (Data Acquisition and Pre-processing)} encompasses sequence acquisition from genomic repositories, systematic taxonomic annotation, and strategic data curation to establish high-quality training datasets for downstream analysis. \textbf{Stage II (Forward Primer Design)} implements our trained convolutional variational autoencoder architecture to generate initial primer candidates, followed by frequency distribution analysis and thermodynamic property assessment to identify optimal forward primers with maximal discriminative capacity. \textbf{Stage III (Reverse Primer Design)} analyzes the downstream genomic regions adjacent to selected forward primer binding sites across target organism sequences, applying the C-VAE model in a second iteration to generate complementary reverse primer candidates, which undergo similar frequency and thermodynamic suitability filtering protocols. \textbf{Stage IV (In-silico PCR and Primer-BLAST Validation)} integrates selected forward and reverse primers into functional amplification pairs, evaluates their combinatorial properties including amplicon size and primer-dimer potential, and validates specificity through hierarchical assessment via BLAST sequence alignment followed by in-silico PCR amplification simulation.

![Primer C-VAE architecture](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Primer_C-VAE_architecture.png?raw=true)

Primer C-VAE architecture implements a specialized convolutional encoder framework for discriminating genomic features between target organisms and their variant populations. This deep learning system integrates three critical functional modules: (1) a multi-layer convolutional encoder that systematically extracts hierarchical sequence features from raw genomic data, (2) a variational representation space where latent vectors $z$ are stochastically sampled via the reparameterization technique utilizing the learned distributional parameters $\mu$ and $\log\sigma^2$, and (3) a bifurcated computational pathway featuring both a classifier component for precise sequence categorization and a reconstruction decoder for generating sequence outputs. The architecture's training protocol optimizes these components simultaneously to maximize feature discrimination while preserving biological sequence integrity.


## Forward Primer Design

##### Flowchart of the project in this part：

![Forward Primer Design](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Flowchart_Forward.jpg?raw=true)

After training the CNN model for Forward Primer Design, you can use the **`other_code/confusion_matrix.py`** file to generate a confusion matrix and plot the images to determine the accuracy of the model's classification results.

![Confusion_Matrix](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Confusion_Matrix.png?raw=true)


## Reverse Primer Design

##### Flowchart of the project in this part：

![Reverse Primer Design](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Flowchart_Reverse.jpg?raw=true)

**There are two ways to run the codes:**
        
        1. - Recommended: Using Amazon Web Services
                - Upload data to AWS S3
                - Creat a SageMaker Notebook
                - Clone this project by using following code:
                    !git clone https://github.com/awc789/Machine-Learning-Based-Primer-Identification-for-the-Detection-of-SARS-CoV-2-Emerging-Variants.git
                - Install the required packages:
                    !pip install -r requirements.txt
                - main.ipynb
        
        2. - Running on local devices
                - Download the data from GISAID/NCBI and this project
                - Change the function 'S3_to_sageMaker' which is using for reading data
                - main.ipynb



<font color='black'><td><tr><table>
ATTENTION: 

-  Integrated DNA Technologies -- IDT: (https://www.idtdna.com/)

- It is **NOT** recommended to use the **`other_code/online_validation.py`** file to complete the In-Silico PCR at the [UCSC In-Silico PCR](https://genome.ucsc.edu/cgi-bin/hgPcr) website.

- Please use the [FastPCR](https://primerdigital.com/fastpcr.html) or [Unipro UGENE](http://ugene.net/) software for the In-Silico PCR to check the availability of the candidate primers.
</td><tr></table></font>

------
## Acknowledgement:
We gratefully acknowledge the following Authors from the Originating laboratories responsible for obtaining the specimens and the Submitting laboratories where genetic sequence data were generated and shared via the GISAID Initiative, on which this research is based.

    EPI_SET ID:	EPI_SET_20220628va
    DOI:	https://doi.org/10.55876/gis8.220628va



## Reference:
This project is based on the work of **Alejandro Lopez‑Rincon**, **Alberto Tonda** and **Lucero Mendoza‑Maldonado**: [Classifcation and specifc primer design for accurate detection of SARS‑CoV‑2 using deep learning](https://www.nature.com/articles/s41598-020-80363-5.pdf)
