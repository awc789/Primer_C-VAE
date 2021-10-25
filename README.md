# Machine-Learning-Based-Primer-Identification-for-the-Detection-of-SARS-CoV-2-Emerging Variants
CORMSIS External Summer Project
（Master Graduation Project）

Tutor: 
Dr. Alain Zemkoho (University of Southampton, UK)
Dr Emmanuel Kagning-Tsinda (Tohoku University, Japan)

2021-06 ---– Now


SARS-CoV-2 virus Gene Sequence Data
==================

## Dataset:
- GISAID (https://www.gisaid.org/)
- NCBI (https://www.ncbi.nlm.nih.gov/)
- ~~NGDC (https://big.ac.cn/ncov/?lang=en)~~


The detail of the data used in this project can be seen in the following file:

**`Gene_sequence_info.md`**

#### Overall 
- ##### For SARS-CoV-2 virus (Homo Sapiens Host):
    - The gene sequence data files downloaded from the **GISAID** database. If the classification of the variant virus has been completed, please move them to **`./code/Dataset/Variant_virus`** with correct variant types. Or please using the [Pangolin](https://cov-lineages.org/resources/pangolin.html) to make sure which type of variants that the gene sequence belongs to and then move to **`./code/Dataset/Variant_virus`** with correct variant types.
    - The gene sequence data files downloaded from the **NCBI** database need to use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to make sure which type of variants that the gene sequence belongs to and then move to **`./code/Dataset/Variant_virus`** with correct variant types.
- ##### For SARS-CoV-2 virus (Non-Homo Sapiens Host):  
    -  Please move files to **`./code/Dataset/other_virus/other_virus_seq`**
- ##### For other taxa (Homo Sapiens Host):
    -  Please move files to **`./code/Dataset/other_virus/other_virus_seq`**

<font color='black'><td><tr><table>
ATTENTION: 

- The GISAID Dataset requires registration and login to download the gene sequnence data.
- The NCBI Dataset will not provide classification of virus variants. If needs to download the data from NCBI Dataset, it will need to use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to implement the dynamic nomenclature of SARS-CoV-2 lineages.
- The NGDC Dataset is alos a popular dataset for SARS-CoV-2 virus. But at the moment working on this project the network always shows that it cannot download the data from the NGDC Database.
</td><tr></table></font>

![](https://github.com/cov-lineages/pangolin/raw/master/docs/logo.png)


v3_code ---- Forward Primer Design
==================

##### Flowchart of the model：
![截屏2021-10-25 19.22.32](https://i.imgur.com/TFAl4ms.png)

**There are two ways to run the codes:**
        
        1. - Step by Step with each file
            - v3_get_data.py
            - v3_filter.py (Train the CNN model)
            - v3_feature.py
            - v3_appearance.py
            - v3_other_appearance.py
        
        2. - One file only
            - v3_main.py

After training the CNN model, you can use the **`v3_confusion_matrix.py`** file to generate a confusion matrix and plot the images to determine the accuracy of the model's classification results.

![Figure_1](https://i.imgur.com/lczZb2h.png)


<font color='black'><td><tr><table>
ATTENTION: 

- It is **NOT** recommended to use the **`v3_online_validation.py`** file to complete the In-Silico PCR at the [UCSC In-Silico PCR](https://genome.ucsc.edu/cgi-bin/hgPcr) website.

- Please use the [FastPCR](https://primerdigital.com/fastpcr.html) software for the In-Silico PCR.
</td><tr></table></font>






## Reference:
This project is based on the work of **Alejandro Lopez‑Rincon**, **Alberto Tonda** and **Lucero Mendoza‑Maldonado**: [Classifcation and specifc primer design for accurate detection of SARS‑CoV‑2 using deep learning](https://www.nature.com/articles/s41598-020-80363-5.pdf)