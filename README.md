# Machine-Learning-Based-Primer-Identification-for-the-Detection-of-SARS-CoV-2-Emerging Variants
CORMSIS External Summer Project
（Master Graduation Project）

Tutor: 

Dr. Alain Zemkoho (University of Southampton, UK)

Dr. Emmanuel Kagning-Tsinda (Tohoku University, Japan)

2021-06 ---– 2022-06


## SARS-CoV-2 virus Gene Sequence Data

## Dataset:
- GISAID (https://www.gisaid.org/)
- NCBI (https://www.ncbi.nlm.nih.gov/)
- ~~NGDC (https://big.ac.cn/ncov/?lang=en)~~


The detail of the data used in this project can be seen in the following file:

**`Gene_sequence_info.md`**

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


## Forward Primer Design

##### Flowchart of the project in this part：

![截屏2022-04-25 23.32.16](https://i.imgur.com/lRsu4Rm.png)

After training the CNN model for Forward Primer Design, you can use the **`other_code/confusion_matrix.py`** file to generate a confusion matrix and plot the images to determine the accuracy of the model's classification results.

![Confusion_Matrix](https://i.imgur.com/GYMqvr6.png)


## Reverse Primer Design

##### Flowchart of the project in this part：

![截屏2022-04-25 23.34.14](https://i.imgur.com/sdr4BLN.png)

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

## Reference:
This project is based on the work of **Alejandro Lopez‑Rincon**, **Alberto Tonda** and **Lucero Mendoza‑Maldonado**: [Classifcation and specifc primer design for accurate detection of SARS‑CoV‑2 using deep learning](https://www.nature.com/articles/s41598-020-80363-5.pdf)
