[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" />
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
<img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" />

# Digital_segmentation_BRCA_547_Capstone

## Project Objective
The project is intended to segment immune cells on the stained images. And we have already bulit an automaic pipeline that 
can extarct every patches from the Whole slide Image(WSI) of cancer tissue and using K-means clustering algorithm to get the 
overlayed immune cells.

## Use Cases
Generate clusters from an individual patch determine every component separately.</br>
Overlay a single, multiple or all clusters at once on a patch depending upon the analysis that clinician needs to perform.</br>
Automated process of an input WSI and multiple output overlayed patches.</br>
Automating components of analysis which are constant and enabling enough interventional capability for the user.</br>
Ability to classify different components within a particular subcluster of interest (if any exists)</br>

## Methodology
<img src=https://github.com/ViditShah98/Digital_segmentation_BRCA_547_Capstone/blob/main/Picture2.png />

## Installation
In order to use the code, you can create a virtual environment on a Windows system as follows:
Note: If you are using Unix or a conda environment, the steps might change accordingly.

**Step 1:** Clone the repository to your computer.

**Step 2:** Open the command prompt/shell and type `pip install virtualenv`

**Step 3:** In the command prompt/shell type `virtualenv environment_cluster`

**Step 4:** Create the virtual environment by typing `virtualenv environment_cluster`

**Step 5:** Activate the virtual environment by typing `.\environment_cluster\Scripts\activate`

**Step 6:** Enter the path to the repository on terminal `cd ./././Digital_segmentation_BRCA_547_Capstone`

**Step 7:** Install dependencies by typing `pip install -r requirements.txt`

## Example
<img src=https://github.com/ViditShah98/Digital_segmentation_BRCA_547_Capstone/blob/main/Picture1.png />

1. Extracted H&E-stained patch.

2. Digitally clustered patch.

3. Overlay of the digitally clustered patch.

4. Digital map of the segmented immune cells

5. Overlay of the segmented patch.

## Repo Structure
```
Digital_segmentation_BRCA_547_Capstone
-----
Digital_segmentation_BRCA_547_Capstone/
|-doc/
| |-example/paper/
| | |-_init_.py
| |-Use_cases_and_design_components.docx
| |-Use_cases_and_design_components.pdf
| |-ChemE_547_Final_Poster.pdf
| |-Pitch.docx
|-tests/
| |-_init_.py  
|-cluster.py
|-requirements.txt
|-test_cluster.py
LICENSE
Picture1.png
Picture2.png
README.md
```
## Ongoing and Future Work
ONGOING: In order to improve the accuracy and performance of the model even further, we are currently applying textural post 
processing techniques.
 
FUTURE WORK: Implement supervised learning for further segmentation of the immune map.


## Timeline
```mermaid
gantt
 title Timeline
 section Done
   Meet with sponsor     : 2022-04-01, 7d
   Generate merged clusters and overlay images    : 2022-04-12, 7d
   Make Gantt chart and update on README.md    : 2022-04-12, 7d
   Using K-Means to build a pipeline:    2022-04-08, 4d
   Extract patches from 17 whole slide images:    2022-04-19, 7d
   Write a python method that can automatically extract patches from WSI:    2022-04-19, 7d
   Optimize cluster number and weight in python:    2022-04-19, 7d
   Using different patches to test the code and analyze the results:  2022-04-26, 7d
   Using patches that have good and bad performance to fit the model and get the segmented immune cells: 2022-05-03, 7d
   Choose patches that can fit most of the testing patches to make super-patch: 2022-05-10, 2d
   Using super-patch to train the model and get test results: 2022-05-12, 5d
   Compare the results from super-patch and from previous model: 2022-05-12, 5d
   Extract new patches from WSI: 2022-05-12, 1d
   Using new patches to train the model and get best performance patches: 2022-05-12, 2d
   Using new super-patch to fit the model and get test results: 2022-05-14, 5d
   Analyze the results and make a future plan: 2022-05-19, 5d
   Make the poster: 2022-05-26, 5d
   Finalize the repo and prepare the final presentation: 2022-06-01, 5d
 
```
## Group Members: Zilun Cai, Yanyao Han, Ruofan Liu, Vidit Shah.
