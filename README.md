# Training and Evaluation Material

This repo includes all datasets and code files to train and evaluate the models used in this project.

## Training

### [LOTClass](https://github.com/yumeng5/LOTClass)

Source code available in folder ```LOTClass```. To train the model on W2 dataset, all parameters can be set in file: [```w2_full_multi_eval.sh```](https://github.com/frmati/TM_Material/blob/main/LOTClass/w2_full_multi_eval.sh).  Trains the model on W2, and evaluates the model on all parallel test datasets in German, French, Italian, Portuguese, English, and the automatic translated testset. This version uses the last stored checkpoint in the dataset folder to continue training at any step. 

### [CTM](https://github.com/MilaNLProc/contextualized-topic-models)
Source code available in folder ```CTM```. To train the model on W2 dataset, use this file's main method to define training parameters: [```baselines_ctm_W2_V2.py```](https://github.com/frmati/TM_Material/blob/main/CTM/baselines_ctm_W2_V2.py) (Lines 231-244). 


## Evaluation
When a model was trained, all data for evaluation needed is stored in the same dataset folder that was specified before training. Scripts can be found in folder ```Evaluation```.

### Multilingual Evaluation Metrics
[```evaluate_files.py```](https://github.com/frmati/TM_Material/blob/main/Evaluation/evaluate_files.py) calculates all multilingual evaluation metrics. Depending on the dataset and model, settings need to be adapted in the main method (Lines 343-352). 

### Test Predicion Evaluation and Visualization
[```compare_test_results.py```](https://github.com/frmati/TM_Material/blob/main/Evaluation/compare_test_results.py) generates tables of comparison, to evaluate test predicions in more details. Depending on the dataset and model, settings need to be adapted in the main method (Lines 186-193). 


