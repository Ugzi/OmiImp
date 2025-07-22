# OmiImp
A omics-to-omics imputation framework based on generative adversarial networks (GANs).
The integration of multi-omics data has emerged as a powerful approach for elucidating interactions across different biological levels, thereby improving clinical outcome prediction and advancing precision medicine. However, technical limitations and high costs of sequencing technologies often result in sparse multi-omics datasets, where only a subset of samples contains complete omics profiles- a challenge known as the “block missing”. To address this issue, we present OmiImp, a novel computational framework that leverages a pre-trained generative adversarial network for accurate omics-omics data imputation. Through comprehensive benchmarking on independent datasets, we demonstrate that OmiImp maintains stable performance across different missing rates. Importantly, OmiImp achieves precise data imputation without introducing systematic biases, and that the imputed data retains utility in diverse prognostic analyses. This advancement enables more robust multi-omics studies even with incomplete datasets.
<img width="1692" height="820" alt="image" src="https://github.com/user-attachments/assets/e3279054-44d4-4abd-a215-00872a48156a" />
OmiImp provides four python files:
* data_preprocess.py：the raw data provided by ROSMAP is processed for both the source omics and target omics to serve as model inputs.
* metric.py：the evaluation metrics are used to assess the performance of OmiImp.
* model.py：the architecture of the OmiImp model is defined in this file.
* train.py: the model undergoes training and testing, followed by performance evaluation using designated metrics.
  
