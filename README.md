
# Description

This repository provides a **reference** implementation of **Layer-wise Relevance Propagation (LRP)** for an LSTM, as initially proposed in the paper [*Explaining Recurrent Neural Network Predictions in Sentiment Analysis, L. Arras, G. Montavon, K.-R. Müller and W. Samek* WASSA@EMNLP'2017](https://doi.org/10.18653/v1/W17-5221) [[arXiv:1706.07206](https://arxiv.org/abs/1706.07206)].

Additionally, it includes an implementation of **Sensitivity Analysis (SA)** and **GradientxInput (GI)**, i.e. of gradient-based relevances.

Note that our implementation is generic and can be easily extended to unidirectional LSTMs, or to other application domains than Natural Language Processing (NLP). 

A few hints on how to apply and extend the code to your needs can be found [here](./DOC.md).



## Dependencies

Python>=3.5 + Numpy + Matplotlib, or alternatively, simply install Anaconda.

Using conda, you can e.g. create a Python 3.6 environment: conda create -n py36 python=3.6 anaconda

Then activate it with: source activate py36

Before being able to use the code, you might need to run in the terminal: export PYTHONPATH=$PYTHONPATH:$pwd



## Usage

The folder model/ contains a word-based bidirectional LSTM model, that was trained for five-class sentiment prediction of phrases and sentences on the [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/index.html) dataset, as released by the authors in *[Visualizing and Understanding Neural Models in NLP, J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)*

The folder data/ contains the test set sentences of the [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/index.html), preprocessed by lowercasing, as was done in *[Visualizing and Understanding Neural Models in NLP, J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)*

The notebook run_example.ipynb provides a usage example of the code, its performs LRP, SA and GI on a test sentence.<br/>
(To correctly render the notebook on GitHub you can copy the notebook's URL to [nbviewer](https://nbviewer.jupyter.org/))



## Follow-up work

Here are some follow-up works using LRP on various recurrent neural network models and tasks (non-exhaustive list): 

[1] [*Evaluating Neural Network Explanation Methods using Hybrid Documents and Morphosyntactic Agreement, N. Poerner, H. Schütze and B. Roth*, ACL 2018](https://www.aclweb.org/anthology/P18-1032) [[arXiv:1801.06422](https://arxiv.org/abs/1801.06422)] [[code](https://github.com/NPoe/neural-nlp-explanation-experiment/tree/master/HybridDocuments/ThirdParty/LRP_and_DeepLIFT)]

[2] [*Explaining Therapy Predictions with Layer-Wise Relevance Propagation in Neural Networks, Y. Yang, V. Tresp, M. Wunderle and P.A. Fasching*, IEEE ICHI 2018](https://doi.org/10.1109/ICHI.2018.00025) [[preprint](http://www.dbs.ifi.lmu.de/~tresp/papers/ICHI2018.pdf)] [[code](https://github.com/Tuyki/TT_RNN/blob/master/MNISTSeq.py)]

[3] [*Analyzing Neuroimaging Data Through Recurrent Deep Learning Models, A.W. Thomas, H.R. Heekeren, K.-R. Müller and W. Samek*, Frontiers in Neuroscience 2019](https://doi.org/10.3389/fnins.2019.01321) [[blog](https://www.notion.so/Analyzing-fMRI-data-with-deep-learning-models-62e0c032d0e244dab1fb077da136b214)]

[4] [*Evaluating Recurrent Neural Network Explanations, L. Arras, A. Osman, K.-R. Müller and W. Samek*, BlackboxNLP@ACL 2019](https://www.aclweb.org/anthology/W19-4813) [[arXiv:1904.11829](https://arxiv.org/abs/1904.11829)] [[oral presentation slides](./misc/Talk_slides.pdf)] 

[5] [*Explaining and Interpreting LSTMs, L. Arras, J. Arjona-Medina, M. Widrich, G. Montavon, M. Gillhofer, K.-R. Müller, S. Hochreiter and W. Samek*, Explainable AI: Interpreting, Explaining and Visualizing Deep Learning, Springer LNCS 11700](https://doi.org/10.1007/978-3-030-28954-6_11) [[arXiv:1909.12114](https://arxiv.org/abs/1909.12114)]



## Acknowledgments

[Visualizing and Understanding Neural Models in NLP, J. Li, X. Chen, E. Hovy and D. Jurafsky, code](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

[Visualizing and Understanding Neural Models in NLP, J. Li, X. Chen, E. Hovy and D. Jurafsky, NAACL 2016](https://doi.org/10.18653/v1/N16-1082)

[Long Short Term Memory Units, repo by W. Zaremba](https://github.com/wojzaremba/lstm)

[Stanford Sentiment Treebank (SST), dataset by R. Socher et al., 2013](https://nlp.stanford.edu/sentiment/index.html)



## Citation

If you find this project useful, please cite the following paper (or one of our follow-up works [4,5]):

    @INPROCEEDINGS{arras2017,
        title     = {{Explaining Recurrent Neural Network Predictions in Sentiment Analysis}},
        author    = {Leila Arras and Gr{\'e}goire Montavon and Klaus-Robert M{\"u}ller and Wojciech Samek},
        booktitle = {Proceedings of the EMNLP 2017 Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
        year      = {2017},
        pages     = {159-168},
        publisher = {Association for Computational Linguistics},
        doi       = {10.18653/v1/W17-5221},
        url       = {https://www.aclweb.org/anthology/W17-5221}
    }



## More information

For further research and other projects involving LRP, you can visit the website [heatmapping.org](http://heatmapping.org)
