
# Description

This repository provides a reference implementation of Layer-wise Relevance Propagation (LRP) for an LSTM, as proposed in the paper [Explaining Recurrent Neural Network Predictions in Sentiment Analysis, by L. Arras, G. Montavon, K.-R. Müller and W. Samek, WASSA@EMNLP 2017](http://aclweb.org/anthology/W/W17/W17-5221.pdf) [[arXiv preprint](https://arxiv.org/abs/1706.07206)].

Additionally, it provides an implementation of Sensitivity Analysis (SA), i.e. of simple gradient-based relevance.

Note that our implementation is generic and can be easily extended to unidirectional LSTMs, or to other application domains than NLP. A few hints on how to adapt the code to your needs can be found [here](https://github.com/ArrasL/LRP_for_LSTM/blob/master/DOC.md).



## Dependencies

Python>=3.5 + Numpy + Matplotlib, or alternatively simply install Anaconda.

Using Anaconda you can e.g. create a Python 3.6 environment: conda create -n py36 python=3.6 anaconda

Then activate it with: source activate py36

Before being able to use the code, you might need to run in the terminal: export PYTHONPATH=$PYTHONPATH:$pwd



## Usage

The folder model/ contains a word-based bidirectional LSTM model, that was trained for five-class sentiment prediction of phrases and sentences on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) dataset, as released by the authors in [Visualizing and Understanding Neural Models in NLP, by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

The folder data/ contains the test set sentences of the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html), preprocessed by lowercasing, as was done in [Visualizing and Understanding Neural Models in NLP, by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

The notebook run_example.ipynb provides a usage example of the code, its performs LRP and SA on a test sentence.



## Follow-up work

Here are some follow-up works using LRP for recurrent neural networks (non-exhaustive list): 

[Evaluating neural network explanation methods using hybrid documents and morphosyntactic agreement, by N. Poerner, B. Roth and H. Schütze, ACL 2018](http://aclweb.org/anthology/P18-1032)

[Explaining Therapy Predictions with Layer-wise Relevance Propagation in Neural Networks, by Y. Yang, V. Tresp, M. Wunderle and P.A. Fasching, IEEE ICHI 2018](https://doi.org/10.1109/ICHI.2018.00025)

[Interpretable LSTMs for Whole-Brain Neuroimaging Analyses, by A.W. Thomas, H.R. Heekeren, K.-R. Müller and W. Samek, arXiv:1810.09945 2018](https://arxiv.org/pdf/1810.09945.pdf)



## Acknowledgments

[Visualizing and Understanding Neural Models in NLP, by J. Li, X. Chen, E. Hovy and D. Jurafsky, code](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

[Visualizing and Understanding Neural Models in NLP, by J. Li, X. Chen, E. Hovy and D. Jurafsky, NAACL 2016](http://aclweb.org/anthology/N/N16/N16-1082.pdf)

[Long Short Term Memory Units, repo by W. Zaremba](https://github.com/wojzaremba/lstm)

[Stanford Sentiment Treebank, dataset by R. Socher et al., 2013](https://nlp.stanford.edu/sentiment/index.html)



## Citation

    @INPROCEEDINGS{arras2017,
        title     = {Explaining Recurrent Neural Network Predictions in Sentiment Analysis},
        author    = {Leila Arras and Gr{\'e}goire Montavon and Klaus-Robert M{\"u}ller and Wojciech Samek},
        booktitle = {Proceedings of the EMNLP 2017 Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
        year      = {2017},
        pages     = {159-168},
        publisher = {Association for Computational Linguistics},
        url       = {http://aclweb.org/anthology/W/W17/W17-5221.pdf}
    }



## More information

For further research and projects involving LRP, visit [heatmapping.org](http://heatmapping.org)
