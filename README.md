
# Description

This code release contains an implementation of two relevance decomposition methods, Layer-wise Relevance Propagation (LRP) and Sensitivity Analysis (SA), for a bidirectional LSTM, as described in the paper [Explaining Recurrent Neural Network Predictions in Sentiment Analysis by L. Arras, G. Montavon, K.-R. Müller and W. Samek, 2017](http://aclweb.org/anthology/W/W17/W17-5221.pdf)

Note that our implementation is generic and can be easily extended to unidirectional LSTMs, or to other applications than NLP.



## Dependencies

Python>=3.5 + Numpy + Matplotlib, or alternatively simply install Anaconda.

Using Anaconda you can e.g. create a Python 3.6 environment: conda create -n py36 python=3.6 anaconda

Then activate it with: source activate py36

Before being able to use the code, you might need to run in the terminal: export PYTHONPATH=$PYTHONPATH:$pwd



## Usage

The folder model/ contains a word-based bidirectional LSTM model, that was trained for five-class sentiment prediction of phrases and sentences on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) dataset, as released by the authors in [Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

The folder data/ contains the test set sentences of the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html), preprocessed by lowercasing, as was done in [Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

The notebook run_example.ipynb provides a usage example of the code, its performs LRP and SA on a test sentence.



## Acknowledgments

[Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016 code](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

[Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016 paper](http://aclweb.org/anthology/N/N16/N16-1082.pdf)

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
