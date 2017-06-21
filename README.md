# Description
This code release contains an implementation of two relevance decomposition methods, Layer-wise Relevance Propagation (LRP) and Sensitivity Analysis (SA), for a bidirectional LSTM, as described in the paper [Explaining Recurrent Neural Network Predictions in Sentiment Analysis by L. Arras, G. Montavon, K.-R. Müller and W. Samek, 2017](https://arxiv.org/)



## Dependencies

Python + Numpy + Matplotlib (or simply install Anaconda)



## Usage

The folder model/ contains a word-based bidirectional LSTM model, that was trained for five-class sentiment prediction of phrases and sentences on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) dataset, as released by the authors in [Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

The folder data/ contains the test set sentences of the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html), preprocessed by lowercasing, as was done in [Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

The notebook run_example.nb provides a usage example of the code, its performs LRP and SA on a test sentence.



## Acknowledgments

[Visualizing and Understanding Neural Models in NLP by J. Li, X. Chen, E. Hovy and D. Jurafsky, 2016](https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP)

[Long Short Term Memory Units repo by W. Zaremba](https://github.com/wojzaremba/lstm)

[Stanford Sentiment Treebank dataset by R. Socher et al., 2013](https://nlp.stanford.edu/sentiment/index.html)



## Citation

```latex
@article{arras2017,
    title   = {Explaining Recurrent Neural Network Predictions in Sentiment Analysis},
    author  = {Leila Arras and Gr{\'e}goire Montavon and Klaus-Robert M{\"u}ller and Wojciech Samek},
    journal = {arXiv},
    year    = {2017}
}
