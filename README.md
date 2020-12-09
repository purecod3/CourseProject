# Latent Dirichlet Allocation
This project is based on the paper written by D. Blei, A. Ng, and M. Jordan - _Latent Dirichlet Allocation_. https://dl.acm.org/doi/pdf/10.5555/944919.944937. Latent Dirichlet Allocation estimates topic disributions and topic word distributions in a generative model that can be used to infer topic distributions and word topic assignments for new documents. We use this modeling capability to evaluate the application of LDA on classification of documents using a significantly reduced number of features as compared to a bag of words based classification method.

## Team Members
1. Ed Pureza, epureza2 (captain)
2. Dan Qian, dansiq2
3. Joe Everton, everton2

## Files
|Deliverable|File|Description|
|----------|----|-----------|
|[Project Proposal](https://github.com/purecod3/CourseProject/blob/main/Project%20Proposal_%20Reproduce%20Latent%20aspect%20rating%20analysis%20without%20aspect%20keyword%20supervision.pdf)|`Project Proposal_Reproduce Latent aspect rating analysis without aspect keyword supervision.pdf`|Original project proposal submitted on October 24, 2020|
|[Progress Report](https://github.com/purecod3/CourseProject/blob/main/ProgressReport.pdf)|`ProgressReport.pdf`|Progress report with accomplishments, challenges, and remaining planned activities as of November 29, 2020|
|__to-do__ [Project Documentation](https://github.com/purecod3/CourseProject/blob/main/ProjectDocumentation.pdf)|`ProjectDocumentation.pdf`|Project documentation submitted December 8, 2020|
|[Project Video Walk-through](https://mediaspace.illinois.edu/media/t/1_jbzbbspv)|https://mediaspace.illinois.edu/media/t/1_jbzbbspv|Video presentation of project|
|[Project Tutorial](https://github.com/purecod3/CourseProject/blob/main/ProjectTutorial.pdf)|`ProjectTutorial.pdf`|Project tutorial for reproducing experiments (also outlined below)|
|[LDA without Smoothing](https://github.com/purecod3/CourseProject/blob/main/lda_var_inf_without_smoothing.py)|`lda_var_inf_without_smoothing.py`|Code for running LDA using variational inference and gensim-based alpha update method|
|[LDA without Smooting v2](https://github.com/purecod3/CourseProject/blob/main/lda_var_inf_without_smoothing_v2.py)|`lda_var_inf_without_smoothing_v2.py`|Code for running LDA using variational inference. Use if Python environment setup issues are encountered.|
|[LDA with Collapsed Gibbs Sampling](https://github.com/purecod3/CourseProject/blob/main/lda_gibbs_sampling.py)|`lda_gibbs_sampling.py`|LDA implementation using Collapsed Gibbs Sampling|
|[Original LDA Code with Variational Inference](https://github.com/purecod3/CourseProject/blob/main/lda_var_inf.py)|`lda_var_inf.py`|First attempt for implement LDA with variational inference method|
|[Fake News Dataset](https://github.com/purecod3/CourseProject/blob/main/FA-KES-Dataset.csv)|`FA-KES-Dataset.csv`|Input dataset with news articles classified as fake news or not fake news|
|[Spam Dataset](https://github.com/purecod3/CourseProject/blob/main/spam.csv)|`spam.csv`|Input dataset with news articles classified as spam or ham (not spam)|

## How to Use
### Progamming Language and Packages
- Python 3.x
- Packages: pandas, numpy, scipy, sklearn, math, re, random, time

### Executing Code
Fork or download Github repo.  

Open in IDE and run file(s) or use command prompt (e.g., `python lda_var_inf_without_smoothing.py`). Start with `lda_var_inf_without_smoothing.py`. If unexpected results are encountered, try `lda_var_inf_without_smoothing_v2.py`. Optionally, you can also try the other variations with `lda_gibbs_sampling.py` and `lda_var_inf.py`.

To use a different input dataset, your file will need text and classification columns. Modify the source file (`input_path`) and column settings (`text_column`, `label_column`) in the `load_csv` function call.  

```python
(vocabulary_size,
     training_term_doc_matrix,
     training_labels,
     testing_term_doc_matrix,
     testing_labels,
     vocabulary) = load_csv(input_path = 'FA-KES-Dataset.csv',
                            test_set_size=100,
                            training_set_size=200,
                            num_stop_words=50,
                            min_word_freq=5,
                            text_column='article_content',
                            label_column='labels',
                            label_dict = {'1': 1, '0': 0})
```

`lda_var_inf_without_smoothing_v2.py` has both datasets (fake news and spam) coded. Comment/uncomment to switch between datasets.  

### Setting Parameters
Set the following parameters to tune the model:  
- `num_topics`: number of topics to model

```python
lda.train(num_topics=10, term_doc_matrix=training_term_doc_matrix, iterations=20, e_iterations=10, e_epsilon=0.1, initial_training_set_size=50, initial_training_iterations=20)
```
See [video walk-thru](https://mediaspace.illinois.edu/media/t/1_jbzbbspv) for additional information.
