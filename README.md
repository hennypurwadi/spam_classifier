## Spam filter

Cloud link: https://hennypurwadi-spam-classifier-spam-classifier-ubc344.streamlit.app/

## Abstract
This project goal is to build a machine learning model to filter and classify messages as spam or 
not spam, developed by using the CRISP-DM (business understanding, data understanding, data 
preparation, modelling, evaluation, and deployment) process, which includes stages of a 
proposal about how to build the project through the folder structure of a Git repository. The project 
objectives are to organize the project structure, analyze the provided data set, pre-process the 
data, vectorize using Tf-Idf, use L1 or L2 regularization, and cross validation to avoid overfitting, 
and then build the machine learning model. This study compares several algorithms, such as 
Decision Tree, Logistic Regression, Naive Bayes, Support Vector Machine, to determine which 
one is the best model for classifying and filtering messages as spam or not. Then the error 
analysis will be performed to understand the weaknesses and limitations of the model and raise 
confidence level from business partner about the model. The results of the analysis will be 
presented with suggestions for future steps in this area, and the graphical user interface will be 
proposed along with a diagram concept plan for the model’s integration into daily work.

![diagram flow](https://github.com/hennypurwadi/spam_classifier/blob/main/spam_filter/docs/images/spam_filter_diagram.jpeg)
## Spam email classification machine learning model requires several steps:
1.First, the model is trained with labelled dataset. Then, save the trained model.
2.Then the stored model predicts spam or not spam when user enters new email.
3.The input email and prediction are added to dataset, to increase model accuracy over time.
4.The model keeps re-trained and re-tested with new data. (Iterative optimization).
5.The machine learning lifecycle on dataset iteration keep looping to maintain accuracy.
6.The model will be incorporated into the service team's workflow to simplify email classification.

## Common spam words
Using the WordCloud Python library, we can investigate the most common words that appear 
in spam categories. Words such as "free”, "won", "win", "awarded", "cash" "prize", "phone",
"urgent", "offer", "now", and "claim" seem often appear.

![diagram flow](https://github.com/hennypurwadi/spam_classifier/blob/main/spam_filter/docs/images/spam_wordcloud.jpeg)

In contrast, those words do not appear in ham categories. 

![diagram flow](https://github.com/hennypurwadi/spam_classifier/blob/main/spam_filter/docs/images/ham_wordcloud.jpeg)

## Compared several Machine Learning Model's performance:

![ML performence](https://github.com/hennypurwadi/spam_classifier/blob/main/spam_filter/docs/images/ML_performance.jpg)

## Error Analysis
Using simpler metrics like accuracy score only without comparing to other metrics can be misleading. Logistic Regression model show high accuracy just by predicting the majority class (ham), although it failed to identify minority class (spam). This can result in a high rate of false negative, where spam emails incorrectly classified as ham. 

Multinomial naiive bayes as the best model, has been saved as both model.pkl and vectorizer.pkl. It has been loaded and deployed to cloud.
Although Multinomial Naïve Bayes performed the best among other algorithms in training and testing data, it still shows several errors in classifying new data.

## Limitation
This study has several limitations, such as: 
1.Too small dataset size, which has led to inadequate training.
2.Imbalance amount of two categories, lack of training for the minor category compared to the majority.   

In datasets with severely class-imbalanced classifiers, the classifier will always “predict” the most common class without performing any feature analysis and will have a high degree of accuracy, but not the correct one.

## Recommendations for Future
Machine learning algorithms perform optimally when the number of samples in each class is about the same. When the data set is imbalanced, a high accuracy rate can be achieved by predicting the majority class, but this will lead to a failure to recognize the minority class, which is often the main objective of creating the model in the first place. Resampling technique can be used to highly imbalanced datasets. Under-sampling will remove samples from the majority class, while over-sampling will add more examples for the minority class. 

10 Techniques to deal with Imbalanced Classes in Machine Learning. (2020, July 23). https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/

## Environment setup:

### To create conda environment from anaconda command prompt (envir.yaml)

(base) C:\> cd\

(base) C:\> cd Users\Asus\PYTHON_C\DLBDSME01\spam_filter

(base) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> conda env create -f envir.yaml

### To activate conda environment from anaconda command prompt:

(base) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> conda activate envir

(envir) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> jupyter notebook

(envir) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> python -m notebook

### To escape:

ctrl + C or ctrl + Z

### To de-activate conda environment from anaconda command prompt:

(envir) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> conda deactivate

### To update conda environment from anaconda command prompt:

(envir) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> conda env update -f envir.yaml

--------------

### To create environment from any command prompt without anaconda(requirements.txt)

C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> py -m venv environ

### To activate environment

C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> environ\Scripts\activate

### To install requirements for environment:

(environ) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter>pip install -r requirements.txt

-------------------

## To run streamlit_app.py from command prompt with streamlit:

(environ) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> streamlit run spam_classifier.py

------------------------
