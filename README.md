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

## Note:

### To create conda environment from anaconda command prompt (envir.yaml)

(base) C:\> cd\

(base) C:\> cd Users\Asus\PYTHON_C\DLBDSME01\spam_filter

(base) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> conda env create -f envir.yaml

### To activate conda environment from anaconda command prompt:

(base) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> conda activate envir

(envir) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> jupyter notebook

(envir) C:\Users\Asus\PYTHON_C\DLBDSME01\spam_filter> python -m notebook

### To escape:

ctrl + C 

ctrl + Z

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


  Local URL: http://localhost:8501
  
  Network URL: http://192.168.100.31:8501

------------------------
