## Spam filter

Cloud link: https://hennypurwadi-spam-classifier-spam-classifier-ubc344.streamlit.app/


![diagram flow] (https://github.com/hennypurwadi/spam_classifier/blob/main/spam_filter/docs/images/spam_filter_diagram.jpeg)

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
