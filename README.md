
<div align="center">

# <span>ECE 143 Project: Makeup Product Categorization for E-Commerce Applications</span>

</div>

## <div align="center"><span style="color: #e67e22;">Team Members</span></div>
- **<span style="color: #e74c3c;">Swapnil Sinha</span>**
- **<span style="color: #e74c3c;">Pragnya Pathak</span>**
- **<span style="color: #e74c3c;"> Xin Pan</span>**
- **<span style="color: #e74c3c;">Avanti Bhandarkar</span>**
- **<span style="color: #e74c3c;">Yuyang Wu</span>**

## <div align="center"><span style="color: #e67e22;">File Structure</span></div>
```
--- Data/
|   +-- makeup_original.csv
|   +-- cleaned_makeup.csv
|   +-- withUSE.csv
|   +-- ingredients.csv
|   +-- ingredients.txt
|   +-- colorants.csv
|   +-- colorants.txt
--- Scripts/
|   +-- utils.py
|   +-- preprocessing.py
|   +-- lda.py
|   +-- models_SVM_TfIdf.ipynb
|   +-- models_SVM_GPT3.ipynb
--- ECE143_ProductCategorization_Visualizations.ipynb
--- LDAvis.html
--- ECE143_Team17_Presentation.pdf
--- README.md
```
* `Data` stores all datasets for analysis.
  * `makeup_original.csv` - dataset from Heroku /makeup API
  * `cleaned_makeup.csv` - dataset after preprocessing
  *  `withUSE.csv` - cleaned dataset with USE word embeddings saved
  * `ingredients.csv / ingredients.txt` - FDA approved cosmetic ingredients dataset
  * `colorants.csv / colorants.txt` - FDA approved cosmetic colorants dataset
* `Scripts` stores all Python scripts.
  * `utils.py` contains helper functions for cleaning data and to perform certain feature engineering operations.
  * `preprocessing.py` contains all preprocessing functions used to preprocess the description column from makeup_original.csv
  * `models_SVM_TfIdf.py` contains SVM + Tfidf model for categorization
  * `models_SVM_GPT3.py` contains SVM + GPT3 model for categorization

* `ECE143_ProductCategorization_Visualizations.ipynb` is our visualization notebook, LDA modelling is excluded (check Scripts/lda.py)
* `LDAvis.html` HTML visualization of Latent Dirichlet Allocation based Topic Modelling
* `ECE143_Team17_Presentation.pdf` is the pdf of our presentation.
* `README.md`
## <div align="center"><span style="color: #e67e22;">Installation</span></div>

Make sure you have Python (version 3.9 or lower) installed on your machine.
Then, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/avanti-bhandarkar/ECE143_FinalProject_ProductCategorization
    ```

2. **Install dependencies:**

Install libraries mentioned in the 3rd party modules section below. Please note that some of these libraries may require the installation of other supplementary modules.

## <div align="center"><span style="color: #e67e22;">3rd Party Modules Required</span></div>
- **<span style="color: #e74c3c;"> Pandas - 1.5.3 </span>**
- **<span style="color: #e74c3c;"> Numpy - 1.23.5 </span>**
- **<span style="color: #e74c3c;"> Matplotlib - 3.7.1
</span>**
- **<span style="color: #e74c3c;"> Seaborn - 0.12.2 </span>**
- **<span style="color: #e74c3c;"> NLTK - 3.8.1 </span>**
- **<span style="color: #e74c3c;"> SpaCy - 3.6.1
 </span>**
- **<span style="color: #e74c3c;"> Gensim - 4.3.2</span>**
- **<span style="color: #e74c3c;"> Sklearn - 1.2.2 </span>**
- **<span style="color: #e74c3c;"> pyLDAvis - 2.1.2 </span>**
- **<span style="color: #e74c3c;"> Wordcloud - 1.9.2 </span>**
- **<span style="color: #e74c3c;"> Tensorflow - 2.14.0 </span>**
- **<span style="color: #e74c3c;"> OpenAI - 0.27.2 </span>**
