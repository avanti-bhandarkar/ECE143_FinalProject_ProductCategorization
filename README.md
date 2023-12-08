# Makeup Product Categorization for E-Commerce Applications

### File Structure
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
|   +-- models.py
--- ECE143_ProductCategorization_Visualizations.ipynb
--- LDAvis.html
--- ECE143_Team17_Presentation.pdf
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

* `ECE143_ProductCategorization_Visualizations.ipynb` is our visualization notebook
* `LDAvis.html` HTML visualization of Latent Dirichlet Allocation based Topic Modelling
* `ECE143_Team17_Presentation.pdf` is the pdf of our presentation.

###
