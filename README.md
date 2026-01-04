## 🎯 Objective

Automatically group news statements into clusters to reveal patterns in truthfulness and content. This aids in detecting misinformation and understanding relationships between statements.

---

## 📝 Problem Statement

Manual classification of news statements is time-consuming and subjective. Using unsupervised machine learning, this project clusters news into meaningful groups, providing consistent insights.

---

## 🛠️ Tech Stack

- **Python**  
- **Pandas, NumPy** – Data manipulation  
- **Matplotlib, Seaborn** – Visualization  
- **NLTK** – Text preprocessing  
- **Scikit-Learn** – TF-IDF, PCA, K-Means, Agglomerative Clustering  
- **Pickle** – Model serialization  
- **Jupyter Notebook** – Development environment  

---

## 📊 Dataset

- **10,239 news statements** labeled as:  
  `true`, `false`, `half-true`, `mostly-true`, `barely-true`, `pants-fire`  
- Relevant columns: `label`, `news_text`  
- Balanced dataset created with **204 samples** (34 per class)

### Preprocessing

- Lowercasing, tokenization, removing punctuation & stop words  
- Stemming with `PorterStemmer`  
- TF-IDF vectorization  
- PCA for dimensionality reduction  

---

## ⚙️ Models

| Model                                 | Optimal Clusters | Silhouette Score |
|--------------------------------------|----------------|----------------|
| K-Means Clustering                    | 6              | 0.358          |
| Agglomerative Hierarchical Clustering | 3              | 0.524          |

> Agglomerative Hierarchical Clustering performed best.

---

## 🔍 Sample Predictions

### Single Input (K-Means)

```python
sample = "Health care reform legislation is likely to mandate free sex change surgeries."
preprocessed = preprocess_text(sample)
vectorized = vectorizer.transform([preprocessed]).toarray()
pca_transformed = pca.transform(vectorized)
cluster = kmeans.predict(pca_transformed)
print(cluster[0])  # Example Output: 2
Multiple Inputs (Agglomerative)
python
Copy code
inputs = [
    "The Chicago Bears have had more starting quarterbacks...",
    "Jim Dunnam has not lived in the district he represents...",
    "Health care reform legislation is likely to mandate..."
]

preprocessed = [preprocess_text(x) for x in inputs]
vectorized = vectorizer.transform(preprocessed).toarray()
pca_transformed = pca.transform(vectorized)
clusters = cluster_hierarcial.fit_predict(pca_transformed)
print(clusters)  # Example Output: [2, 1, 0]
📈 Visualizations
Elbow Method → Optimal clusters for K-Means

Dendrogram → Optimal clusters for Agglomerative

Scatter Plot (PCA) → Cluster visualization

Replace images/*.png with your actual plot file paths.

💾 Files Generated
modelkmeans.pkl → K-Means model

modelhierarchy.pkl → Agglomerative model

vectorizer.pkl → TF-IDF Vectorizer

pca.pkl → PCA transformer

python
Copy code
import pickle
pickle.dump(kmeans, open('modelkmeans.pkl', 'wb'))
pickle.dump(cluster_hierarcial, open('modelhierarchy.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(pca, open('pca.pkl', 'wb'))
🚀 Next Steps
Experiment with DBSCAN, K-Medoids, or Spectral Clustering

Use Word Embeddings (Word2Vec, GloVe, BERT) instead of TF-IDF for better semantic understanding

Expand dataset with more news sources for better generalization

Deploy a web interface to input statements and get cluster predictions in real-time

👤 Author
Esraa Naji 

GitHub:EsraaCodes
LinkedIn: https://linkedin.com/in/esraa-naji/


