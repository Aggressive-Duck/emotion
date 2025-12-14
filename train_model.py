import pandas as pd
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os

# Import our custom preprocessing module
import preprocessing

# Configuration
DATA_PATH = "./content/Emotion616.xlsx"
MODEL_DIR = "./models"

def main():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    df = pd.read_excel(DATA_PATH)
    
    # 1. Basic Cleaning
    print("Preprocessing text...")
    df['clean_text'] = df["Emails"].str.lower()
    df['clean_text'] = df['clean_text'].apply(preprocessing.remove_punctuation)
    df['clean_text'] = df['clean_text'].apply(preprocessing.replace_space)
    df['clean_text'] = df['clean_text'].apply(preprocessing.remove_stopword)

    # 2. Identify Frequent & Rare Words (Logic from notebook)
    print("Identifying frequent and rare words...")
    cnt = Counter()
    for text in df['clean_text']:
        for word in text.split():
            cnt[word] += 1
    
    # Notebook logic: Top 10 frequent words
    FREQUENT_WORDS = set([word for (word, wc) in cnt.most_common(10)])
    
    # Notebook logic: Rare words (last 10 in reverse, slicing from most_common list)
    # The notebook code was: RARE_WORDS = set([word for (word, wc) in cnt.most_common()[:-10:-1]])
    # Note: cnt.most_common() returns *all* items sorted. [:-10:-1] takes the last 9 items.
    RARE_WORDS = set([word for (word, wc) in cnt.most_common()[:-10:-1]])

    # 3. Remove Frequent & Rare Words
    df['clean_text'] = df['clean_text'].apply(lambda x: preprocessing.remove_specific_words(x, FREQUENT_WORDS))
    df['clean_text'] = df['clean_text'].apply(lambda x: preprocessing.remove_specific_words(x, RARE_WORDS))

    # 4. Lemmatization
    print("Lemmatizing...")
    df['lemmatized_text'] = df['clean_text'].apply(preprocessing.lemmatize_words)

    # 5. Vectorization (TF-IDF)
    print("Vectorizing...")
    # Notebook used max_features=100
    vectorizer = TfidfVectorizer(max_features=100)
    x = vectorizer.fit_transform(df['lemmatized_text']).toarray()

    # 6. Standardization
    print("Scaling...")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 7. PCA
    print("Applying PCA...")
    # Notebook used n_components=0.95
    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(x_scaled)

    # 8. Label Encoding
    print("Encoding Labels...")
    le = LabelEncoder()
    y_encode = le.fit_transform(df['Label'])
    
    # 9. Train SVM
    print("Training SVM Model...")
    # Notebook split the data, but for the final app model, it's often better to train on all data 
    # OR stick to the split if we want to validate.
    # To be safe and consistent with your notebook, I will use the split logic to ensure it runs, 
    # but I will fit the final model on the training set (or full set if preferred). 
    # Let's fit on the full dataset for the "production" app to maximize knowledge, 
    # or follow the notebook strictly. 
    # Notebook: x_train, x_test, y_train, y_test = train_test_split(x_pca, y_encode, test_size=0.2, random_state=42)
    # Notebook: svm.fit(x_train, y_train)
    
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y_encode, test_size=0.2, random_state=42)
    
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr', probability=True)
    svm_model.fit(x_train, y_train) # Fitting on train split as per notebook

    # 10. Save Artifacts
    print("Saving model artifacts...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.joblib'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.joblib'))
    
    # Save metadata (frequent/rare words)
    metadata = {
        'frequent_words': FREQUENT_WORDS,
        'rare_words': RARE_WORDS
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, 'metadata.joblib'))

    print("Success! Model and artifacts saved to ./models/")

if __name__ == "__main__":
    main()
