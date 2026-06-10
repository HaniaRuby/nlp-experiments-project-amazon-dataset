# scripts/train.py
import os
import pickle
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from app.pipeline import TextPreprocessor

# --- LOAD CENTRALIZED CONFIG ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- DAGSHUB CONFIGURATION FROM CONFIG ---
os.environ["MLFLOW_TRACKING_USERNAME"] = config["mlflow"]["tracking_username"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = "22a664b8cca9908b3cb6d685aac3c27c2f01c956" 

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

def load_and_standardize_data(dataset_path, dataset_name):
    print(f"Loading and parse-cleaning: {dataset_name} from {dataset_path}")
    
    if dataset_name == "amazon":
        df = pd.read_csv(dataset_path)
        df = df[df["Score"] != 3].copy()
        df["sentiment"] = df["Score"].apply(lambda x: "positive" if x > 3 else "negative")
        df = df.rename(columns={"Text": "text"})
        
    elif dataset_name == "sentiment140":
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=cols)
        df["sentiment"] = df["target"].apply(lambda x: "positive" if x == 4 else "negative")
    else:
        raise ValueError(f"Unknown dataset key: {dataset_name}")
        
    df = df[["text", "sentiment"]].dropna()
    return df.sample(n=config["data"]["sample_size"], random_state=config["data"]["random_state"])

def run_experiment(dataset_path, dataset_name, vectorizer_type="tfidf", preprocess_mode="lemmatize"):
    run_name = f"{dataset_name}_{vectorizer_type}_{preprocess_mode}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Starting Experiment Run: {run_name} ---")
        
        try:
            df = load_and_standardize_data(dataset_path, dataset_name)
        except Exception as e:
            print(f"Data loading failed for {dataset_name}: {e}")
            return
        
        preprocessor = TextPreprocessor(mode=preprocess_mode)
        print("Transforming and tokenizing corpus text fields...")
        df['processed_text'] = df['text'].apply(preprocessor.process)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['sentiment'], 
            test_size=config["data"]["test_size"], 
            random_state=config["data"]["random_state"]
        )
        
        # Max features handled dynamically via configuration file
        max_feats = config["features"]["max_features"]
        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(max_features=max_feats)
        else:
            vectorizer = CountVectorizer(max_features=max_feats)
            
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(max_iter=config["model"]["max_iter"])
        model.fit(X_train_vec, y_train)
        
        predictions = model.predict(X_test_vec)
        acc = accuracy_score(y_test, predictions)
        
        unique_labels = set(y_test)
        pos_label = 'positive' if 'positive' in unique_labels else 1
        f1 = f1_score(y_test, predictions, average='binary', pos_label=pos_label) 
        
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("vectorizer", vectorizer_type)
        mlflow.log_param("preprocess_mode", preprocess_mode)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        os.makedirs("models", exist_ok=True)
        with open(f"models/{dataset_name}_best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(f"models/{dataset_name}_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
            
        mlflow.sklearn.log_model(model, "model")
        print(f"Run Complete! Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    amazon_csv_path = config["data"]["amazon"]["path"]
    sentiment_csv_path = config["data"]["sentiment140"]["path"]
    
    vectorizers = ["bow", "tfidf"]
    preprocessing_modes = ["stem", "lemmatize"]
    
    if os.path.exists(amazon_csv_path):
        print("\nLAUNCHING AMAZON EXPERIMENT SUITE VIA CONFIG")
        for vec in vectorizers:
            for mode in preprocessing_modes:
                run_experiment(amazon_csv_path, "amazon", vec, mode)

    if os.path.exists(sentiment_csv_path):
        print("\nLAUNCHING SENTIMENT140 EXPERIMENT SUITE VIA CONFIG")
        for vec in vectorizers:
            for mode in preprocessing_modes:
                run_experiment(sentiment_csv_path, "sentiment140", vec, mode)