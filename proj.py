import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class FakeNewsDetector:
    def _init_(self, model_type='bert'):
        self.model_type = model_type
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.classical_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        
    def preprocess_text(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = text.split()
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        # Join tokens back to string
        return ' '.join(tokens)
    
    def load_data(self, filepath):
        # Load dataset
        df = pd.read_csv(filepath)
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        return df
    
    def train_classical_model(self, df):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Create pipeline with TF-IDF and classifier
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classical_model = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', PassiveAggressiveClassifier(max_iter=50))
        ])
        
        # Train model
        self.classical_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classical_model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def train_bert_model(self, df):
        # Prepare dataset for BERT
        class NewsDataset(Dataset):
            def _init_(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len
                
            def _len_(self):
                return len(self.texts)
                
            def _getitem_(self, item):
                text = str(self.texts[item])
                label = self.labels[item]
                
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        # Initialize BERT tokenizer and model
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = NewsDataset(
            texts=X_train.tolist(),
            labels=y_train.tolist(),
            tokenizer=self.bert_tokenizer,
            max_len=128
        )
        
        test_dataset = NewsDataset(
            texts=X_test.tolist(),
            labels=y_test.tolist(),
            tokenizer=self.bert_tokenizer,
            max_len=128
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Train BERT model
        trainer.train()
        
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        return {'accuracy': accuracy}
    
    def train(self, data_path):
        df = self.load_data(data_path)
        
        if self.model_type == 'classical':
            self.train_classical_model(df)
        elif self.model_type == 'bert':
            self.train_bert_model(df)
        else:
            raise ValueError("Invalid model type. Choose 'classical' or 'bert'")
    
    def predict(self, text):
        if self.model_type == 'classical' and self.classical_model:
            processed_text = self.preprocess_text(text)
            return self.classical_model.predict([processed_text])[0]
        elif self.model_type == 'bert' and self.bert_model:
            inputs = self.bert_tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            return pred
        else:
            raise Exception("Model not trained yet")

# Example usage
if _name_ == "_main_":
    # Initialize detector (choose 'classical' or 'bert')
    detector = FakeNewsDetector(model_type='bert')
    
    # Train the model (replace with your dataset path)
    # Dataset should have 'text' and 'label' columns (0 for real, 1 for fake)
    detector.train('fake_news_dataset.csv')
    
    # Test prediction
    test_news = """
    Scientists have discovered a new species of flying elephants in the Amazon rainforest.
    The creatures are said to have a wingspan of 15 feet and can carry up to 3 people.
    """
    prediction = detector.predict(test_news)
    print("Prediction:", "Fake" if prediction == 1 else "Real")