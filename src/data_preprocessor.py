import json
import numpy as np
import pandas as pd
import nltk
import pickle
import os
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.stemmer = PorterStemmer()
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = config.MAX_SEQUENCE_LENGTH
        self.responses_dict = {}
        
    def load_data(self, data_path=None, data_format=None):
        """Load training data from various formats"""
        if data_path is None:
            data_path = self.config.DATA_PATH
        if data_format is None:
            data_format = self.config.DATA_FORMAT
            
        if data_format == 'csv':
            return self._load_csv_data(data_path)
        elif data_format == 'txt':
            return self._load_txt_data(data_path)
        elif data_format == 'json':
            return self._load_json_data(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _load_csv_data(self, data_path):
        """Load data from CSV file"""
        df = pd.read_csv(data_path)
        
        # Handle different CSV formats
        if self.config.CSV_INTENT_COL in df.columns:
            # Format: question, answer, intent
            data = {'intents': []}
            intents_dict = defaultdict(lambda: {'patterns': [], 'responses': []})
            
            for _, row in df.iterrows():
                intent = row[self.config.CSV_INTENT_COL]
                intents_dict[intent]['patterns'].append(row[self.config.CSV_QUESTION_COL])
                intents_dict[intent]['responses'].append(row[self.config.CSV_ANSWER_COL])
            
            for intent, content in intents_dict.items():
                data['intents'].append({
                    'tag': intent,
                    'patterns': content['patterns'],
                    'responses': content['responses']
                })
        else:
            # Format: question, answer (single intent)
            data = {'intents': [{
                'tag': 'general',
                'patterns': df[self.config.CSV_QUESTION_COL].tolist(),
                'responses': df[self.config.CSV_ANSWER_COL].tolist()
            }]}
        
        return data
    
    def _load_txt_data(self, data_path):
        """Load data from TXT file"""
        data = {'intents': [{'tag': 'conversation', 'patterns': [], 'responses': []}]}
        
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if self.config.TXT_DELIMITER in line:
                parts = line.split(self.config.TXT_DELIMITER)
                if len(parts) >= 2:
                    data['intents'][0]['patterns'].append(parts[0].strip())
                    data['intents'][0]['responses'].append(parts[1].strip())
            else:
                # If no delimiter, treat as question (you might want to handle this differently)
                data['intents'][0]['patterns'].append(line)
                data['intents'][0]['responses']("I understand. Can you tell me more?")
        
        return data
    
    def _load_json_data(self, data_path):
        """Load data from JSON file (original format)"""
        with open(data_path, 'r') as file:
            data = json.load(file)
        return data
    
    def preprocess_text(self, text):
        """Preprocess individual text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token.isalnum()]
        return ' '.join(tokens)
    
    def prepare_training_data(self, data_path=None, data_format=None):
        """Prepare training data from various file formats"""
        data = self.load_data(data_path, data_format)
        
        documents = []
        labels = []
        
        # Build responses dictionary
        self.responses_dict = {}
        for intent in data['intents']:
            self.responses_dict[intent['tag']] = intent['responses']
            for pattern in intent['patterns']:
                # Preprocess pattern
                processed_pattern = self.preprocess_text(pattern)
                documents.append(processed_pattern)
                labels.append(intent['tag'])
        
        # Create and fit tokenizer
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.tokenizer.fit_on_texts(documents)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(documents)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        # Save preprocessing objects
        self.save_preprocessing_objects()
        
        return X, y, len(self.tokenizer.word_index) + 1, len(self.label_encoder.classes_)
    
    def prepare_single_input(self, text):
        """Prepare single input for prediction"""
        if self.tokenizer is None:
            self.load_preprocessing_objects()
            
        processed_text = self.preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
        return padded_sequence
    
    def save_preprocessing_objects(self):
        """Save tokenizer, label encoder, and responses"""
        os.makedirs(os.path.dirname(self.config.TOKENIZER_SAVE_PATH), exist_ok=True)
        
        with open(self.config.TOKENIZER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        with open(self.config.LABEL_ENCODER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        with open(self.config.RESPONSES_SAVE_PATH, 'wb') as f:
            pickle.dump(self.responses_dict, f)
    
    def load_preprocessing_objects(self):
        """Load tokenizer, label encoder, and responses"""
        with open(self.config.TOKENIZER_SAVE_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
            
        with open(self.config.LABEL_ENCODER_SAVE_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        with open(self.config.RESPONSES_SAVE_PATH, 'rb') as f:
            self.responses_dict = pickle.load(f)