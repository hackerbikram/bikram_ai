import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model parameters
    LEARNING_RATE = 0.001
    EPOCHS = 500
    HIDDEN_LAYERS = 2
    HIDDEN_UNITS = 128
    DROPOUT_RATE = 0.5
    
    # Data parameters
    MAX_SEQUENCE_LENGTH = 20
    MIN_CONFIDENCE = 0.7
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'intents.csv')  # Changed to CSV
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_model')
    TOKENIZER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')
    LABEL_ENCODER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
    RESPONSES_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'responses.pkl')
    
    # Training
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Data format settings
    DATA_FORMAT = 'csv'  # 'csv', 'txt', 'json'
    CSV_QUESTION_COL = 'question'  # Column name for questions/patterns
    CSV_ANSWER_COL = 'answer'      # Column name for answers
    CSV_INTENT_COL = 'intent'      # Column name for intent tags
    
    # TXT file settings
    TXT_DELIMITER = '|'  # Delimiter for TXT files