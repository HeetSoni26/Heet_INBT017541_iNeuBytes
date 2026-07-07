# chatbot_engine.py
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK resources quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class CineBotEngine:
    def __init__(self, data_path='data/cornell movie-dialogs corpus'):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.responses = []
        self.corpus = []
        
        print("Initializing CineBot Engine...")
        self._load_and_process_data(data_path)
        self._train_model()
        print("CineBot Engine Ready!")

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def _load_and_process_data(self, data_path):
        # Load movie lines
        lines_path = os.path.join(data_path, 'movie_lines.txt')
        conversations_path = os.path.join(data_path, 'movie_conversations.txt')
        
        line_dict = {}
        with open(lines_path, 'r', encoding='iso-8859-1', errors='ignore') as f:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) >= 4:
                    line_dict[parts[0].strip()] = parts[4].strip()

        # Build conversation pairs
        with open(conversations_path, 'r', encoding='iso-8859-1', errors='ignore') as f:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) >= 4:
                    conv_ids = eval(parts[3])
                    for i in range(len(conv_ids) - 1):
                        input_line = line_dict.get(conv_ids[i], '')
                        response_line = line_dict.get(conv_ids[i+1], '')
                        if input_line and response_line:
                            self.corpus.append(self._clean_text(input_line))
                            self.responses.append(response_line)
                            
                            # Limit to 5000 conversations to ensure fast boot and low RAM usage on Render
                            if len(self.corpus) >= 5000:
                                return

    def _train_model(self):
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.vectorizer.fit(self.corpus)

    def get_response(self, user_input):
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you please type a message about movies?"
        
        cleaned_input = self._clean_text(user_input)
        input_vector = self.vectorizer.transform([cleaned_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, self.vectorizer.transform(self.corpus))
        best_match_idx = np.argmax(similarities)
        best_score = similarities[0, best_match_idx]
        
        # Threshold to avoid irrelevant responses
        if best_score > 0.2:
            return self.responses[best_match_idx]
        else:
            return "That's an interesting thought! I mostly know about movies. Try asking about a film scene or dialogue!"
