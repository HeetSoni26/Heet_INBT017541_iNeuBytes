# chatbot_engine.py
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─────────────────────────────────────────────
#  INTENT DEFINITIONS  (patterns → responses)
# ─────────────────────────────────────────────
INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "howdy", "greetings", "sup", "what's up",
            "good morning", "good evening", "good afternoon", "hiya", "yo"
        ],
        "responses": [
            "Hey there, cinephile! 🎬 I'm CineBot. Ask me for movie recommendations, trivia, or just chat about films!",
            "Hello! 🍿 Ready to talk movies? Ask me for a recommendation or quiz me on film trivia!",
            "Hey! 🎥 CineBot at your service. What kind of movie are you in the mood for?",
            "Greetings! I live and breathe cinema. Ask me anything about movies! 🎞️"
        ]
    },
    {
        "tag": "recommend_action",
        "patterns": [
            "recommend action movie", "best action films", "action movie suggestion",
            "good action movies", "action film recommendation", "thriller movie",
            "recommend thriller", "best thrillers", "action", "give me action"
        ],
        "responses": [
            "🔥 **Top Action Picks:**\n• **Mad Max: Fury Road** (2015) – Pure adrenaline masterpiece\n• **John Wick** (2014) – The ultimate hitman saga\n• **Die Hard** (1988) – The gold standard of action\n• **The Dark Knight** (2008) – Action with depth\n• **Mission Impossible: Fallout** (2018) – Best stunts ever filmed",
            "💥 **Must-Watch Action Films:**\n• **Inception** (2010) – Mind-bending action\n• **The Raid** (2011) – Jaw-dropping martial arts\n• **Top Gun: Maverick** (2022) – Modern classic\n• **Heat** (1995) – Crime action perfection\n• **Gladiator** (2000) – Epic and intense"
        ]
    },
    {
        "tag": "recommend_comedy",
        "patterns": [
            "recommend comedy", "funny movies", "best comedies", "comedy film",
            "make me laugh", "hilarious movie", "comedy recommendation", "funny film",
            "comedy", "want to laugh"
        ],
        "responses": [
            "😂 **Hilarious Must-Watches:**\n• **The Grand Budapest Hotel** (2014) – Quirky, witty perfection\n• **Superbad** (2007) – Timeless teen comedy\n• **What We Do in the Shadows** (2014) – Vampire mockumentary gold\n• **Knives Out** (2019) – Comedy mystery brilliance\n• **Game Night** (2018) – Surprisingly hilarious",
            "🤣 **Comedy Gems:**\n• **The Nice Guys** (2016) – Buddy comedy at its best\n• **Three Billboards Outside Ebbing, Missouri** (2017) – Dark comedy masterpiece\n• **Parasite** (2019) – Dark humor with depth\n• **Bridesmaids** (2011) – Laugh-out-loud classic\n• **Hunt for the Wilderpeople** (2016) – Heart and humor"
        ]
    },
    {
        "tag": "recommend_horror",
        "patterns": [
            "horror movie", "scary movie", "best horror", "recommend horror",
            "horror film", "want to be scared", "scary films", "ghost movie",
            "horror", "frightening movie"
        ],
        "responses": [
            "👻 **Scary Essentials:**\n• **Hereditary** (2018) – Deeply terrifying and emotional\n• **Get Out** (2017) – Psychological horror genius\n• **The Conjuring** (2013) – Modern horror classic\n• **A Quiet Place** (2018) – Silent dread perfection\n• **Midsommar** (2019) – Unsettling daylight horror",
            "🩸 **Horror Picks:**\n• **It Follows** (2014) – Creepy and original concept\n• **The Babadook** (2014) – Psychological masterpiece\n• **Us** (2019) – Intelligent and terrifying\n• **Sinister** (2012) – Genuinely disturbing\n• **The Witch** (2015) – Slow-burn atmospheric dread"
        ]
    },
    {
        "tag": "recommend_scifi",
        "patterns": [
            "sci fi movie", "science fiction", "space movie", "recommend sci fi",
            "best science fiction", "space film", "futuristic movie", "alien movie",
            "scifi", "sci-fi"
        ],
        "responses": [
            "🚀 **Sci-Fi Masterpieces:**\n• **Interstellar** (2014) – Emotional space epic\n• **Blade Runner 2049** (2017) – Visual and philosophical perfection\n• **Arrival** (2016) – Thought-provoking alien contact\n• **The Martian** (2015) – Feel-good survival in space\n• **Ex Machina** (2014) – AI thriller brilliance",
            "🌌 **Must-See Sci-Fi:**\n• **2001: A Space Odyssey** (1968) – The ultimate sci-fi\n• **Dune** (2021) – Epic world-building\n• **Moon** (2009) – Intimate psychological sci-fi\n• **District 9** (2009) – Gritty and original\n• **Everything Everywhere All At Once** (2022) – Multiverse madness"
        ]
    },
    {
        "tag": "recommend_romance",
        "patterns": [
            "romance movie", "love story", "romantic film", "best romance",
            "recommend romance", "date night movie", "romantic comedy",
            "romantic", "love movie", "feel good romance"
        ],
        "responses": [
            "❤️ **Romance Must-Watches:**\n• **La La Land** (2016) – Bittersweet and beautiful\n• **Before Sunrise** (1995) – Perfect conversation-driven romance\n• **Eternal Sunshine of the Spotless Mind** (2004) – Love and memory\n• **Her** (2013) – Unconventional and deeply touching\n• **Pride & Prejudice** (2005) – Timeless classic",
            "💕 **Love Story Gems:**\n• **Call Me By Your Name** (2017) – Summer romance masterpiece\n• **500 Days of Summer** (2009) – Realistic and relatable\n• **About Time** (2013) – Heartwarming time-travel romance\n• **The Notebook** (2004) – Classic tearjerker\n• **Crazy, Stupid, Love** (2011) – Charming and funny"
        ]
    },
    {
        "tag": "recommend_animated",
        "patterns": [
            "animated movie", "animation", "cartoon movie", "pixar", "disney",
            "anime film", "animated film", "best animation", "recommend animation"
        ],
        "responses": [
            "✨ **Animation Masterpieces:**\n• **Spider-Man: Into the Spider-Verse** (2018) – Visually revolutionary\n• **Spirited Away** (2001) – Miyazaki's magnum opus\n• **Coco** (2017) – Beautiful and emotional\n• **The Lion King** (1994) – Timeless classic\n• **Your Name** (2016) – Stunning Japanese animation",
            "🎨 **Animation Gems:**\n• **Ratatouille** (2007) – Surprisingly deep and funny\n• **Howl's Moving Castle** (2004) – Magical and moving\n• **Wall-E** (2008) – Minimal dialogue, maximum emotion\n• **Encanto** (2021) – Vibrant and heartfelt\n• **Princess Mononoke** (1997) – Epic environmental fantasy"
        ]
    },
    {
        "tag": "movie_trivia",
        "patterns": [
            "movie trivia", "film fact", "interesting movie fact", "tell me something about movies",
            "movie fact", "did you know", "movie trivia question", "surprise me",
            "fun fact", "film trivia"
        ],
        "responses": [
            "🎬 **Did you know?**\nThe famous line *'I'm gonna make him an offer he can't refuse'* from **The Godfather** was improvised by Marlon Brando. The original script was very different!",
            "🎥 **Film Fact!**\n**Interstellar**'s black hole, 'Gargantua', was so accurately simulated using Einstein's equations that it led to a new scientific discovery about gravitational lensing — published in a real astrophysics journal!",
            "🍿 **Fun Fact:**\nThe crew of **Mad Max: Fury Road** spent 480 days in the Namib Desert. Nearly 80% of the stunts were practical — almost no CGI!",
            "🎞️ **Did you know?**\n**Parasite** (2019) is only the second film in Oscar history to win both Best Picture AND Best Director. It's also the first non-English language film to ever win Best Picture.",
            "⭐ **Film Trivia:**\nThe shower scene in **Psycho** (1960) used chocolate syrup for blood because it looked more realistic in black and white. It took 7 days to film 45 seconds of footage.",
            "🎭 **Movie Fact:**\nTom Hanks had *zero* lines of dialogue for the first 25 minutes of **Cast Away** (2000) — yet the audience is completely riveted. Pure acting mastery.",
            "🎬 **Did you know?**\n**The Dark Knight** (2008) had a budget of $185M but made $1 billion worldwide — mostly because Heath Ledger's Joker was so iconic that word-of-mouth was unstoppable."
        ]
    },
    {
        "tag": "best_movies_all_time",
        "patterns": [
            "best movies ever", "greatest films of all time", "top movies", "most acclaimed films",
            "best film ever made", "greatest movie ever", "all time classics", "legendary movies",
            "what are the best films", "cinema masterpieces"
        ],
        "responses": [
            "🏆 **The Greatest Films Ever Made:**\n\n🥇 **The Godfather** (1972) – Perfect storytelling\n🥈 **Schindler's List** (1993) – Devastating and important\n🥉 **2001: A Space Odyssey** (1968) – Changed cinema forever\n\n**Also Essential:**\n• Pulp Fiction (1994)\n• The Shawshank Redemption (1994)\n• Parasite (2019)\n• Citizen Kane (1941)\n• Blade Runner (1982)\n• Apocalypse Now (1979)\n• Spirited Away (2001)"
        ]
    },
    {
        "tag": "director_info",
        "patterns": [
            "best director", "greatest directors", "who directed", "famous directors",
            "christopher nolan", "nolan movies", "spielberg", "kubrick", "tarantino",
            "favorite director", "director recommendation"
        ],
        "responses": [
            "🎬 **Legendary Directors & Their Best Works:**\n\n• **Christopher Nolan** – Inception, The Dark Knight, Interstellar\n• **Stanley Kubrick** – 2001, The Shining, Full Metal Jacket\n• **Quentin Tarantino** – Pulp Fiction, Django Unchained, Kill Bill\n• **Steven Spielberg** – Schindler's List, Jaws, E.T.\n• **Martin Scorsese** – Goodfellas, Taxi Driver, The Departed\n• **Hayao Miyazaki** – Spirited Away, Princess Mononoke, Howl's Moving Castle\n• **Bong Joon-ho** – Parasite, Memories of Murder, The Host"
        ]
    },
    {
        "tag": "sentiment_positive",
        "patterns": [
            "that movie was amazing", "i loved that film", "great movie", "awesome film",
            "incredible movie", "best movie ever", "loved it", "fantastic film",
            "it was perfect", "brilliant movie", "i enjoyed it", "that was good"
        ],
        "responses": [
            "🎉 Glad you loved it! That passion for cinema is what it's all about! Want me to recommend something similar?",
            "❤️ A great movie stays with you forever, doesn't it? Want me to suggest more films in that style?",
            "🍿 Amazing taste! That kind of movie is exactly why cinema is so special. Want more recommendations?"
        ]
    },
    {
        "tag": "sentiment_negative",
        "patterns": [
            "that movie was bad", "i hated it", "terrible movie", "boring film",
            "worst movie", "didn't like it", "waste of time", "awful film",
            "disappointing movie", "it was bad", "not good"
        ],
        "responses": [
            "😬 Ouch! Not every film is a winner. Tell me what you *do* like and I'll find you something you'll actually enjoy!",
            "That happens! Cinema is subjective. What genres or moods do you enjoy? I'll find you a perfect match! 🎬",
            "Sounds like that one missed the mark for you! No worries — what kind of movies *do* you love? I'll find something great. 🍿"
        ]
    },
    {
        "tag": "oscar_info",
        "patterns": [
            "oscar", "academy awards", "best picture winner", "oscar winning movie",
            "academy award", "oscar nominations", "who won oscar", "best actress oscar"
        ],
        "responses": [
            "🏆 **Recent Best Picture Oscar Winners:**\n• 2024 – **Oppenheimer** (Nolan's masterpiece)\n• 2023 – **Everything Everywhere All at Once** (multiverse madness)\n• 2022 – **CODA** (heartwarming underdog story)\n• 2021 – **Nomadland** (quiet, beautiful film)\n• 2020 – **Parasite** (historic first non-English winner)\n• 2019 – **Green Book** (controversial choice)\n• 2018 – **The Shape of Water** (del Toro's fairy tale)"
        ]
    },
    {
        "tag": "rating_inquiry",
        "patterns": [
            "is it worth watching", "should i watch", "is it good", "is the movie good",
            "recommend it", "worth seeing", "is it worth it", "should i see it"
        ],
        "responses": [
            "I'd need to know which movie you mean! Name the film and I'll give you my honest take. 🎬",
            "Tell me the movie title and I'll let you know if it's worth your time! 🍿",
            "Which film are you asking about? Drop the title and I'll give you the full rundown! 🎥"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "farewell", "later", "cya", "take care",
            "good night", "see ya", "peace out", "i'm leaving"
        ],
        "responses": [
            "Goodbye, fellow cinephile! 🎬 Come back whenever you want to talk movies. Stay legendary! 🍿",
            "See you at the movies! 👋 Keep watching great cinema!",
            "Farewell! May every film you watch be a masterpiece. 🎥✨"
        ]
    },
    {
        "tag": "thanks",
        "patterns": [
            "thank you", "thanks", "thank you so much", "appreciate it", "cheers",
            "thx", "ty", "thanks a lot", "helpful"
        ],
        "responses": [
            "Happy to help! 🎬 Ask me anything else about movies!",
            "Anytime! That's what I'm here for. 🍿 Any other film questions?",
            "My pleasure! Cinema is life. What else can I help you with? 🎥"
        ]
    },
    {
        "tag": "about_bot",
        "patterns": [
            "who are you", "what are you", "what can you do", "tell me about yourself",
            "what do you know", "how do you work", "what is cinebot", "your purpose"
        ],
        "responses": [
            "🎬 I'm **CineBot** — your personal AI movie companion!\n\nHere's what I can do:\n• 🎭 Recommend movies by genre (action, horror, sci-fi, romance, comedy...)\n• 🏆 Tell you the greatest films of all time\n• 🎥 Share interesting movie trivia and fun facts\n• 🎬 Talk about legendary directors\n• 🏅 Give you Oscar history\n• 💬 Discuss your movie opinions\n\nJust ask away!"
        ]
    },
    {
        "tag": "fallback",
        "patterns": [""],
        "responses": [
            "Hmm, I'm not sure I caught that! 🎬 Try asking me for a **movie recommendation** — like 'recommend a sci-fi movie' or ask for some **movie trivia**!",
            "I'm more of a movie expert than a general chatbot 😄 Ask me: 'What are the best horror movies?' or 'Tell me a movie fact!'",
            "That one went over my head! 🎥 But I know everything about movies. Try: 'Best movies of all time' or 'Recommend a comedy'!"
        ]
    }
]

class CineBotEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.classifier = LogisticRegression(max_iter=1000, C=5)
        self.tag_to_responses = {}
        
        print("Initializing CineBot Engine...")
        self._train()
        print("CineBot Engine Ready!")

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    def _train(self):
        X, y = [], []
        for intent in INTENTS:
            self.tag_to_responses[intent["tag"]] = intent["responses"]
            for pattern in intent["patterns"]:
                X.append(self._preprocess(pattern))
                y.append(intent["tag"])
        
        self.vectorizer.fit(X)
        X_vec = self.vectorizer.transform(X)
        self.classifier.fit(X_vec, y)

    def get_response(self, user_input: str) -> str:
        if not user_input or not user_input.strip():
            return "Say something! I'm all ears — especially about movies. 🎬"
        
        processed = self._preprocess(user_input)
        vec = self.vectorizer.transform([processed])
        
        proba = self.classifier.predict_proba(vec)[0]
        max_proba = max(proba)
        predicted_tag = self.classifier.classes_[np.argmax(proba)]
        
        # If confidence is low, use fallback
        if max_proba < 0.25:
            predicted_tag = "fallback"
        
        responses = self.tag_to_responses.get(predicted_tag, 
                    self.tag_to_responses["fallback"])
        
        return random.choice(responses)
