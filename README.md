iNeuBytes Internship - Final Project Report

Name: Soni Heet Kalpeshbhai
Registration No.: INBT017541
Course ID: AIINB10626
GitHub Repository: HeetSoni26/Heet_INBT017541_iNeuBytes 
Live Application URL: CineBot – Your AI Movie Companion 
Section 1: Computer Vision using CNN Models
Overview
The objective of this task was to implement and evaluate Convolutional Neural Networks (CNNs) for image classification on the CIFAR-10 dataset.

Part A: Traditional CNN Implementation
I built a traditional AlexNet-style Convolutional Neural Network adapted for 32x32 images. The model consisted of three convolutional blocks (with 64, 128, and 256 filters), followed by fully connected dense layers. The network utilized ReLU activation, MaxPooling, Batch Normalization, and Dropout to prevent overfitting.
Goal: Achieve ≥70% accuracy.
Results: The model successfully processed the images, extracting foundational features, and converged effectively during training.


Part B: Customized CNN Architecture
To improve upon the traditional model, I designed a customized architecture. This model introduced data augmentation techniques (random flips, rotations, and zooms) to improve generalization. Furthermore, it utilized more advanced block structures, increased filter depths, and optimized learning rate scheduling (ReduceLROnPlateau).
Goal: Improve upon Part A's accuracy by at least 3%.
Results: The custom architecture successfully learned more complex hierarchical features and outperformed the traditional model, showcasing the impact of modern architectural choices and regularization.
Section 2: Sentiment Analysis using ML and DL
Overview
This task involved Natural Language Processing (NLP) to classify the sentiment of movie reviews using the IMDb dataset. 

Part A: Traditional Machine Learning Models
In this phase, I utilized classic ML algorithms: Logistic Regression and Support Vector Machines (SVM).
Preprocessing: The raw text was cleaned (lowercased, punctuation removed) and vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) with a cap of 5000 features.
Results: The traditional models provided a strong, fast baseline for sentiment classification, successfully differentiating between positive and negative reviews based on word frequencies.
Part B: Deep Learning with LSTM
I built a deep learning model using an LSTM (Long Short-Term Memory) network to capture the sequential nature and context of the text.
Architecture: An Embedding layer initialized the word vectors, followed by a Bidirectional LSTM layer, GlobalMaxPooling, and dense output layers.
Results: The LSTM model demonstrated a superior ability to understand the contextual meaning of sentences compared to the TF-IDF approach, resulting in more nuanced and accurate predictions.
Major Project: Personalized NLP Chatbot (CineBot)
Overview
For the Major Project, I developed a Full-Stack AI Web Application named “CineBot”, an interactive, personalized chatbot designed to answer questions and analyze sentiments related to movies. 
Technical Architecture
Backend: A robust RESTful API built with Python and Flask. It exposes a /chat endpoint that processes user input and a /health endpoint for monitoring. It includes robust error handling to return clean JSON errors rather than server crashes.
Frontend: A clean, user-friendly HTML/CSS/JavaScript interface that allows non-technical users to interact seamlessly with the AI model.
AI Engine: The chatbot utilizes a machine learning NLP pipeline. It extracts intents using TF-IDF and Logistic Regression, enabling it to classify whether a user is asking for recommendations, providing a review, or greeting the bot.
Deployment & Security: The application was designed for cloud deployment, utilizing .env variables to keep configurations secure and avoiding hard-coded secrets.

API Testing & Documentation
The backend APIs were thoroughly tested using Postman to ensure reliable response times (under 3-5 seconds) and appropriate error handling.
Section 3: Key Learnings & Takeaways

Throughout this internship, I gained hands-on experience in the complete lifecycle of AI application development:
 Model Optimization: Learned how to effectively use data augmentation, dropout, and batch normalization to prevent overfitting and boost the performance of CNNs in Computer Vision tasks.
 NLP Pipelines: Gained practical knowledge in comparing traditional TF-IDF + ML approaches with modern sequential Deep Learning architectures (LSTMs) for text analysis.
 Full-Stack AI Integration: Successfully bridged the gap between a machine learning model and a user interface by wrapping an ML pipeline in a Flask API.
 Best Practices: Developed an understanding of production-ready coding standards, including environment variable security, comprehensive API testing with Postman, and structured GitHub repository management.

LIVE CHAT DEMO : <img width="1379" height="976" alt="image" src="https://github.com/user-attachments/assets/db79076d-0b50-428f-bd99-abe7d11dd851" />





