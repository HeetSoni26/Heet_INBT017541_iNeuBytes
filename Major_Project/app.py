# app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from chatbot_engine import CineBotEngine

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) # Allow frontend to communicate with backend

# Security: Use environment variables for sensitive info
app.secret_key = os.getenv('SECRET_KEY', 'default-dev-secret-change-in-production')

# Load the model ONCE when the server starts
print("Loading NLP Model...")
bot_engine = CineBotEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "CineBot server is running smoothly."}), 200

@app.route('/respond', methods=['POST'])
def respond():
    try:
        data = request.get_json()
        
        # Error Handling: Check if input exists
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field in request body."}), 400
            
        user_message = data['message']
        
        if not isinstance(user_message, str):
            return jsonify({"error": "Message must be a string."}), 400

        # Get response from NLP model
        bot_response = bot_engine.get_response(user_message)
        
        return jsonify({
            "response": bot_response,
            "status": "success"
        }), 200
        
    except Exception as e:
        # Error Handling: Prevent 500 server crashes, return clear JSON
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500

if __name__ == '__main__':
    # Use environment variable for port (required for Render/Railway)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
