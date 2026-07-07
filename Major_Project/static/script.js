const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message to chat
    appendMessage(text, 'user-message');
    userInput.value = '';
    
    // Add typing indicator
    const typingIndicator = appendMessage('...', 'bot-message');

    try {
        const response = await fetch('/respond', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });
        
        const data = await response.json();
        chatBox.removeChild(typingIndicator);
        
        if (response.ok) {
            appendMessage(data.response, 'bot-message');
        } else {
            appendMessage(data.error || "Something went wrong.", 'bot-message');
        }
    } catch (error) {
        chatBox.removeChild(typingIndicator);
        appendMessage("Error connecting to server.", 'bot-message');
    }
}

function appendMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.textContent = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}
