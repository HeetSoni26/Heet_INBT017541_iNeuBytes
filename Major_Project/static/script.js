const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function fillInput(text) {
    userInput.value = text;
    userInput.focus();
    sendMessage();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message
    appendMessage(text, 'user');
    userInput.value = '';
    document.getElementById('send-btn').disabled = true;

    // Typing indicator
    const typingRow = appendTyping();

    try {
        const res = await fetch('/respond', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await res.json();
        chatBox.removeChild(typingRow);

        if (res.ok) {
            appendMessage(data.response, 'bot');
        } else {
            appendMessage(data.error || 'Something went wrong. Try again!', 'bot');
        }
    } catch (err) {
        chatBox.removeChild(typingRow);
        appendMessage('Oops! Connection error. Please try again.', 'bot');
    } finally {
        document.getElementById('send-btn').disabled = false;
        userInput.focus();
    }
}

function appendMessage(text, sender) {
    const row = document.createElement('div');
    row.classList.add('message-row', sender === 'user' ? 'user-row' : 'bot-row');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar', sender === 'user' ? 'user-avatar' : 'bot-avatar');
    avatar.textContent = sender === 'user' ? 'H' : '🎬';

    const bubble = document.createElement('div');
    bubble.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

    // Render newlines and bold markdown
    bubble.innerHTML = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
    return row;
}

function appendTyping() {
    const row = document.createElement('div');
    row.classList.add('message-row', 'bot-row');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar', 'bot-avatar');
    avatar.textContent = '🎬';

    const bubble = document.createElement('div');
    bubble.classList.add('message', 'bot-message', 'typing-indicator');
    bubble.innerHTML = '<span></span><span></span><span></span>';

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
    return row;
}
