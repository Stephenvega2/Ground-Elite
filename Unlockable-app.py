# secure_messenger_app.py
# This code is released under the MIT License.
# See LICENSE.txt for full details.

"""
MIT License

Copyright (c) 2025 stephenvega2 / Ground-Elite

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# app.py (Single file for Flask app with embedded HTML)

from flask import Flask, request, jsonify, render_template_string
import hashlib
import base64
from Crypto.Cipher import AES
import secrets # Used in EnergyCommSystem

# --- Import EnergyCommSystem and simulated_blockchain_ledger from your game file ---
# For this to work, you need to ensure from game.py exists and has been run
# to populate the global simulated_blockchain_ledger dictionary.
try:
    from game import simulated_blockchain_ledger, EnergyCommSystem
except ImportError:
    print("Warning: Could not import simulated_blockchain_ledger and EnergyCommSystem from game.py.")
    print("Please ensure game.py exists and has been run to populate the ledger for this demo,")
    print("or manually define a sample elite token in simulated_blockchain_ledger for testing.")
    # Fallback for testing Flask app without running game first
    simulated_blockchain_ledger = {
        # Example Elite Token for testing if game hasn't been run:
        # Replace with an actual token ID and values from your game run if possible
        536870912: { # Example elite token ID (high bit set) - must be a real token ID from your game
            'code_hash': 'some_hash_value_for_demo',
            'energy_saved': 1000,
            'timestamp': 1700000000,
            'sender_id': 'GamePlayer',
            'receiver_id': 'BlockchainLedger_Elite',
            'nonce_b64': base64.b64encode(b'some_nonce_bytes_for_demo').decode('utf-8'),
            'tag_b64': base64.b64encode(b'some_tag_bytes_for_demo').decode('utf-8'),
            'is_elite': True
        }
    }
    class EnergyCommSystem:
        def __init__(self):
            pass # Minimal init for key generation
        def generate_aes_key(self, seed_phrase: str) -> bytes:
            """Generates an AES key directly from a string seed phrase for the messaging app."""
            # Use SHA256 of the seed phrase to get a consistent 32-byte key
            return hashlib.sha256(seed_phrase.encode('utf-8')).digest()

        def encrypt_message(self, key: bytes, message: str) -> tuple:
            cipher = AES.new(key, AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(message.encode('utf-8'))
            return cipher.nonce, ciphertext, tag

        def decrypt_message(self, key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> tuple:
            try:
                cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
                message = cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')
                return message, message.endswith('.')
            except ValueError:
                return None, False


app = Flask(__name__)
# Global instance of our messaging system to maintain state
messaging_system_backend = {
    "is_unlocked": False,
    "aes_key": None,
    "messages": []
}
energy_comm_instance = EnergyCommSystem() # Instance to use crypto methods


# --- Embedded HTML Template ---
# Using a multiline string for the HTML content
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Messaging App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #282c34;
            color: #ffffff;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background-color: #3a3f47;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 500px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        h1 {
            text-align: center;
            color: #61dafb;
            margin-bottom: 20px;
        }
        .section {
            border: 1px solid #444;
            border-radius: 5px;
            padding: 15px;
            background-color: #2c3038;
        }
        input[type="text"], input[type="number"] {
            width: calc(100% - 20px);
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #3e4451;
            color: #eee;
            font-size: 1em;
        }
        input[type="text"]::placeholder, input[type="number"]::placeholder {
            color: #aaa;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #61dafb;
            color: #333;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #21a1f1;
        }
        button:disabled {
            background-color: #555;
            cursor: not-allowed;
        }
        #unlockStatus {
            text-align: center;
            font-weight: bold;
            color: red; /* Default locked color */
            margin-top: 10px;
        }
        #messageLog {
            height: 200px;
            overflow-y: auto;
            border: 1px solid #555;
            padding: 10px;
            border-radius: 4px;
            background-color: #3e4451;
            margin-bottom: 10px;
            word-wrap: break-word; /* Ensure long words wrap */
        }
        .message {
            margin-bottom: 8px;
            padding: 5px;
            border-radius: 3px;
        }
        .message.you { color: white; }
        .message.system { color: #00ff00; } /* Green */
        .message.echo { color: #00aaff; } /* Blue */
        .message.error { color: #ff0000; } /* Red */
    </style>
</head>
<body>
    <div class="container">
        <h1>Secure Messaging App</h1>

        <div class="section unlock-section">
            <h2>Unlock Messaging</h2>
            <input type="number" id="tokenIdInput" placeholder="Elite Token ID">
            <input type="text" id="seedInput" placeholder="Unlock Seed Phrase">
            <button id="unlockButton">Unlock Messaging</button>
            <div id="unlockStatus">Status: Locked</div>
        </div>

        <div class="section messaging-section">
            <h2>Message Log</h2>
            <div id="messageLog"></div>
            <input type="text" id="messageInput" placeholder="Type your message here..." disabled>
            <button id="sendMessageButton" disabled>Send Encrypted Message</button>
        </div>
    </div>

    <script>
        const tokenIdInput = document.getElementById('tokenIdInput');
        const seedInput = document.getElementById('seedInput');
        const unlockButton = document.getElementById('unlockButton');
        const unlockStatus = document.getElementById('unlockStatus');
        const messageLog = document.getElementById('messageLog');
        const messageInput = document.getElementById('messageInput');
        const sendMessageButton = document.getElementById('sendMessageButton');

        let isUnlocked = false;

        function updateUIState() {
            messageInput.disabled = !isUnlocked;
            sendMessageButton.disabled = !isUnlocked;
            if (isUnlocked) {
                unlockStatus.textContent = 'Status: Unlocked!';
                unlockStatus.style.color = 'lightgreen';
                tokenIdInput.disabled = true;
                seedInput.disabled = true;
                unlockButton.disabled = true; // Disable unlock button after successful unlock
            } else {
                unlockStatus.textContent = 'Status: Locked';
                unlockStatus.style.color = 'red';
                tokenIdInput.disabled = false;
                seedInput.disabled = false;
                unlockButton.disabled = false;
            }
        }

        async function fetchMessages() {
            const response = await fetch('/get_messages');
            const data = await response.json();
            messageLog.innerHTML = ''; // Clear current log
            data.messages.forEach(msg => {
                const p = document.createElement('p');
                p.classList.add('message');
                // Assign specific classes for styling based on sender/type
                if (msg.sender === 'You') p.classList.add('you');
                else if (msg.sender === 'System') p.classList.add('system');
                else if (msg.sender.includes('Decrypted Echo')) p.classList.add('echo');
                else if (msg.sender.includes('failed')) p.classList.add('error'); // For errors from simulation

                p.textContent = `${msg.sender}: ${msg.message}`;
                messageLog.appendChild(p);
            });
            // Scroll to bottom
            messageLog.scrollTop = messageLog.scrollHeight;
        }

        unlockButton.addEventListener('click', async () => {
            const tokenId = tokenIdInput.value;
            const seedPhrase = seedInput.value;

            if (!tokenId || !seedPhrase) {
                alert("Please enter both Token ID and Seed Phrase.");
                return;
            }

            try {
                const response = await fetch('/unlock', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ tokenId, seedPhrase })
                });

                const result = await response.json();
                alert(result.message); // Show a popup alert

                if (response.ok) {
                    isUnlocked = true;
                    fetchMessages(); // Fetch initial system messages
                } else {
                    isUnlocked = false;
                }
                updateUIState();
            } catch (error) {
                console.error('Unlock error:', error);
                alert("An error occurred during unlock attempt.");
                isUnlocked = false;
                updateUIState();
            }
        });

        sendMessageButton.addEventListener('click', async () => {
            const message = messageInput.value;
            if (!message) {
                alert("Message cannot be empty.");
                return;
            }

            try {
                const response = await fetch('/send_message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message })
                });

                const result = await response.json();
                if (!response.ok) {
                    alert(`Error sending message: ${result.message}`);
                }
                messageInput.value = ''; // Clear input field
                fetchMessages(); // Refresh message log
            } catch (error) {
                console.error('Send message error:', error);
                alert("An error occurred while sending the message.");
            }
        });

        // Initial UI state setup and periodic message fetch
        updateUIState();
        // Fetch messages periodically to simulate real-time updates (optional, but good for demo)
        setInterval(fetchMessages, 2000); // Fetch messages every 2 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serves the main HTML page for the messaging app directly from a string."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/unlock', methods=['POST'])
def unlock_messaging():
    """API endpoint to attempt unlocking the messaging system."""
    data = request.get_json()
    token_id_str = data.get('tokenId')
    seed_phrase = data.get('seedPhrase', '').strip()

    if not token_id_str or not seed_phrase:
        return jsonify({"status": "error", "message": "Missing Token ID or Seed Phrase."}), 400

    try:
        token_id = int(token_id_str)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid Token ID format."}), 400

    # Verify token on the simulated blockchain ledger
    token_data = simulated_blockchain_ledger.get(token_id)

    if token_data and token_data.get('is_elite') and token_data.get('sender_id') == "GamePlayer":
        # Generate AES key from the provided seed phrase
        messaging_system_backend["aes_key"] = energy_comm_instance.generate_aes_key(seed_phrase)
        messaging_system_backend["is_unlocked"] = True
        messaging_system_backend["messages"].append({"sender": "System", "message": "Messaging unlocked! Start chatting securely.", "color": "green"})
        return jsonify({"status": "success", "message": "Messaging App Unlocked!"})
    else:
        messaging_system_backend["is_unlocked"] = False
        messaging_system_backend["aes_key"] = None
        return jsonify({"status": "error", "message": "Unlock Failed: Invalid Elite Token ID or not an Elite Token."}), 403

@app.route('/send_message', methods=['POST'])
def send_message():
    """API endpoint to send an encrypted message."""
    if not messaging_system_backend["is_unlocked"]:
        return jsonify({"status": "error", "message": "Messaging is locked. Please unlock first."}), 401

    data = request.get_json()
    message_text = data.get('message', '').strip()

    if not message_text:
        return jsonify({"status": "error", "message": "Message cannot be empty."}), 400

    try:
        # Encrypt the message
        nonce, ciphertext, tag = energy_comm_instance.encrypt_message(messaging_system_backend["aes_key"], message_text)

        # For demonstration, simulate receiving and decrypting it immediately
        # In a real app, you'd send nonce, ciphertext, tag over network
        # and a recipient would decrypt with their shared key.
        decrypted_message, integrity_ok = energy_comm_instance.decrypt_message(
            messaging_system_backend["aes_key"], nonce, ciphertext, tag
        )

        if integrity_ok:
            messaging_system_backend["messages"].append({"sender": "You", "message": message_text, "color": "white"})
            messaging_system_backend["messages"].append({"sender": "Decrypted Echo (Self)", "message": decrypted_message, "color": "blue"})
            return jsonify({"status": "success", "message": "Message sent and simulated decryption successful!"})
        else:
            messaging_system_backend["messages"].append({"sender": "System", "message": "Message failed decryption/integrity check after sending (simulated).", "color": "red"})
            return jsonify({"status": "error", "message": "Message encryption/decryption failed."}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": f"Encryption/Decryption Error: {e}"}), 500

@app.route('/get_messages', methods=['GET'])
def get_messages():
    """API endpoint to retrieve the current message log."""
    return jsonify({"messages": messaging_system_backend["messages"]})

if __name__ == '__main__':
    # Make sure main_game.py is in the same directory and has been run
    # to populate the simulated_blockchain_ledger
    app.run(debug=True)
