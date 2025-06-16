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

import pygame
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import json
import secrets

# --- Constants for Messaging App ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
LIGHT_GREY = (200, 200, 200)
DARK_BLUE = (0, 50, 100)
GREEN_SUCCESS = (0, 180, 0)
RED_FAIL = (180, 0, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Secure Messenger (Elite Token Unlock)")
font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

# Mocked Secure Communication (in a real app, this would be network communication)
# For this demo, it's just local storage.
mock_inbox = []
mock_outbox = []

def generate_aes_key_from_seed(seed: str) -> bytes:
    """Generate AES key from the Elite Token seed."""
    return hashlib.sha256(seed.encode('utf-8')).digest()

def encrypt_message_for_messenger(key: bytes, message: str, sender_id: str) -> tuple:
    """Encrypts a message for the secure messenger."""
    full_message = f"[{sender_id}] {message}"
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(full_message.encode('utf-8'))
    return cipher.nonce, ciphertext, tag

def decrypt_message_for_messenger(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> str:
    """Decrypts a message for the secure messenger."""
    try:
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        message = cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')
        return message
    except ValueError:
        return "[Decryption Failed: Tampered or Incorrect Key]"

class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = WHITE
        self.text = text
        self.txt_surface = font.render(text, True, BLACK)
        self.active = False
        self.blink_counter = 0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = LIGHT_GREY if self.active else WHITE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = font.render(self.text, True, BLACK)

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2) # Draw outline
        # Draw blinking cursor
        if self.active:
            self.blink_counter = (self.blink_counter + 1) % 60 # ~1 second cycle at 60 FPS
            if self.blink_counter < 30: # Show cursor for half the cycle
                cursor_x = self.rect.x + self.txt_surface.get_width() + 7
                pygame.draw.line(screen, BLACK, (cursor_x, self.rect.y + 5), (cursor_x, self.rect.y + self.rect.height - 5), 2)

class SecureMessengerApp:
    def __init__(self):
        self.unlocked = False
        self.token_id_input = InputBox(SCREEN_WIDTH // 2 - 150, 150, 300, 40, 'Enter Elite Token ID')
        self.seed_input = InputBox(SCREEN_WIDTH // 2 - 150, 200, 300, 40, 'Enter UNLOCK SEED')
        self.unlock_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 75, 260, 150, 40)
        self.status_message = ""
        self.status_color = WHITE
        
        self.current_user = "Player1" # Placeholder for sender in the messenger
        self.message_input = InputBox(50, SCREEN_HEIGHT - 80, SCREEN_WIDTH - 200, 40, 'Type your message...')
        self.send_button_rect = pygame.Rect(SCREEN_WIDTH - 140, SCREEN_HEIGHT - 80, 100, 40)
        self.messages_display = [] # Stores (decrypted_text, original_encrypted_payload)

        self.encryption_key = None # Derived from the seed

    def authenticate(self):
        try:
            entered_token_id = int(self.token_id_input.text.strip())
            entered_seed = self.seed_input.text.strip()

            # For demo, we rely on the user having the correct token_id and seed.
            # In a real system, you'd verify this against a server/blockchain.
            # For this demo, let's just make sure they're not empty and match the expected format
            # A truly secure check would involve a cryptographic proof or actual lookup.
            
            # Since we printed the correct elite_token_id and elite_token_seed in the game,
            # we'll assume matching them locally is enough for this demonstration.
            # In a real scenario, the messaging app would *query the blockchain* with token_id
            # to retrieve its encrypted_payload, then attempt to decrypt using the provided seed.
            # If decryption works and contains a valid structure, then it's unlocked.

            # For this demo, we'll simplify: if they provide a seed starting with "ELITE_MSG_UNLOCK_CODE_"
            # and a reasonable token ID, we consider it "unlocked".
            
            # This is a *major simplification* for demo purposes.
            # A real system would involve a blockchain lookup for the token_id,
            # fetching its associated encrypted payload, and then *decrypting that payload*
            # with the user's entered seed to see if it reveals a valid "unlock" phrase.
            
            if entered_seed.startswith("ELITE_MSG_UNLOCK_CODE_") and entered_token_id > 0: # Basic validation
                self.unlocked = True
                self.status_message = "ACCESS GRANTED! Welcome to Secure Messenger."
                self.status_color = GREEN_SUCCESS
                self.encryption_key = generate_aes_key_from_seed(entered_seed)
            else:
                self.status_message = "ACCESS DENIED: Invalid Token ID or UNLOCK SEED."
                self.status_color = RED_FAIL
                self.unlocked = False
                self.encryption_key = None

        except ValueError:
            self.status_message = "Invalid Token ID format. Must be a number."
            self.status_color = RED_FAIL
            self.unlocked = False
            self.encryption_key = None

    def send_message(self):
        if not self.unlocked or not self.encryption_key:
            return

        message_text = self.message_input.text.strip()
        if not message_text or message_text == 'Type your message...':
            return

        try:
            nonce, ciphertext, tag = encrypt_message_for_messenger(self.encryption_key, message_text, self.current_user)
            encrypted_payload = base64.b64encode(nonce + ciphertext + tag).decode('utf-8')
            
            # In a real app, this encrypted_payload would be sent over a secure channel
            # For demo, we just add it to a mock inbox/outbox.
            mock_outbox.append({
                'sender': self.current_user,
                'encrypted_payload': encrypted_payload,
                'timestamp': pygame.time.get_ticks() # Milliseconds since pygame.init()
            })
            
            # Immediately decrypt and add to display for sender's view
            self.messages_display.append((f"You: {message_text}", encrypted_payload))
            self.message_input.text = '' # Clear input box
            self.message_input.txt_surface = font.render(self.message_input.text, True, BLACK) # Update surface
            self.status_message = "Message sent securely!"
            self.status_color = GREEN_SUCCESS
            print(f"Message Sent: {encrypted_payload}")

        except Exception as e:
            self.status_message = f"Failed to send message: {e}"
            self.status_color = RED_FAIL
            print(f"Error sending message: {e}")

    def fetch_and_decrypt_messages(self):
        """Mock function to simulate receiving and decrypting messages."""
        if not self.unlocked or not self.encryption_key:
            return

        # For demo, simulate a 'received' message every few seconds
        if secrets.randbelow(300) == 0: # Roughly every 5 seconds at 60 FPS
            mock_encrypted_message = None
            if mock_outbox:
                # Simulate receiving the last sent message from someone else
                last_sent = mock_outbox[-1]
                # To make it feel like another user, change sender ID
                mock_encrypted_message = encrypt_message_for_messenger(
                    self.encryption_key, f"Echo: {last_sent['sender']} said '{decrypt_message_for_messenger(self.encryption_key, *[base64.b64decode(last_sent['encrypted_payload'])[i:j] for i,j in [(0,16),(16,-16),(-16,None)]])}'", "EchoBot"
                )
            else:
                # If no messages sent yet, send a generic welcome
                mock_encrypted_message = encrypt_message_for_messenger(
                    self.encryption_key, "Welcome! This is a secure channel.", "System"
                )
            
            if mock_encrypted_message:
                nonce, ciphertext, tag = mock_encrypted_message
                received_payload_b64 = base64.b64encode(nonce + ciphertext + tag).decode('utf-8')
                
                # Decrypt and add to display
                decrypted_text = decrypt_message_for_messenger(self.encryption_key, nonce, ciphertext, tag)
                self.messages_display.append((decrypted_text, received_payload_b64))
                self.status_message = "New message received!"
                self.status_color = LIGHT_GREY
                print(f"Message Received: {received_payload_b64}")


    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if not self.unlocked:
                    self.token_id_input.handle_event(event)
                    self.seed_input.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if self.unlock_button_rect.collidepoint(event.pos):
                            self.authenticate()
                else: # Messaging mode
                    self.message_input.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if self.send_button_rect.collidepoint(event.pos):
                            self.send_message()

            screen.fill(DARK_BLUE) # Dark background for messenger

            if not self.unlocked:
                title_text = font.render("Secure Messenger - Elite Token Access Required", True, WHITE)
                title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 80))
                screen.blit(title_text, title_rect)

                self.token_id_input.draw(screen)
                self.seed_input.draw(screen)

                pygame.draw.rect(screen, (0, 100, 200), self.unlock_button_rect, border_radius=5)
                unlock_text = font.render("Unlock", True, WHITE)
                unlock_text_rect = unlock_text.get_rect(center=self.unlock_button_rect.center)
                screen.blit(unlock_text, unlock_text_rect)

                status_surface = small_font.render(self.status_message, True, self.status_color)
                status_rect = status_surface.get_rect(center=(SCREEN_WIDTH // 2, 320))
                screen.blit(status_surface, status_rect)

            else: # Messaging interface
                self.fetch_and_decrypt_messages() # Simulate incoming messages

                # Message display area
                msg_area_rect = pygame.Rect(20, 70, SCREEN_WIDTH - 40, SCREEN_HEIGHT - 170)
                pygame.draw.rect(screen, BLACK, msg_area_rect)
                pygame.draw.rect(screen, LIGHT_GREY, msg_area_rect, 1) # Border

                y_offset = 0
                for i, (msg_text, _) in enumerate(reversed(self.messages_display)): # Show latest messages at bottom
                    msg_surface = small_font.render(msg_text, True, WHITE)
                    if msg_area_rect.y + msg_area_rect.height - 10 - y_offset - msg_surface.get_height() < msg_area_rect.y + 10:
                        break # Stop if messages go outside bounds
                    screen.blit(msg_surface, (msg_area_rect.x + 10, msg_area_rect.y + msg_area_rect.height - 10 - y_offset - msg_surface.get_height()))
                    y_offset += msg_surface.get_height() + 5

                # Input box and Send button
                self.message_input.draw(screen)
                pygame.draw.rect(screen, (0, 150, 0), self.send_button_rect, border_radius=5)
                send_text = font.render("Send", True, WHITE)
                send_text_rect = send_text.get_rect(center=self.send_button_rect.center)
                screen.blit(send_text, send_text_rect)

                # Status message for messenger
                status_surface = small_font.render(self.status_message, True, self.status_color)
                screen.blit(status_surface, (20, 30)) # Top left

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        print("Secure Messenger Exited.")

if __name__ == '__main__':
    app = SecureMessengerApp()
    app.run()
