import pygame
import numpy as np
from scipy.integrate import odeint
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
from datetime import datetime, timezone, timedelta
from pysolar.solar import get_altitude
import json
import secrets
import math

# --- Simulated Blockchain Ledger (Global for simplicity in demo) ---
simulated_blockchain_ledger = {}

class EnergyCommSystem:
    """A system for simulating energy dynamics, encrypting messages, and minting energy credit tokens."""

    def __init__(self, latitude: float = 36.1699, longitude: float = -115.1398, utc_offset: int = -7,
                 user_id: str = "GamePlayer"):
        """Initialize the system with location and cryptographic parameters."""
        self.latitude = latitude
        self.longitude = longitude
        self.utc_offset = utc_offset
        self.update_timestamp()

        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.sigma_y = np.array([[0, -1j], [1j, 0]])
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.entropy_pool = get_random_bytes(32)

        self.ects = {}
        self.metadata_map = {}
        self.ect_counter = 0
        self.user_id = user_id
        self.total_energy_saved = 0
        self.energy_credits = {}

        # New: To store Elite Token details if minted
        self.elite_token_minted_details = None

    def update_timestamp(self):
        self.timestamp = int(datetime.now(timezone.utc).timestamp())
        self.timestamp_str = datetime.now(timezone.utc).astimezone(
            timezone(timedelta(hours=self.utc_offset))).strftime('%Y-%m-%d %H:%M:%S')

    def simulate_circuit(self, t: np.ndarray, surge_voltage: float = 10000) -> np.ndarray:
        """Simulate circuit dynamics for grounding a voltage surge."""
        R, C, R_ground = 50, 1e-6, 10
        def circuit_dynamics(V, t):
            return -(V / (R * C)) - (V / R_ground)
        return odeint(circuit_dynamics, surge_voltage, t).flatten()

    def simulate_signal(self, t: np.ndarray, initial_snr: float = 10) -> np.ndarray:
        """Simulate signal-to-noise ratio degradation over time."""
        distance_factor = 0.5
        interference = 0.1 * np.sin(100 * t)
        return initial_snr * np.exp(-distance_factor * t) + interference

    def generate_aes_key(self, params: np.ndarray) -> bytes:
        """Generate an AES-256 key from optimization parameters."""
        return hashlib.sha256(params.tobytes()).digest()

    def encrypt_message(self, key: bytes, message: str) -> tuple:
        """Encrypt a message using AES-256 in EAX mode."""
        if not message.endswith('.'):
            message += '.'
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(message.encode('utf-8'))
        return cipher.nonce, ciphertext, tag

    def decrypt_message(self, key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> tuple:
        """Decrypt a message and verify its integrity."""
        try:
            cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
            message = cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')
            return message, message.endswith('.')
        except ValueError as e:
            print(f"Decryption failed: {e}. Message tampered with or incorrect key/nonce/tag.")
            return None, False

    def optimize_params(self, final_voltage: float) -> np.ndarray:
        """Run classical optimization using a quantum-inspired Hamiltonian."""
        ZZ = np.kron(self.sigma_z, self.sigma_z)
        XI = np.kron(self.sigma_x, np.eye(2))
        IX = np.kron(np.eye(2), np.eye(2))
        hamiltonian = 1.0 * ZZ + 0.5 * XI + 0.5 * IX

        def ansatz_state(theta):
            state = np.zeros(4)
            # Fix: Access the scalar value from the array for trigonometric functions
            state[0] = np.cos(theta[0] / 2)
            state[3] = np.sin(theta[0] / 2)
            return state / np.linalg.norm(state)

        def objective(theta):
            state = ansatz_state(theta)
            return np.real(state.T @ hamiltonian @ state)

        params = np.array([0.0])
        learning_rate = 0.05
        for _ in range(200):
            grad = (objective(params + 0.01) - objective(params - 0.01)) / 0.02
            params -= learning_rate * grad
            if np.abs(grad) < 1e-4:
                break
        return params

    def calculate_energy_credit(self, final_voltage: float, final_snr: float,
                             solar_energy_input: float) -> int:
        """Calculate energy credits based on grounding, signal, and solar input."""
        sigma_sum = int(self.sigma_x.sum().real + self.sigma_y.sum().real + self.sigma_z.sum().real)
        entropy_factor = len(self.entropy_pool)
        grounding_energy = (10000 - final_voltage) * 0.001
        base_energy = entropy_factor * abs(sigma_sum)

        # Use the current timestamp string for calculation
        current_time_for_solar = datetime.fromtimestamp(self.timestamp, tz=timezone(timedelta(hours=self.utc_offset)))
        try:
            solar_altitude = get_altitude(self.latitude, self.longitude, current_time_for_solar)
            solar_factor = max(0, solar_altitude) / 90
        except ValueError:
            solar_factor = 0.0
            print("Warning: Could not calculate solar altitude. Solar factor set to 0.")

        return int(base_energy * 5 + grounding_energy + solar_energy_input * solar_factor)

    def mint_ect(self, message: str, receiver_id: str, solar_energy_input: float = 10) -> tuple:
        """Mint an Energy Credit Token with encrypted message and energy credits."""
        self.update_timestamp()
        t = np.linspace(0, 1, 100)
        voltages = self.simulate_circuit(t)
        snr_values = self.simulate_signal(t)
        final_voltage, final_snr = voltages[-1], snr_values[-1]

        params = self.optimize_params(final_voltage)
        aes_key = self.generate_aes_key(params)

        nonce, ciphertext, tag = self.encrypt_message(aes_key, message)
        encrypted_code_payload = base64.b64encode(nonce + ciphertext + tag).decode('utf-8')
        code_hash = hashlib.sha256(encrypted_code_payload.encode('utf-8')).hexdigest()

        trace_sum = int(self.sigma_x.trace().real + self.sigma_y.trace().real +
                       self.sigma_z.trace().real)
        token_id = int(hashlib.sha256(
            (str(trace_sum) + str(self.timestamp) + str(self.ect_counter) + self.user_id + receiver_id).encode()
        ).hexdigest(), 16) % 100000000

        energy_credit = self.calculate_energy_credit(final_voltage, final_snr, solar_energy_input)

        metadata = {
            'code_hash': code_hash,
            'timestamp': self.timestamp,
            'timestamp_str': self.timestamp_str,
            'description': 'Game Event ECT: Grounding Energy and Secure Communication',
            'energy_saved': energy_credit,
            'final_voltage': final_voltage,
            'final_snr': final_snr,
            'location': f'Lat: {self.latitude}, Lon: {self.longitude}',
            'optimal_params': params.tolist(),
            'sender_id': self.user_id,
            'receiver_id': receiver_id,
            'nonce_b64': base64.b64encode(nonce).decode('utf-8'),
            'tag_b64': base64.b64encode(tag).decode('utf-8')
        }

        global simulated_blockchain_ledger
        simulated_blockchain_ledger[token_id] = {
            'code_hash': code_hash,
            'energy_saved': energy_credit,
            'timestamp': self.timestamp,
            'sender_id': self.user_id,
            'receiver_id': receiver_id,
            'nonce_b64': base64.b64encode(nonce).decode('utf-8'),
            'tag_b64': base64.b64encode(tag).decode('utf-8')
        }

        self.ects[token_id] = {
            'encrypted_code_payload': encrypted_code_payload,
            'metadata': metadata,
            'energy_credit': energy_credit,
            'timestamp_str': self.timestamp_str
        }
        self.metadata_map[code_hash] = metadata
        self.energy_credits[self.user_id] = self.energy_credits.get(self.user_id, 0) + energy_credit
        self.total_energy_saved += energy_credit
        self.ect_counter += 1

        print(f"\n--- ECT Minted for Game Event ---")
        print(f"Minted ECT ID {token_id}: Energy Saved = {energy_credit} Wh, SNR = {final_snr:.2f} dB")
        print(f"Code Hash (On-Chain): {code_hash}")

        return token_id, encrypted_code_payload, aes_key, energy_credit, final_snr

    def attempt_mint_elite_ect(self, message: str, receiver_id: str, solar_energy_input: float,
                                final_voltage: float, final_snr: float, current_score: int) -> tuple:
        """
        Attempts to mint an Elite Energy Credit Token (ECT-E) based on extremely difficult conditions.
        Returns (token_id, encrypted_payload, decrypted_seed_phrase) if successful, otherwise (None, None, None).
        """
        self.update_timestamp()

        # Define Elite Token conditions
        # Condition 1: Extremely good game performance (e.g., final_voltage < 100, final_snr > 9.5, score > 900)
        performance_met = (final_voltage < 100 and final_snr > 9.5 and current_score > 900)

        # Condition 2: Specific solar altitude (e.g., peak solar noon for maximum power)
        current_time_for_solar = datetime.fromtimestamp(self.timestamp, tz=timezone(timedelta(hours=self.utc_offset)))
        solar_altitude = 0.0
        try:
            solar_altitude = get_altitude(self.latitude, self.longitude, current_time_for_solar)
        except ValueError:
            pass # Solar altitude calculation failed, solar_altitude remains 0.0

        # Peak solar for Las Vegas (adjust based on what time of day you'd like it to be special)
        # Using a wide range for demonstration purposes, narrow this down for true rarity
        solar_condition_met = (solar_altitude > 70.0 and solar_altitude <= 90.0) # Peak solar

        # Condition 3: Low probability roll
        prob_roll_success = (secrets.randbelow(500) == 0) # 1 in 500 chance, adjust for rarity

        print(f"DEBUG ELITE: Perf: {performance_met}, Solar: {solar_condition_met} ({solar_altitude:.2f}°), Prob: {prob_roll_success}")

        if performance_met and solar_condition_met and prob_roll_success:
            secret_seed_phrase = f"ELITE_MSG_UNLOCK_CODE_{secrets.token_hex(16)}"
            elite_message = f"ELITE TOKEN GRANTED! Messaging App Unlock Seed: {secret_seed_phrase}. " + message

            # Re-simulate dynamics for key generation based on *attempted* conditions
            t = np.linspace(0, 1, 100)
            # Use original surge_voltage for key derivation, not final_voltage.
            # Assuming 'surge_voltage' in simulate_circuit is the initial voltage.
            # The 'params' are optimized based on 'final_voltage' though.
            # Let's use a consistent value that's within a reasonable range for encryption key generation.
            # Using the optimal_params based on current final_voltage is correct for the key.
            
            params = self.optimize_params(final_voltage)
            aes_key = self.generate_aes_key(params)

            nonce, ciphertext, tag = self.encrypt_message(aes_key, elite_message)
            encrypted_code_payload = base64.b64encode(nonce + ciphertext + tag).decode('utf-8')
            code_hash = hashlib.sha256(encrypted_code_payload.encode('utf-8')).hexdigest()

            trace_sum = int(self.sigma_x.trace().real + self.sigma_y.trace().real +
                           self.sigma_z.trace().real)
            elite_token_id_base = int(hashlib.sha256(
                (str(trace_sum) + str(self.timestamp) + str(self.ect_counter) + self.user_id + receiver_id + "ELITE_SECRET").encode()
            ).hexdigest(), 16)
            elite_token_id = (elite_token_id_base % 100000000) | (1 << 29) # Set a high bit for distinction (makes it >= 536870912)

            elite_energy_credit = self.calculate_energy_credit(final_voltage, final_snr, solar_energy_input) * 10 # 10x multiplier

            elite_metadata = {
                'code_hash': code_hash,
                'timestamp': self.timestamp,
                'timestamp_str': self.timestamp_str,
                'description': 'ELITE ENERGY CREDIT TOKEN: Unlocks Secure Messaging App',
                'energy_saved': elite_energy_credit,
                'final_voltage': final_voltage,
                'final_snr': final_snr,
                'location': f'Lat: {self.latitude}, Lon: {self.longitude}',
                'optimal_params': params.tolist(),
                'sender_id': self.user_id,
                'receiver_id': receiver_id,
                'nonce_b64': base64.b64encode(nonce).decode('utf-8'),
                'tag_b64': base64.b64encode(tag).decode('utf-8'),
                'is_elite': True
            }

            global simulated_blockchain_ledger
            simulated_blockchain_ledger[elite_token_id] = {
                'code_hash': code_hash,
                'energy_saved': elite_energy_credit,
                'timestamp': self.timestamp,
                'sender_id': self.user_id,
                'receiver_id': receiver_id,
                'nonce_b64': base64.b64encode(nonce).decode('utf-8'),
                'tag_b64': base64.b64encode(tag).decode('utf-8'),
                'is_elite': True
            }

            self.ects[elite_token_id] = {
                'encrypted_code_payload': encrypted_code_payload,
                'metadata': elite_metadata,
                'energy_credit': elite_energy_credit,
                'timestamp_str': self.timestamp_str
            }
            self.metadata_map[code_hash] = elite_metadata
            self.energy_credits[self.user_id] = self.energy_credits.get(self.user_id, 0) + elite_energy_credit
            self.total_energy_saved += elite_energy_credit
            self.ect_counter += 1

            print(f"\n!!! ELITE ECT MINTED !!!")
            print(f"Elite ECT ID {elite_token_id}: Energy Saved = {elite_energy_credit} Wh, SNR = {final_snr:.2f} dB")
            print(f"Code Hash (On-Chain): {code_hash}")

            self.elite_token_minted_details = {
                'token_id': elite_token_id,
                'decrypted_seed': secret_seed_phrase,
                'encrypted_payload': encrypted_code_payload
            }
            return elite_token_id, encrypted_code_payload, secret_seed_phrase
        else:
            return None, None, None

    def redeem_credits(self, amount: int) -> None:
        """Redeem energy credits for the sender."""
        if self.energy_credits.get(self.user_id, 0) >= amount:
            self.energy_credits[self.user_id] -= amount
            self.total_energy_saved -= amount
            print(f"Redeemed {amount} Wh for {self.user_id}. Remaining: {self.energy_credits[self.user_id]} Wh")
        else:
            print("Error: Insufficient credits")

    def get_energy_credits(self, user: str) -> int:
        """Get the energy credit balance for a user."""
        return self.energy_credits.get(user, 0)

# --- Game Constants ---
MEME_FEEDBACK = {
    'great': ["Absolute chad energy—nailed it!", "Big brain move, chaos mastered!"],
    'good': ["Solid vibes, not too shabby!", "Kinda based, entropy’s friend!"],
    'bad': ["Total chaos, yeeted into the void!", "Mega oof, disorder wins!"],
    'overheat': ["Toasty vibes, it’s imploding!", "Spicy fail, too hot to handle!"],
    'level_up': ["Level up! You’re a chaos god!", "Dank ascension, keep it rollin’!"],
    'win': ["YOU WON! Chaos king crowned!", "Entropy bows to your dankness!"],
    'dank_spike': ["Dank spike—power’s lit!", "Dank spike—temp’s roasted!"]
}
DANK_SOUNDBITES = ["*boop*", "*yeet*", "*bruh*", "*womp*", "*vibes*"]

# --- Pygame Initialization ---
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Charging Game: Energy Comm")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)

# Fonts
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 28)

class ChargingGamePygame:
    def __init__(self, energy_comm_system_instance):
        self.energy_comm_system = energy_comm_system_instance
        self.target = secrets.randbelow(601) + 300
        self.player_power = 0
        self.score = 0
        self.temperature = 30
        self.holding = False
        self.hold_time = 0
        self.chaos_factor = self.calculate_chaos()
        self.target_jitter = 0
        self.level = 1
        self.level_targets = [100, 250, 500, 750, 1000] # Score needed to advance level
        self.is_game_over = False
        self.ect_counter_display = 0
        self.feedback_message = "Score: 0 | Level: 1 | Dankness awaits!"
        self.dank_spike_msg = None # To hold dank spike message temporarily

        # Define button rectangles here so they are properties of the instance
        self.start_btn_rect = pygame.Rect(50, SCREEN_HEIGHT - 60, (SCREEN_WIDTH - 150) / 2, 40)
        self.stop_btn_rect = pygame.Rect(50 + (SCREEN_WIDTH - 150) / 2 + 50, SCREEN_HEIGHT - 60, (SCREEN_WIDTH - 150) / 2, 40)

        # New: Elite token state for the game display
        self.elite_token_found = False
        self.elite_token_id = None
        self.elite_token_payload = None # This is the base64 encoded nonce+ciphertext+tag from the Elite Token
        self.elite_token_seed = None # This is the *decrypted* secret from the Elite Token payload


    def calculate_chaos(self):
        a, b = secrets.randbelow(10) + 1, secrets.randbelow(10) + 1
        c, d = secrets.randbelow(10) + 1, secrets.randbelow(10) + 1
        trace = a + d
        det = a * d - b * c
        discriminant = trace**2 - 4 * det
        if discriminant < 0:
            return 1.0 # Pure imaginary eigenvalues, stable chaos
        eig1 = (trace + math.sqrt(discriminant)) / 2
        return max(0.5, min(2.0, eig1 / 5)) # Keep chaos factor within a reasonable range

    def start_action(self):
        if self.is_game_over:
            self.reset_game()
            self.feedback_message = "Score: 0 | Level: 1 | Dankness awaits!"
            self.elite_token_found = False # Reset unlock prompt
            self.elite_token_id = None
            self.elite_token_seed = None

        if not self.holding:
            self.holding = True
            self.hold_time = 0
            self.chaos_factor = self.calculate_chaos()
            self.dank_spike_msg = None # Clear any old dank spike message

    def stop_action(self):
        if not self.holding or self.is_game_over:
            return None, (None, None, None) # Return nothing if not charging or game over

        self.holding = False
        diff = abs(self.target - self.player_power)
        chaos_boost = secrets.randbelow(21) - 10 # Random bonus/penalty

        meme = ""
        ect_info_str = ""
        
        # Approximate final_voltage and final_snr for ECT calculation.
        # In a real game, you might want more precise values from the simulation
        # that are directly tied to the player's performance.
        simulated_final_voltage = 10000 - self.player_power * 10 # Inverse relationship
        simulated_final_snr = self.player_power / 1000.0 * 10 # Direct relationship to power

        # Determine game feedback and score adjustment
        if self.dank_spike_msg:
            meme = self.dank_spike_msg
            self.dank_spike_msg = None # Clear after use
        elif self.temperature >= 80:
            self.score = max(0, self.score - 10 - chaos_boost)
            meme = secrets.choice(MEME_FEEDBACK['overheat'])
            self.is_game_over = True
        elif diff < 50:
            self.score += int((100 - diff) * self.chaos_factor) + chaos_boost
            meme = secrets.choice(MEME_FEEDBACK['great'])
        elif diff < 100:
            self.score += int((50 - diff // 2) * self.chaos_factor) + chaos_boost
            meme = secrets.choice(MEME_FEEDBACK['good'])
        else:
            self.score = max(0, self.score - 5 - chaos_boost)
            meme = secrets.choice(MEME_FEEDBACK['bad'])

        # Check for level up or win condition
        if self.level <= 5 and self.score >= self.level_targets[self.level - 1]:
            if self.level == 5:
                meme = secrets.choice(MEME_FEEDBACK['win'])
                self.score = 1000 # Max score for winning
                self.is_game_over = True
            else:
                self.level += 1
                meme = secrets.choice(MEME_FEEDBACK['level_up'])
                self.target = secrets.randbelow(601) + 300
                self.temperature = max(30, self.temperature - 10)

        # Reset for next round if not game over
        if not self.is_game_over:
            self.player_power = 0
            self.temperature = max(30, self.temperature - 5)
            self.target = secrets.randbelow(601) + 300
            self.target_jitter = 0
        else: # Game over state
            self.player_power = 0
            self.target_jitter = 0

        # --- Trigger ECT Minting based on game outcome ---
        elite_token_details = (None, None, None) # Initialize to no elite token found

        if self.score >= 0 and not (self.temperature >= 80 and self.is_game_over):
            # Attempt to mint Elite Token first
            elite_token_id, elite_payload, elite_seed = self.energy_comm_system.attempt_mint_elite_ect(
                message=f"Game Round {self.level}-{self.ect_counter_display+1} Result: {meme} (Score: {self.score})",
                receiver_id="BlockchainLedger_Elite",
                solar_energy_input=self.player_power / 100.0,
                final_voltage=simulated_final_voltage,
                final_snr=simulated_final_snr,
                current_score=self.score
            )

            if elite_token_id:
                ect_info_str = f"| !!!ELITE ECT!!! ID: {elite_token_id} | Saved: {self.energy_comm_system.ects[elite_token_id]['energy_credit']}Wh | SNR: {self.energy_comm_system.ects[elite_token_id]['metadata']['final_snr']:.2f}dB"
                self.ect_counter_display = self.energy_comm_system.ect_counter
                
                # Store elite token details in the game instance
                self.elite_token_found = True
                self.elite_token_id = elite_token_id
                self.elite_token_payload = elite_payload
                self.elite_token_seed = elite_seed
                
                elite_token_details = (elite_token_id, elite_payload, elite_seed) # Pass back for prompt

            else: # If no elite token, mint a regular one
                ect_message = f"Game Round {self.level}-{self.ect_counter_display+1} Result: {meme} (Score: {self.score})"
                try:
                    token_id, _, _, energy_saved_wh, snr_val = self.energy_comm_system.mint_ect(
                        message=ect_message, receiver_id="BlockchainLedger", solar_energy_input=self.player_power / 100.0
                    )
                    self.ect_counter_display = self.energy_comm_system.ect_counter
                    ect_info_str = f"| ECT ID: {token_id} | Saved: {energy_saved_wh}Wh | SNR: {snr_val:.2f}dB"
                except Exception as e:
                    ect_info_str = f"| ECT Minting Failed: {e}"
                    print(f"Error during ECT minting: {e}")

        soundbite = secrets.choice(DANK_SOUNDBITES)
        
        # Prepare the feedback message based on whether an elite token was found
        final_feedback_msg = f"Hold: {self.hold_time:.2f}s | Diff: {diff} | Chaos: {self.chaos_factor:.2f} | {meme} {soundbite} {ect_info_str}"
        if self.is_game_over:
            final_feedback_msg += "\nGAME OVER! Press Yeet to Restart."
        
        self.feedback_message = final_feedback_msg # Update game's internal feedback

        return final_feedback_msg, elite_token_details # Return feedback string and elite token status


    def reset_game(self):
        self.target = secrets.randbelow(601) + 300
        self.player_power = 0
        self.score = 0
        self.temperature = 30
        self.holding = False
        self.hold_time = 0
        self.chaos_factor = self.calculate_chaos()
        self.target_jitter = 0
        self.level = 1
        self.is_game_over = False
        self.ect_counter_display = 0
        self.feedback_message = "Score: 0 | Level: 1 | Dankness awaits!"
        self.dank_spike_msg = None
        self.elite_token_found = False
        self.elite_token_id = None
        self.elite_token_payload = None
        self.elite_token_seed = None


    def charge_up(self, dt):
        if self.is_game_over:
            return

        self.hold_time += dt
        self.player_power = min(1000, self.player_power + secrets.randbelow(26) + 5)
        self.temperature = min(100, self.temperature + secrets.randbelow(29) / 100 + 0.02) # Cap temperature at 100

        # Dank spike chance
        if secrets.randbelow(100) < 10 and self.hold_time > 0.5:
            if secrets.randbelow(2) == 0:
                self.player_power = min(1000, self.player_power * 2)
                self.dank_spike_msg = MEME_FEEDBACK['dank_spike'][0]
            else:
                self.temperature = min(100, self.temperature * 2) # Cap dank spike temp at 100
                self.dank_spike_msg = MEME_FEEDBACK['dank_spike'][1]
            self.feedback_message = f"Dank Spike! {self.dank_spike_msg}" # Immediately update feedback

        self.target_jitter = secrets.randbelow(21) - 10 # Jitter target

    def cool_down(self, dt):
        if not self.holding and not self.is_game_over:
            self.temperature = max(30, self.temperature - 0.5 * dt * 60) # Scale dt for consistent cool down rate

    def update_feedback_display(self):
        target_display = self.level_targets[self.level - 1] if self.level <= 5 else "WON!"
        
        # If a dank spike message is active and we are still holding, prioritize it
        if self.dank_spike_msg and self.holding:
            feedback_part = f" | Dank Spike! {self.dank_spike_msg}"
        else:
            # Otherwise, use the general feedback message
            feedback_part = f" | {self.feedback_message}" if self.feedback_message else ""
            # Only show general message if not game over and not currently dank spiking

        self.full_feedback_text = (
            f"Score: {self.score} | Level: {self.level} | "
            f"Temp: {self.temperature:.1f}°C | Target: {target_display}"
            + feedback_part
        )


    def draw(self, screen):
        screen.fill(BLACK) # Clear screen each frame

        # Target Line (Red)
        jittered_target_x = 100 + (self.target + self.target_jitter) / 1000 * (SCREEN_WIDTH - 200) # Scale target to screen width
        pygame.draw.line(screen, RED, (jittered_target_x, 100), (jittered_target_x, SCREEN_HEIGHT - 150), 2)

        # Player Power Dot (Blue)
        power_y_pos = SCREEN_HEIGHT - 150 - (self.player_power / 1000) * (SCREEN_HEIGHT - 250) # Scale power height
        pygame.draw.circle(screen, BLUE, (100 + secrets.randbelow(11) - 5, int(power_y_pos)), 15)

        # Temperature Bar (Green/Red)
        # Ensure temp_color values are clamped between 0 and 255
        # The temperature is clamped between 30 and 100.
        # Normalize temperature from [30, 100] to [0, 1]
        normalized_temp = (self.temperature - 30) / 70.0
        normalized_temp = max(0.0, min(1.0, normalized_temp)) # Clamp to ensure it's within 0-1

        red_val = 0
        green_val = 0

        # Transition from Green (0,255,0) to Yellow (255,255,0) to Red (255,0,0)
        if normalized_temp <= 0.5: # First half: Green to Yellow (0.0 to 0.5)
            # Red component increases from 0 to 255
            red_val = int(normalized_temp * 2 * 255)
            green_val = 255
        else: # Second half: Yellow to Red (0.5 to 1.0)
            # Green component decreases from 255 to 0
            red_val = 255
            green_val = int((1 - (normalized_temp - 0.5) * 2) * 255)
        
        # Ensure final color components are within 0-255
        temp_color = (max(0, min(255, red_val)), max(0, min(255, green_val)), 0)
        
        temp_bar_width = self.temperature * ((SCREEN_WIDTH - 100) / 100) # Scale to screen width
        pygame.draw.rect(screen, temp_color, (50, SCREEN_HEIGHT - 100, temp_bar_width, 20))


        # Buttons and Labels
        # Feedback Label
        text_surface = font.render(self.full_feedback_text, True, WHITE)
        screen.blit(text_surface, (10, 10)) # Top left

        # Instructions Label
        instruction_surface = small_font.render("Yeet juice to the shaky red line! Beat 5 levels, don’t overheat!", True, GREY)
        screen.blit(instruction_surface, (10, 50)) # Below feedback

        # Draw Buttons using their stored Rects
        pygame.draw.rect(screen, (0, 200, 0), self.start_btn_rect, border_radius=5) # Darker green for button
        pygame.draw.rect(screen, (200, 0, 0), self.stop_btn_rect, border_radius=5)   # Darker red for button

        start_text = font.render("Yeet the Juice", True, WHITE) # White text for better contrast
        stop_text = font.render("Drop the Dank", True, WHITE)   # White text for better contrast

        start_text_rect = start_text.get_rect(center=self.start_btn_rect.center)
        stop_text_rect = stop_text.get_rect(center=self.stop_btn_rect.center)

        screen.blit(start_text, start_text_rect)
        screen.blit(stop_text, stop_text_rect)

        pygame.display.flip() # Update the full display

# --- Main Game Loop ---
def run_game():
    energy_comm_system = EnergyCommSystem(user_id="GamePlayer") # Latitude/Longitude default to Las Vegas in class
    game = ChargingGamePygame(energy_comm_system)

    running = True
    clock = pygame.time.Clock()
    
    # Event for periodic updates like charging and cooling
    CHARGE_EVENT = pygame.USEREVENT + 1
    COOL_EVENT = pygame.USEREVENT + 2
    pygame.time.set_timer(CHARGE_EVENT, 50) # 50ms = 0.05s
    pygame.time.set_timer(COOL_EVENT, 1000) # 1000ms = 1s

    # Game State Flags
    show_unlock_prompt = False

    while running:
        dt = clock.tick(60) / 1000.0 # Delta time in seconds, capped at 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.start_btn_rect.collidepoint(event.pos):
                    game.start_action()
                    show_unlock_prompt = False # Hide prompt if starting new game/round
                elif game.stop_btn_rect.collidepoint(event.pos):
                    feedback_msg, elite_token_details = game.stop_action() # Get elite token info
                    # Update game's feedback message using the first return value
                    game.feedback_message = feedback_msg

                    if elite_token_details[0] is not None: # elite_token_id is not None
                        show_unlock_prompt = True
                        # Elite token details are already stored in game instance (game.elite_token_id, game.elite_token_seed)
                
                # Handle interaction with the "Continue" button on the unlock prompt
                if show_unlock_prompt:
                    continue_btn_rect = pygame.Rect(SCREEN_WIDTH // 2 - 75, SCREEN_HEIGHT - 100, 150, 40)
                    if continue_btn_rect.collidepoint(event.pos):
                        show_unlock_prompt = False # Hide the prompt

            elif event.type == CHARGE_EVENT and game.holding:
                game.charge_up(dt)
            elif event.type == COOL_EVENT:
                game.cool_down(dt)

        # Update feedback message every frame to reflect real-time changes
        game.update_feedback_display()
        game.draw(screen) # Draw the main game elements

        # --- Draw Unlock Prompt if an Elite Token was found ---
        if show_unlock_prompt:
            # Draw a translucent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)) # Black with 180 alpha (out of 255)
            screen.blit(overlay, (0,0))

            prompt_text = font.render("ELITE TOKEN FOUND! Secure Messaging App Unlocked!", True, WHITE)
            prompt_rect = prompt_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(prompt_text, prompt_rect)

            token_info_text = small_font.render(f"Token ID: {game.elite_token_id}", True, WHITE)
            token_info_rect = token_info_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(token_info_text, token_info_rect)

            seed_info_text = small_font.render(f"UNLOCK SEED: {game.elite_token_seed}", True, WHITE)
            seed_info_rect = seed_info_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
            screen.blit(seed_info_text, seed_info_rect)

            instruction_text = small_font.render("Copy these values to unlock the Secure Messenger.", True, GREY)
            instruction_rect = instruction_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
            screen.blit(instruction_text, instruction_rect)

            # "Continue" button
            continue_btn_rect = pygame.Rect(SCREEN_WIDTH // 2 - 75, SCREEN_HEIGHT - 100, 150, 40)
            pygame.draw.rect(screen, (0, 100, 200), continue_btn_rect, border_radius=5) # Blue button
            continue_text = font.render("Continue", True, WHITE)
            continue_text_rect = continue_text.get_rect(center=continue_btn_rect.center)
            screen.blit(continue_text, continue_text_rect)

            pygame.display.flip() # Update for prompt

    pygame.quit()
    print("Game Exited. Final Energy Credits:", energy_comm_system.get_energy_credits("GamePlayer"))
    print("Full Blockchain Ledger:", simulated_blockchain_ledger)

if __name__ == '__main__':
    run_game()
