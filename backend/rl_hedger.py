# backend/rl_hedger.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout # LSTM not used in _build_model, can remove if not planned
from tensorflow.keras.optimizers import Adam # Ensure tf.keras.optimizers.legacy.Adam for newer TF if needed
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import pickle
import os
import time # Added for timestamp in hedge execution

from backend.portfolio_hedger import PortfolioHedger, HedgeExecution # Import HedgeExecution if used
from backend import config
from backend.utils import setup_logger
# --- ADD IMPORT FOR TYPE HINTING ---
from backend.advanced_pricing_engine import AdvancedPricingEngine # Assuming this is the correct path and class name
# --- END ADD IMPORT ---

logger = setup_logger(__name__)

# DQNAgent class remains the same as you provided.
# Minor detail: in _build_model, Dense(64, input_dim=...) is for direct vector input.
# If state becomes a sequence (e.g., for LSTM), input_shape would be (timesteps, features).
class DQNAgent:
    """Deep Q-Network agent for hedging decisions."""
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001, gamma_discount: float = 0.95): # Added gamma_discount
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=config.get_config_value('RL_MEMORY_MAXLEN', 10000))
        self.epsilon = config.get_config_value('RL_EPSILON_START', 1.0)
        self.epsilon_min = config.get_config_value('RL_EPSILON_MIN', 0.01)
        self.epsilon_decay = config.get_config_value('RL_EPSILON_DECAY', 0.995)
        self.learning_rate = learning_rate
        self.gamma = gamma_discount # Discount factor for future rewards
        self.batch_size = config.get_config_value('RL_BATCH_SIZE', 32)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        logger.info(f"DQNAgent initialized. State size: {state_size}, Action size: {action_size}")

    def _build_model(self) -> Sequential:
        """Build DQN model architecture."""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_initializer='he_uniform'),
            Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform') # Linear for Q-values
        ])
        # Consider using tf.keras.optimizers.legacy.Adam with TF > 2.6 if Adam itself gives issues.
        model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate)) # Huber loss can be more robust
        return model

    def update_target_model(self):
        """Update target model weights."""
        self.target_model.set_weights(self.model.get_weights())
        # logger.debug("Target model updated.")

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Ensure state is correctly shaped for predict (batch_size, state_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self) -> float:
        """Train the model on a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            return 0.0 # Not enough samples to train

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Predict Q-values for current states and next states
        current_q_matrix = self.model.predict(states, verbose=0)
        next_q_matrix_target_net = self.target_model.predict(next_states, verbose=0) # Use target network for stability

        # Create target Q-values
        target_q_matrix = current_q_matrix.copy()

        for i in range(self.batch_size):
            if dones[i]:
                target_q_matrix[i, actions[i]] = rewards[i]
            else:
                # Q(s,a) = r + gamma * max_a'(Q_target(s',a'))
                target_q_matrix[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_matrix_target_net[i])
        
        # Train the main model
        history = self.model.fit(states, target_q_matrix, epochs=1, verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

    def save(self, filepath_prefix: str): # Changed to filepath_prefix
        """Save model weights and agent state."""
        try:
            self.model.save_weights(f"{filepath_prefix}_model_weights.weights.h5") # Save weights only
            agent_state = {
                'epsilon': self.epsilon,
                'memory': list(self.memory) # Convert deque to list for pickling
            }
            with open(f"{filepath_prefix}_agent_state.pkl", 'wb') as f:
                pickle.dump(agent_state, f)
            logger.info(f"DQNAgent model weights and state saved to prefix {filepath_prefix}")
        except Exception as e:
            logger.error(f"Error saving DQNAgent: {e}", exc_info=True)

    def load(self, filepath_prefix: str): # Changed to filepath_prefix
        """Load model weights and agent state."""
        try:
            model_weights_path = f"{filepath_prefix}_model_weights.weights.h5"
            agent_state_path = f"{filepath_prefix}_agent_state.pkl"

            if os.path.exists(model_weights_path):
                self.model.load_weights(model_weights_path)
                self.update_target_model() # Sync target model
                logger.info(f"DQNAgent model weights loaded from {model_weights_path}")
            else:
                logger.warning(f"DQNAgent model weights not found at {model_weights_path}")

            if os.path.exists(agent_state_path):
                with open(agent_state_path, 'rb') as f:
                    agent_state = pickle.load(f)
                self.epsilon = agent_state.get('epsilon', self.epsilon)
                # Re-initialize deque with loaded memory to ensure maxlen is respected
                loaded_memory = agent_state.get('memory', [])
                self.memory = deque(loaded_memory, maxlen=self.memory.maxlen or config.get_config_value('RL_MEMORY_MAXLEN', 10000))
                logger.info(f"DQNAgent state (epsilon, memory) loaded from {agent_state_path}")
            else:
                logger.warning(f"DQNAgent state file not found at {agent_state_path}")
        except Exception as e:
            logger.error(f"Error loading DQNAgent: {e}", exc_info=True)


class RLHedger(PortfolioHedger):
    """Reinforcement Learning enhanced portfolio hedger."""

    # --- MODIFIED __init__ ---
    def __init__(self, pricing_engine: AdvancedPricingEngine): # Expect pricing_engine
        super().__init__(pricing_engine=pricing_engine) # Pass it to the parent
    # --- END MODIFICATION ---
        
        # RL specific attributes
        self.state_size = 15 # Number of state features as defined in get_current_state
        self.action_size = 7 # 7 possible hedging actions
        # Use learning rate from config or a default
        rl_learning_rate = getattr(config, 'RL_LEARNING_RATE', 0.001)
        self.agent = DQNAgent(self.state_size, self.action_size, learning_rate=rl_learning_rate)
        
        # Action mapping: [no_hedge, small_buy, medium_buy, large_buy, small_sell, medium_sell, large_sell]
        # These are BTC amounts to adjust the current hedge position by
        self.action_mapping_btc_adjustment: Dict[int, float] = {
            0: 0.0,   # No change to hedge
            1: getattr(config, 'RL_HEDGE_ACTION_SMALL_BTC', 0.05),   # Buy small BTC
            2: getattr(config, 'RL_HEDGE_ACTION_MEDIUM_BTC', 0.15),  # Buy medium BTC
            3: getattr(config, 'RL_HEDGE_ACTION_LARGE_BTC', 0.30),   # Buy large BTC
            4: -getattr(config, 'RL_HEDGE_ACTION_SMALL_BTC', 0.05),  # Sell small BTC
            5: -getattr(config, 'RL_HEDGE_ACTION_MEDIUM_BTC', 0.15), # Sell medium BTC
            6: -getattr(config, 'RL_HEDGE_ACTION_LARGE_BTC', 0.30)   # Sell large BTC
        }
        
        # Experience tracking
        self.last_state_np: Optional[np.ndarray] = None # Store as numpy array
        self.last_action_idx: Optional[int] = None       # Store action index
        self.last_portfolio_pnl_usd: float = 0.0         # Store total P&L for reward calc
        self.last_effective_delta_btc: float = 0.0       # Store effective delta for risk reduction reward
        
        self.training_enabled = getattr(config, 'RL_TRAINING_ENABLED', True)
        self.model_save_path_prefix = getattr(config, 'RL_MODEL_SAVE_PATH_PREFIX', "models/rl_hedger_live")

        # Performance tracking deques
        self.episode_rewards = deque(maxlen=1000) # Assuming an "episode" is one decision step for now
        self.training_losses = deque(maxlen=1000) # Store loss from agent.replay()
        
        logger.info("RLHedger initialized.")
        self.load_rl_model(self.model_save_path_prefix) # Attempt to load pre-trained model

    def get_current_state(self, current_btc_price: float) -> np.ndarray:
        """Extracts and normalizes state features for the RL agent."""
        # Ensure risk_metrics are up-to-date before extracting state
        current_risk_metrics = self.calculate_portfolio_risk() # This updates self.risk_metrics

        # Features for the state vector (ensure order and normalization are consistent)
        # All features should be scaled to a similar range (e.g., -1 to 1, or 0 to 1, or small numbers)
        
        norm_factor_delta = getattr(config, 'RL_NORM_FACTOR_DELTA', 10.0) # Max expected delta (BTC)
        norm_factor_gamma = getattr(config, 'RL_NORM_FACTOR_GAMMA', 5.0)  # Max expected gamma
        norm_factor_theta = getattr(config, 'RL_NORM_FACTOR_THETA', 200.0) # Max expected theta (USD/day)
        norm_factor_vega = getattr(config, 'RL_NORM_FACTOR_VEGA', 100.0)   # Max expected vega (USD/1% vol)
        norm_factor_exposure = getattr(config, 'RL_NORM_FACTOR_EXPOSURE', 200000.0) # Max USD exposure
        norm_factor_hedge_pos = getattr(config, 'RL_NORM_FACTOR_HEDGE_POS', 5.0)    # Max hedge BTC pos
        norm_factor_pnl = getattr(config, 'RL_NORM_FACTOR_PNL', 5000.0)           # Max PNL swing
        norm_factor_price = getattr(config, 'RL_NORM_FACTOR_PRICE', 150000.0)      # Max BTC price expected
        norm_factor_num_pos = getattr(config, 'RL_NORM_FACTOR_NUM_POSITIONS', 50.0) # Max number of positions

        # Raw values
        raw_delta_options = current_risk_metrics.net_delta
        raw_current_hedge = self.hedge_btc_position
        raw_effective_delta = raw_delta_options + raw_current_hedge

        features = [
            # Portfolio risk metrics (normalized and capped)
            np.clip(raw_effective_delta / norm_factor_delta, -2.0, 2.0), # Effective total delta
            np.clip(current_risk_metrics.net_gamma / norm_factor_gamma, -2.0, 2.0),
            np.clip(current_risk_metrics.net_theta / norm_factor_theta, -2.0, 2.0),
            np.clip(current_risk_metrics.net_vega / norm_factor_vega, -2.0, 2.0),
            np.clip(current_risk_metrics.total_exposure_usd / norm_factor_exposure, 0, 5.0),
            
            # Current hedge position (normalized and capped)
            np.clip(self.hedge_btc_position / norm_factor_hedge_pos, -2.0, 2.0),
            np.sign(self.hedge_btc_position) if self.hedge_btc_position != 0 else 0.0, # Hedge direction (-1, 0, 1)
            
            # Recent performance (normalized P&L since last step - needs careful handling)
            # For now, using total P&L to avoid statefulness issues here, reward function handles change
            np.clip(current_risk_metrics.option_pnl / norm_factor_pnl, -2.0, 2.0),
            np.clip(current_risk_metrics.hedging_pnl / norm_factor_pnl, -2.0, 2.0),
            
            # Market context
            np.clip(current_btc_price / norm_factor_price, 0.1, 2.0), # Normalized BTC price
            
            # Time-based / Portfolio structure features
            np.clip(current_risk_metrics.total_positions / norm_factor_num_pos, 0, 2.0), # Number of positions
            
            # Volatility proxy (from VolatilityEngine, if accessible, or use a simpler proxy)
            # self.pricing_engine.vol_engine.get_current_volatility_metric() # Example
            # For now, a placeholder:
            0.5, # Placeholder for normalized volatility metric

            # Time to next major expiry (e.g. daily, weekly - requires more complex logic)
            0.5, # Placeholder

            # Recent hedge cost normalized (e.g. cost of last hedge / typical trade size)
            np.clip(self.risk_metrics.hedge_executions[-1].hedge_cost_usd / (current_btc_price * 0.1) if self.risk_metrics.hedge_executions else 0, -2.0, 2.0),
            
            # Previous action's immediate reward (normalized) - tricky, use current PnL components instead.
            np.tanh(self.last_portfolio_pnl_usd / norm_factor_pnl) if self.last_state_np is not None else 0.0
        ]
        
        # Ensure state_size matches the number of features
        if len(features) != self.state_size:
            logger.error(f"State size mismatch! Expected {self.state_size}, got {len(features)}. Adjust RLHedger.state_size or feature list.")
            # Fallback to a zero vector of correct size to prevent crash, but this indicates a config error
            return np.zeros(self.state_size, dtype=np.float32)
            
        return np.array(features, dtype=np.float32)

    def calculate_reward(self, old_effective_delta_btc: float, new_effective_delta_btc: float,
                         pnl_change_usd: float, hedge_cost_usd: float, action_taken_idx: int) -> float:
        """Calculates reward for the RL agent based on the outcome of a hedging action."""
        reward = 0.0
        
        # 1. Reward for P&L change (main objective)
        # Scale P&L change to be significant for RL
        reward_pnl_scale = getattr(config, 'RL_REWARD_PNL_SCALE', 0.1) # e.g. 0.1 means $10 PNL = 1 reward
        reward += pnl_change_usd * reward_pnl_scale
        
        # 2. Reward for Delta Reduction (risk management objective)
        delta_reduction = abs(old_effective_delta_btc) - abs(new_effective_delta_btc)
        reward_delta_reduction_scale = getattr(config, 'RL_REWARD_DELTA_REDUCTION_SCALE', 5.0) # e.g. 5 per BTC delta reduced
        reward += delta_reduction * reward_delta_reduction_scale
        
        # 3. Penalty for Hedging Costs (transaction costs)
        reward_hedge_cost_scale = getattr(config, 'RL_REWARD_HEDGE_COST_SCALE', -2.0) # e.g. -2 per USD cost
        reward += abs(hedge_cost_usd) * reward_hedge_cost_scale # Cost is usually negative, so abs
        
        # 4. Penalty for "doing nothing" if delta is large (to encourage action)
        inaction_penalty_threshold_btc = getattr(config, 'RL_INACTION_PENALTY_THRESHOLD_BTC', 0.2)
        inaction_penalty_scale = getattr(config, 'RL_INACTION_PENALTY_SCALE', -10.0)
        if action_taken_idx == 0 and abs(new_effective_delta_btc) > inaction_penalty_threshold_btc: # Action 0 is "no hedge"
            reward += (abs(new_effective_delta_btc) - inaction_penalty_threshold_btc) * inaction_penalty_scale

        # 5. Small penalty for taking very large hedge actions unless necessary (to prefer smaller, precise hedges)
        large_action_penalty_scale = getattr(config, 'RL_LARGE_ACTION_PENALTY_SCALE', -1.0)
        action_magnitude = abs(self.action_mapping_btc_adjustment[action_taken_idx])
        if action_magnitude >= getattr(config, 'RL_HEDGE_ACTION_LARGE_BTC', 0.30) and delta_reduction < action_magnitude * 0.5 : # Large hedge didn't reduce delta much
            reward += large_action_penalty_scale * action_magnitude
            
        # Clip reward to a reasonable range
        min_reward = getattr(config, 'RL_REWARD_CLIP_MIN', -100.0)
        max_reward = getattr(config, 'RL_REWARD_CLIP_MAX', 100.0)
        final_reward = np.clip(reward, min_reward, max_reward)
        
        # logger.debug(f"RL Reward: {final_reward:.2f} (PNL Change: ${pnl_change_usd:.2f}, Delta Reduction: {delta_reduction:.3f} BTC, Hedge Cost: ${hedge_cost_usd:.2f})")
        return final_reward

    def execute_rl_hedge(self, current_btc_price: float) -> Optional[HedgeExecution]:
        """Decide and execute hedge using RL agent, then learn from experience."""
        if not self.pricing_engine or self.pricing_engine.current_price <= 0:
            logger.warning("RLHedger: Cannot execute RL hedge, pricing engine or current price unavailable.")
            return None

        # 1. Get Current State & Portfolio P&L
        current_state_np = self.get_current_state(current_btc_price) # self.risk_metrics updated here
        current_portfolio_pnl_usd = self.risk_metrics.option_pnl + self.risk_metrics.hedging_pnl
        current_effective_delta_btc = self.risk_metrics.net_delta + self.hedge_btc_position # From updated risk_metrics

        # 2. If there was a previous action, calculate reward and remember experience
        if self.last_state_np is not None and self.last_action_idx is not None and self.training_enabled:
            pnl_change_since_last_step = current_portfolio_pnl_usd - self.last_portfolio_pnl_usd
            
            # Cost of the hedge that LED to this state (if any)
            # This logic needs to correctly attribute cost to the *previous* action
            # For simplicity, assume hedge_cost refers to the cost of the action *about to be taken*
            # Or, we need to store the cost of the *last* action taken.
            # Let's assume the reward is for the transition from last_state to current_state due to last_action.
            # The "hedge_cost_usd" for the reward should be the cost incurred by `self.last_action_idx`.
            # This is tricky because `execute_delta_hedge` returns the cost of the *current* action.
            # For now, let's simplify: the reward considers the P&L change and delta change. Cost is for current action.
            # A more advanced reward would look at the cost of the *previous* action.

            # For now, calculate reward based on delta change and P&L change, and cost of current action.
            # This is slightly misaligned but simpler to implement initially.
            # A better way: store last_hedge_cost from previous step.
            
            # For simplicity, let's assume reward function uses cost of CURRENT action for now.
            # This means the agent learns based on the immediate cost of the action it's taking now,
            # and the P&L/delta change that happened *before* this action due to market moves and previous action.
            pass # Reward calculation will happen *after* current action is taken and its cost is known.


        # 3. Agent decides on an action
        action_idx = self.agent.act(current_state_np)
        btc_adjustment_for_hedge = self.action_mapping_btc_adjustment[action_idx]

        # 4. Execute the hedge (if adjustment is significant)
        hedge_execution_details: Optional[HedgeExecution] = None
        actual_hedge_cost_usd = 0.0

        # Define a minimum threshold for BTC adjustment to be considered a hedge action
        min_significant_hedge_btc = getattr(config, 'RL_MIN_SIGNIFICANT_HEDGE_BTC', 0.01)

        if abs(btc_adjustment_for_hedge) >= min_significant_hedge_btc:
            # Use parent's logic to simulate execution but with RL's chosen BTC amount
            slippage_bps_config = config.HEDGE_SLIPPAGE_BPS
            slippage_factor_val = 1 + (slippage_bps_config / 10000.0)
            
            exec_price_usd: float
            if btc_adjustment_for_hedge > 0: # Buying BTC
                exec_price_usd = current_btc_price * slippage_factor_val
            else: # Selling BTC
                exec_price_usd = current_btc_price / slippage_factor_val # Assuming slippage is symmetric for simplicity

            actual_hedge_cost_usd = -(btc_adjustment_for_hedge * exec_price_usd) # Cost is cash outflow if buying BTC

            # Update hedge position and cost basis (from parent or manage here)
            # This part must be consistent with how PortfolioHedger manages these.
            # For simplicity, let parent handle the actual position update.
            # We are just getting the *decision* from RL.
            # Let's re-evaluate: RLHedger *is* a PortfolioHedger, so it should manage its own state.
            
            # Update hedge position (self.hedge_btc_position)
            if (self.hedge_btc_position + btc_adjustment_for_hedge) != 0 and self.hedge_btc_position != -(btc_adjustment_for_hedge): # Avoid division by zero if new position is zero unless old was opposite
                new_total_cost = (self.hedge_cost_basis * self.hedge_btc_position) + (exec_price_usd * btc_adjustment_for_hedge)
                self.hedge_cost_basis = new_total_cost / (self.hedge_btc_position + btc_adjustment_for_hedge)
            elif (self.hedge_btc_position + btc_adjustment_for_hedge) == 0: # Position becomes zero
                 self.hedge_cost_basis = 0.0
            # else: cost_basis remains if no change to position size for this component

            self.hedge_btc_position += btc_adjustment_for_hedge
            
            hedge_execution_details = HedgeExecution(
                timestamp=time.time(),
                btc_quantity=btc_adjustment_for_hedge,
                btc_price=exec_price_usd,
                hedge_cost_usd=actual_hedge_cost_usd, # USD value of BTC traded (negative if buying)
                reason=f"RL Action {action_idx}: Adj {btc_adjustment_for_hedge:.3f} BTC"
            )
            self.risk_metrics.hedge_executions.append(hedge_execution_details) # Add to history
            self.last_hedge_timestamp = hedge_execution_details.timestamp
            logger.info(f"RL Hedger action: {btc_adjustment_for_hedge:.4f} BTC @ ${exec_price_usd:.2f}. New hedge pos: {self.hedge_btc_position:.4f} BTC.")
        else:
            # logger.debug(f"RL action {action_idx} resulted in no significant hedge ({btc_adjustment_for_hedge:.4f} BTC).")
            pass


        # 5. AFTER taking action and knowing its cost, calculate reward for PREVIOUS step
        if self.last_state_np is not None and self.last_action_idx is not None and self.training_enabled:
            # pnl_change_since_last_step was calculated before current action
            # current_effective_delta_btc is from *after* current action
            # new_effective_delta_btc for reward should be from *before* current action
            # This is getting complicated. Let's simplify the reward input.
            # Reward is for (last_state, last_action) -> current_state (before new action)
            
            # Recalculate risk metrics *after* the current hedge action to get the true new state
            new_risk_metrics_after_hedge = self.calculate_portfolio_risk()
            new_portfolio_pnl_after_hedge = new_risk_metrics_after_hedge.option_pnl + new_risk_metrics_after_hedge.hedging_pnl
            new_effective_delta_after_hedge = new_risk_metrics_after_hedge.net_delta + self.hedge_btc_position # self.hedge_btc_position is now updated

            pnl_change_for_reward = new_portfolio_pnl_after_hedge - self.last_portfolio_pnl_usd
            
            # The reward function should evaluate the transition from last_state to current_state (which is now post-action)
            # old_effective_delta was self.last_effective_delta_btc
            # new_effective_delta is new_effective_delta_after_hedge
            # hedge_cost is actual_hedge_cost_usd of the action just taken
            reward = self.calculate_reward(
                old_effective_delta_btc=self.last_effective_delta_btc,
                new_effective_delta_btc=new_effective_delta_after_hedge,
                pnl_change_usd=pnl_change_for_reward,
                hedge_cost_usd=actual_hedge_cost_usd, # Cost of the action that led to new_effective_delta
                action_taken_idx=action_idx # The action just taken
            )
            
            # Store experience: (s, a, r, s')
            # s = self.last_state_np
            # a = self.last_action_idx
            # r = reward (calculated for the transition s,a -> current_state_np)
            # s' = current_state_np (state AFTER action_idx was taken)
            # done = False (hedging is a continuous task)
            self.agent.remember(self.last_state_np, self.last_action_idx, reward, current_state_np, False)
            self.episode_rewards.append(reward) # Store reward for tracking
            
            # Train agent (replay)
            if len(self.agent.memory) > self.batch_size * 2 and self.training_enabled : # Ensure enough memory for diverse batches
                if np.random.rand() < getattr(config, 'RL_TRAINING_PROBABILITY', 0.25): # Train stochastically
                    loss = self.agent.replay()
                    if loss > 0: self.training_losses.append(loss)
            
            # Update target network periodically
            # A common schedule is every N steps or episodes
            rl_target_update_freq = getattr(config, 'RL_TARGET_UPDATE_FREQUENCY', 500) # steps
            if len(self.episode_rewards) % rl_target_update_freq == 0 and len(self.episode_rewards) > 0:
                self.agent.update_target_model()
                logger.info(f"RL Target Network updated at step {len(self.episode_rewards)}.")

        # 6. Update "last" variables for the next step
        self.last_state_np = current_state_np.copy()
        self.last_action_idx = action_idx
        self.last_portfolio_pnl_usd = new_portfolio_pnl_after_hedge if hedge_execution_details else current_portfolio_pnl_usd
        self.last_effective_delta_btc = new_effective_delta_after_hedge if hedge_execution_details else current_effective_delta_btc

        return hedge_execution_details # Return details of the hedge executed, if any

    def should_rehedge(self, current_btc_price: float) -> bool:
        """RL agent decides if rehedging is needed. For RL, this might always be true to get a decision."""
        # In an RL context, we might want the agent to make a decision more frequently
        # and the "no hedge" action handles not hedging.
        # For compatibility with a PortfolioHedger structure, this method can still exist.
        # Let's make it so RL decides at each hedging interval defined by config.
        
        current_time = time.time()
        time_since_last_eval_min = (current_time - self.last_hedge_timestamp) / 60 # Using last_hedge_timestamp as proxy for last eval
        
        # Use DELTA_HEDGE_FREQUENCY_MINUTES from config as the interval for RL decision making
        rl_decision_interval_min = config.DELTA_HEDGE_FREQUENCY_MINUTES
        
        if time_since_last_eval_min >= rl_decision_interval_min:
            # logger.debug(f"RLHedger: Time to make a hedging decision (interval: {rl_decision_interval_min} min).")
            return True # Let execute_rl_hedge make the actual decision (which could be "no action")
        
        return False # Not yet time for an RL decision based on fixed interval

    def execute_delta_hedge(self, current_btc_price: float) -> Optional[HedgeExecution]:
        """Override to use RL-based hedging if enabled."""
        if getattr(config, 'USE_RL_HEDGER', False) and getattr(config, 'HEDGING_ENABLED', False):
            # The should_rehedge might have already determined if it's time.
            # execute_rl_hedge will make the actual hedge (or no hedge) decision.
            return self.execute_rl_hedge(current_btc_price)
        else:
            # Fallback to parent's delta hedging if RL is not active
            # logger.debug("RLHedger: Falling back to PortfolioHedger's execute_delta_hedge.")
            return super().execute_delta_hedge(current_btc_price)

    def get_rl_performance_metrics(self) -> Dict:
        """Get RL training performance metrics."""
        return {
            "episode_rewards": {
                "count": len(self.episode_rewards),
                "mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                "std": float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
                "recent_100_mean": float(np.mean(list(self.episode_rewards)[-100:])) if len(self.episode_rewards) >= 100 else 0.0
            },
            "training_losses": {
                "count": len(self.training_losses),
                "mean": float(np.mean(self.training_losses)) if self.training_losses else 0.0,
                "recent_50_mean": float(np.mean(list(self.training_losses)[-50:])) if len(self.training_losses) >= 50 else 0.0
            },
            "exploration_rate_epsilon": self.agent.epsilon,
            "replay_memory_size": len(self.agent.memory),
        }

    def save_rl_model(self, filepath_prefix: Optional[str] = None):
        """Save RL model and training state. Uses config path if None."""
        path_to_save = filepath_prefix or self.model_save_path_prefix
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        self.agent.save(path_to_save) # DQNAgent.save now takes prefix
        logger.info(f"RLHedger model and agent state saved with prefix: {path_to_save}")

    def load_rl_model(self, filepath_prefix: Optional[str] = None):
        """Load RL model and training state. Uses config path if None."""
        path_to_load = filepath_prefix or self.model_save_path_prefix
        # Check for the model weights file specifically, as DQNAgent.load checks both
        model_weights_path = f"{path_to_load}_model_weights.weights.h5"
        if os.path.exists(model_weights_path):
            self.agent.load(path_to_load) # DQNAgent.load now takes prefix
            logger.info(f"RLHedger model and agent state loaded from prefix: {path_to_load}")
        else:
            logger.warning(f"RLHedger: No model weights found at {model_weights_path}. Agent will start fresh.")

    def set_training_mode(self, enabled: bool):
        """Enable/disable RL training and adjust exploration."""
        self.training_enabled = enabled
        if not enabled:
            self.agent.epsilon = self.agent.epsilon_min # Minimal exploration if not training
            logger.info("RLHedger training disabled. Epsilon set to min.")
        else:
            # Could reset epsilon to a higher value if re-enabling training after a while
            # self.agent.epsilon = config.get_config_value('RL_EPSILON_START', 1.0) 
            logger.info("RLHedger training enabled.")

