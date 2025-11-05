import torch
from typing import Optional, List, Tuple
from collections import deque


class DynamicRolloutScheduler:
    """
    Manages dynamic rollout adjustment based on discriminator performance.
    Increases rollout when CM is winning, decreases gradually when disc is too strong.
    Emphasizes stability - keeps each rollout level longer before adjusting.
    """

    def __init__(
            self,
            initial_rollout: int = 10,
            min_rollout: int = 1,
            max_rollout: int = 10,
            warmup_epochs: int = 10,
            # Performance thresholds - more conservative
            cm_winning_threshold: float = 0.20,  # Gap below this = CM is fooling disc
            avg_fake_upper_limit: float = 0.40,  # Threshold for which above the CM is considered to outperform Disc
            avg_fake_lower_limit: float = 0.20,  # Threshold for which below the Disc is considered to outperform CM
            disc_too_strong_threshold: float = 0.70,  # Gap above this = disc way too strong
            # Adjustment parameters - longer patience
            adjustment_cooldown: int = 8,  # Longer wait between adjustments
            increase_patience: int = 5,  # Epochs of CM winning before increasing
            decrease_patience: int = 5,  # Many epochs of disc domination before decreasing
            history_size: int = 5,  # Larger history for more stable averaging
            # Step sizes
            increase_step: int = 1,
            decrease_step: int = 1,
            # Additional stability
            min_epochs_at_rollout: int = 5,  # Minimum epochs to stay at current rollout
    ):
        self.current_rollout = initial_rollout
        self.min_rollout = min_rollout
        self.max_rollout = max_rollout
        self.warmup_epochs = warmup_epochs

        # Thresholds - more conservative
        self.cm_winning_threshold = cm_winning_threshold
        self.disc_too_strong_threshold = disc_too_strong_threshold
        self.avg_fake_upper_limit = avg_fake_upper_limit
        self.avg_fake_lower_limit = avg_fake_lower_limit

        # Adjustment control - longer patience
        self.adjustment_cooldown = adjustment_cooldown
        self.increase_patience = increase_patience
        self.decrease_patience = decrease_patience
        self.increase_step = increase_step
        self.decrease_step = decrease_step
        self.min_epochs_at_rollout = min_epochs_at_rollout

        # State tracking
        self.last_adjustment_epoch = -adjustment_cooldown
        self.epochs_at_current_rollout = 0
        self.epochs_cm_winning = 0  # Consecutive epochs where CM is winning
        self.epochs_disc_too_strong = 0  # Consecutive epochs where disc is too strong
        self.performance_history = deque(maxlen=history_size)

        # Placeholder for disc predictions
        self.avg_gap = 0
        self.avg_fake = 0

        # Statistics
        self.adjustment_history = []

    def update(
            self,
            current_epoch: int,
            disc_real_mean: float,
            disc_fake_mean: float,
            disc_loss: Optional[float] = None,
    ) -> Tuple[int, dict]:
        """
        Update rollout based on discriminator performance.
        Only increases when CM is consistently winning.
        Only decreases when discriminator is extremely dominant for a long time.

        Args:
            current_epoch: Current training epoch
            disc_real_mean: Mean discriminator output for real samples (after sigmoid)
            disc_fake_mean: Mean discriminator output for fake samples (after sigmoid)
            disc_loss: Optional discriminator loss for additional context

        Returns:
            Tuple of (new_rollout, info_dict)
        """
        # Calculate performance gap
        performance_gap = abs(disc_real_mean - disc_fake_mean)

        # Store in history
        self.performance_history.append({
            'epoch': current_epoch,
            'gap': performance_gap,
            'real': disc_real_mean,
            'fake': disc_fake_mean,
            'loss': disc_loss,
        })

        # Increment epochs at current rollout
        self.epochs_at_current_rollout += 1

        # During warmup, keep initial rollout
        if current_epoch < self.warmup_epochs:
            return self.current_rollout, {
                'action': 'warmup',
                'gap': performance_gap,
                'rollout': self.current_rollout,
                'epochs_at_rollout': self.epochs_at_current_rollout,
            }

        # Check if we're in cooldown
        epochs_since_adjustment = current_epoch - self.last_adjustment_epoch
        if epochs_since_adjustment < self.adjustment_cooldown:
            return self.current_rollout, {
                'action': 'cooldown',
                'gap': performance_gap,
                'rollout': self.current_rollout,
                'cooldown_remaining': self.adjustment_cooldown - epochs_since_adjustment,
                'epochs_at_rollout': self.epochs_at_current_rollout,
            }

        # Enforce minimum time at current rollout
        if self.epochs_at_current_rollout < self.min_epochs_at_rollout:
            return self.current_rollout, {
                'action': 'min_epochs_not_met',
                'gap': performance_gap,
                'rollout': self.current_rollout,
                'epochs_remaining': self.min_epochs_at_rollout - self.epochs_at_current_rollout,
                'epochs_at_rollout': self.epochs_at_current_rollout,
            }

        # Calculate average gap over recent history
        if len(self.performance_history) >= 3:
            recent_history = list(self.performance_history)[-self.increase_patience:]
            avg_gap = sum(h['gap'] for h in recent_history) / len(recent_history)
            avg_real = sum(h['real'] for h in recent_history) / len(recent_history)
            avg_fake = sum(h['fake'] for h in recent_history) / len(recent_history)
        else:
            return self.current_rollout, {
                'action': 'insufficient_history',
                'gap': performance_gap,
                'rollout': self.current_rollout,
                'epochs_at_rollout': self.epochs_at_current_rollout,
            }

        old_rollout = self.current_rollout
        action = 'maintain'

        self.avg_gap = avg_gap
        self.avg_fake = avg_fake

        # Track consecutive epochs in each state
        # CM is winning: small gap + disc can't distinguish (low fake score)
        if avg_gap < self.cm_winning_threshold and avg_fake > self.avg_fake_upper_limit:
            self.epochs_cm_winning += 1
            self.epochs_disc_too_strong = 0  # Reset counter
        # Disc is too strong: large gap + disc very confident
        elif avg_gap > self.disc_too_strong_threshold and avg_fake < self.avg_fake_lower_limit:
            self.epochs_disc_too_strong += 1
            self.epochs_cm_winning = 0  # Reset counter
        # In between - balanced or transitioning
        else:
            # Decay counters slowly (don't reset immediately)
            self.epochs_cm_winning = max(0, self.epochs_cm_winning - 1)
            self.epochs_disc_too_strong = max(0, self.epochs_disc_too_strong - 1)

        # Decision logic
        # 1. CM is consistently winning -> INCREASE rollout (make task easier for disc)
        if self.epochs_cm_winning >= self.increase_patience:
            if self.current_rollout < self.max_rollout:
                self.current_rollout = min(self.current_rollout + self.increase_step, self.max_rollout)
                action = 'increase_cm_winning'
                self.last_adjustment_epoch = current_epoch
                self.epochs_at_current_rollout = 0
                self.epochs_cm_winning = 0  # Reset after action
                print(f"\n📈 INCREASE Rollout: {old_rollout} -> {self.current_rollout}")
                print(f"   Reason: CM winning consistently ({self.increase_patience}+ epochs)")
                print(f"   Gap: {avg_gap:.4f}, Fake: {avg_fake:.4f}")

        # 2. Disc is TOO strong for a LONG time -> DECREASE rollout (make task harder)
        elif self.epochs_disc_too_strong >= self.decrease_patience:
            if self.current_rollout > self.min_rollout:
                self.current_rollout = max(self.current_rollout - self.decrease_step, self.min_rollout)
                action = 'decrease_disc_too_strong'
                self.last_adjustment_epoch = current_epoch
                self.epochs_at_current_rollout = 0
                self.epochs_disc_too_strong = 0  # Reset after action
                print(f"\n📉 DECREASE Rollout: {old_rollout} -> {self.current_rollout}")
                print(f"   Reason: Disc too strong for {self.decrease_patience}+ epochs")
                print(f"   Gap: {avg_gap:.4f}, Fake: {avg_fake:.4f}")

        # 3. Otherwise maintain current rollout
        else:
            action = 'maintain'

        # Record adjustment if rollout changed
        if old_rollout != self.current_rollout:
            self.adjustment_history.append({
                'epoch': current_epoch,
                'old_rollout': old_rollout,
                'new_rollout': self.current_rollout,
                'action': action,
                'avg_gap': avg_gap,
                'avg_real': avg_real,
                'avg_fake': avg_fake,
                'epochs_cm_winning': self.epochs_cm_winning,
                'epochs_disc_too_strong': self.epochs_disc_too_strong,
            })

        return self.current_rollout, {
            'action': action,
            'gap': performance_gap,
            'avg_gap': avg_gap,
            'rollout': self.current_rollout,
            'old_rollout': old_rollout,
            'changed': old_rollout != self.current_rollout,
            'epochs_at_rollout': self.epochs_at_current_rollout,
            'epochs_cm_winning': self.epochs_cm_winning,
            'epochs_disc_too_strong': self.epochs_disc_too_strong,
        }

    def get_rollout(self) -> int:
        """Get current rollout value."""
        return self.current_rollout

    def get_statistics(self) -> dict:
        """Get statistics about adjustments."""
        increases = 0
        decreases = 0

        if self.adjustment_history:
            increases = sum(1 for h in self.adjustment_history if h['new_rollout'] > h['old_rollout'])
            decreases = sum(1 for h in self.adjustment_history if h['new_rollout'] < h['old_rollout'])

        return {
            'total_adjustments': len(self.adjustment_history),
            'increases': increases,
            'decreases': decreases,
            'current_rollout': self.current_rollout,  # Always include current rollout
            'epochs_at_current_rollout': self.epochs_at_current_rollout,
            'epochs_cm_winning': self.epochs_cm_winning,
            'epochs_disc_too_strong': self.epochs_disc_too_strong,
            'adjustment_history': self.adjustment_history,
            'avg_gap': self.avg_gap,
            'avg_fake': self.avg_fake
        }