from typing import Dict

from app.config import settings
from agent.policy_rules import apply_policy_rules
from agent.explanation_engine import generate_explanation
from app_logging.event_logger import log_event


class DecisionEngine:
    """
    Central autonomous decision engine.

    Responsibilities:
    - Interpret aggregated ML score
    - Apply policy rules
    - Produce final decision + explanation
    """

    def __init__(self, media_type: str = "video"):
        """
        Args:
            media_type: "video" or "audio" - determines which thresholds to use
        """
        self.media_type = media_type
        
        if media_type == "audio":
            # Use audio-specific thresholds (if they exist in config)
            self.high_threshold = getattr(settings, 'AUDIO_DEEPFAKE_THRESHOLD_HIGH', 0.70)
            self.low_threshold = getattr(settings, 'AUDIO_DEEPFAKE_THRESHOLD_LOW', 0.30)
        else:
            # Use video thresholds
            self.high_threshold = settings.DEEPFAKE_THRESHOLD_HIGH
            self.low_threshold = settings.DEEPFAKE_THRESHOLD_LOW

    def decide(self, aggregated_score: float) -> Dict:
        """
        Makes a final decision based on model confidence and policy rules.

        Args:
            aggregated_score (float): Media-level deepfake confidence

        Returns:
            dict containing:
                - verdict
                - confidence
                - risk_level
                - explanation
        """

        # Base classification from thresholds
        if aggregated_score >= self.high_threshold:
            base_verdict = "DEEPFAKE"
            risk_level = "HIGH"
        elif aggregated_score <= self.low_threshold:
            base_verdict = "REAL"
            risk_level = "LOW"
        else:
            base_verdict = "Real"
            risk_level = "MEDIUM"

        # Apply policy rules (can override or annotate)
        policy_result = apply_policy_rules(
            verdict=base_verdict,
            confidence=aggregated_score,
            risk_level=risk_level
        )

        # Generate human-readable explanation
        explanation = generate_explanation(
            verdict=policy_result["verdict"],
            confidence=aggregated_score,
            risk_level=policy_result["risk_level"]
        )

        decision = {
            "verdict": policy_result["verdict"],
            "confidence": round(aggregated_score, 4),
            "risk_level": policy_result["risk_level"],
            "explanation": explanation
        }

        log_event(
            "DECISION_MADE",
            decision
        )

        return decision


def make_decision(aggregated_score: float, media_type: str = "video") -> Dict:
    """
    Makes a decision for video or audio analysis.
    
    Args:
        aggregated_score: Deepfake probability (0-1)
        media_type: "video" or "audio" (default: "video")
        
    Returns:
        Decision dict with verdict, confidence, risk_level, explanation
    """
    engine = DecisionEngine(media_type=media_type)
    return engine.decide(aggregated_score)