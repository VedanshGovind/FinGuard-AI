"""
agent/policy_rules.py

Responsibilities:
- Apply business logic and safety guardrails to ML scores.
- Handle edge cases where numerical thresholds might be misleading.
- Annotate decisions with policy-based metadata.
"""

from typing import Dict, Any
from app.config import settings

def apply_policy_rules(verdict: str, confidence: float, risk_level: str) -> Dict[str, Any]:
    """
    Refines the base decision based on organizational and safety policies.
    
    Args:
        verdict: Initial classification ('REAL', 'DEEPFAKE', 'UNCERTAIN')
        confidence: Aggregated probability score [0, 1]
        risk_level: Initial risk assessment ('LOW', 'MEDIUM', 'HIGH')
        
    Returns:
        Dict: Modified verdict and risk level with policy annotations.
    """
    
    final_verdict = verdict
    final_risk_level = risk_level
    policy_flags = []

    # Rule 1: Extreme Confidence Override
    
    if confidence > 0.98:
        final_risk_level = "CRITICAL"
        policy_flags.append("EXTREME_CONFIDENCE_LOCK")
    
    # Rule 2: The "Ambiguity Zone" Policy
    
    if settings.DEEPFAKE_THRESHOLD_LOW < confidence < settings.DEEPFAKE_THRESHOLD_HIGH:
        final_verdict = "UNCERTAIN"
        final_risk_level = "MEDIUM"#use nahihua hai kahi pe 
        policy_flags.append("REQUIRES_HUMAN_REVIEW")

    # Rule 3: Edge Case Safety
   
    if settings.RUNTIME_MODE == "EDGE_OFFLINE" and final_verdict == "UNCERTAIN":
        final_risk_level = "HIGH"
        policy_flags.append("OFFLINE_PRECAUTIONARY_ESCALATION")

    # Rule 4: Integrity Check Integration
    
    return {
        "verdict": final_verdict,
        "risk_level": final_risk_level,
        "policy_applied": policy_flags,
        "action_required": "MANUAL_INSPECTION" if final_risk_level in ["HIGH", "CRITICAL"] else "NONE"
    }