"""Enterprise webhook triage and automated decision runner.

Demonstrates an end-to-end deterministic production workflow:
1. Ingests raw webhook events (financial transactions, DevOps alerts, customer tickets).
2. Evaluates semantic priority and action category in a single pass.
3. Enforces mathematical confidence gating:
   - If calibrated confidence >= 0.85 and evidence is sufficient -> autonomous software dispatch.
   - If calibrated confidence < 0.85 or __insufficient_evidence__ -> human review queue escalation.
4. Deterministic code maintains complete audit trails with millisecond latencies.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.calibration import TemperatureCalibrator
from core.engine_encoder import DecisionEngine
from core.primitives import Choice, Level, Option, Query, Score

WEBHOOK_PAYLOADS = [
    {
        "event_id": "evt_fin_9821",
        "source": "stripe_dispute_webhook",
        "timestamp": "2026-09-17T05:30:00Z",
        "context": "Cardholder usr_4910 was charged $120.00 twice on credit card ending in 4491 at merchant CloudHost Ltd. Customer submitted bank statement showing duplicate debit.",
        "payload": {"amount": 120.00, "currency": "USD", "dispute_reason": "duplicate_charge"},
    },
    {
        "event_id": "evt_ops_1042",
        "source": "pagerduty_alert",
        "timestamp": "2026-09-17T05:30:15Z",
        "context": "Primary database cluster leader node in region SG crashed with disk corruption. 100% of read/write queries failing.",
        "payload": {"service": "pg_primary", "cluster": "prod-sg-1", "severity": "CRITICAL"},
    },
    {
        "event_id": "evt_support_3319",
        "source": "zendesk_incoming",
        "timestamp": "2026-09-17T05:30:30Z",
        "context": "Customer message received: 'I need urgent assistance with my recent transaction.' No account number, amount, or transaction ID provided.",
        "payload": {"channel": "email", "subject": "urgent help"},
    },
    {
        "event_id": "evt_audit_7712",
        "source": "aws_guardduty",
        "timestamp": "2026-09-17T05:30:45Z",
        "context": "IAM role prod-data-access assumed by unrecognized developer IP 192.0.2.14 outside business hours.",
        "payload": {"account": "123456789012", "arn": "arn:aws:iam::prod-data-access"},
    },
    {
        "event_id": "evt_ood_0099",
        "source": "unfiltered_slack_bot",
        "timestamp": "2026-09-17T05:31:00Z",
        "context": "The 2024 total solar eclipse crossed North America, passing over Mexico, the United States, and Canada.",
        "payload": {"channel": "general", "user": "guest"},
    },
]


@dataclass(frozen=True)
class PolicyDispatchResult:
    event_id: str
    decision_type: str
    selected_action: str
    calibrated_confidence: float
    routing_destination: str
    action_taken: str
    audit_notes: str
    latency_ms: float


class EnterpriseTriageRunner:
    """Production decision controller with calibrated confidence gating."""

    def __init__(
        self,
        checkpoint_dir: str = "artifacts",
        confidence_threshold: float = 0.85,
    ):
        self.confidence_threshold = confidence_threshold
        cal_path = Path(checkpoint_dir) / "calibrator_modernbert.json"
        if cal_path.exists():
            print(f"Loading calibrated temperature from {cal_path}...")
            self.calibrator = TemperatureCalibrator.load(cal_path)
        else:
            print("Notice: Calibrator file not found; initializing unit temperature.")
            self.calibrator = TemperatureCalibrator(initial_temperature=1.0)

        safetensors_path = Path(checkpoint_dir) / "model.safetensors"
        if not safetensors_path.exists():
            safetensors_path = Path(checkpoint_dir) / "openjev_modernbert.safetensors"
        model_source = "knowledgator/gliclass-modern-base-v2.0"
        print(f"Initializing Verdict-open-jev DecisionEngine with calibrated temperature (T={self.calibrator.temperature:.3f})...")
        self.engine = DecisionEngine(
            model_name_or_path=model_source,
            calibrator=self.calibrator,
            device="cpu",
        )

        if safetensors_path.exists():
            print(f"Loading fine-tuned checkpoint weights into engine: {safetensors_path}...")
            from safetensors.torch import load_file

            self.engine.model.load_state_dict(load_file(str(safetensors_path)))
            self.engine.model.eval()

    def process_event(self, event: dict[str, Any]) -> PolicyDispatchResult:
        """Evaluate a single webhook event under deterministic threshold policy."""
        event_id = event["event_id"]
        context = event["context"]

        # Formulate typed decision query
        query = Choice(
            id="triage_action",
            question="What is the operational triage or remediation action for this event?",
            options=[
                Option(
                    id="auto_refund_dispute",
                    description="Initiate standard merchant chargeback dispute for billed goods or services",
                ),
                Option(
                    id="freeze_account_fraud",
                    description="Freeze account immediately due to suspected unauthorized or fraudulent activity",
                ),
                Option(
                    id="p0_failover_page",
                    description="P0 critical outage requiring immediate on-call page and failover dispatch",
                ),
                Option(
                    id="security_rotate_creds",
                    description="Security anomaly alert requiring credential rotation and access audit",
                ),
            ],
        )

        start_time = time.perf_counter()
        batch_result = self.engine.evaluate(context=context, queries=[query])
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision = batch_result.results[0]
        selected_id = decision.selected_id
        confidence = decision.selected_probability
        is_abstention = decision.is_abstention

        # Deterministic software controls policy
        if is_abstention:
            destination = "HUMAN_TIER_1_TRIAGE"
            action = "Escalated to human support queue (Missing or conflicting evidence)"
            notes = f"Model explicitly abstained (__insufficient_evidence__). Prob: {confidence:.2f}"
        elif confidence >= self.confidence_threshold:
            destination = "AUTONOMOUS_EXECUTION"
            action = f"Auto-executed programmatic dispatch: [{selected_id}]"
            notes = f"Calibrated confidence {confidence:.2f} clears policy threshold >= {self.confidence_threshold:.2f}"
        else:
            destination = "HUMAN_ESCALATION_QUEUE"
            action = f"Held for supervisor confirmation (Recommended: {selected_id})"
            notes = f"Confidence {confidence:.2f} is below autonomous safety threshold ({self.confidence_threshold:.2f})"

        return PolicyDispatchResult(
            event_id=event_id,
            decision_type=selected_id,
            selected_action=selected_id,
            calibrated_confidence=confidence,
            routing_destination=destination,
            action_taken=action,
            audit_notes=notes,
            latency_ms=latency_ms,
        )


def main() -> None:
    print("=" * 80)
    print("VERDICT-OPEN-JEV ENTERPRISE TRIAGE RUNNER (DETERMINISTIC GATING DEMO)")
    print("=" * 80)

    runner = EnterpriseTriageRunner(confidence_threshold=0.85)

    print("\nProcessing incoming webhook stream...")
    for event in WEBHOOK_PAYLOADS:
        res = runner.process_event(event)
        print("\n" + "-" * 70)
        print(f"EVENT:        {res.event_id} ({event['source']})")
        print(f"CONTEXT:      {event['context']}")
        print(f"DECISION:     {res.selected_action}")
        print(f"CONFIDENCE:   {res.calibrated_confidence * 100:.1f}% (Calibrated)")
        print(f"DESTINATION:  {res.routing_destination}")
        print(f"ACTION:       {res.action_taken}")
        print(f"AUDIT LOG:    {res.audit_notes}")
        print(f"LATENCY:      {res.latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
