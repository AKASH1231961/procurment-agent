"""Parent agent for reviewing completed negotiations and coaching future ones."""

from pathlib import Path
from typing import Any


class ParentNegotiationAgent:
    """Evaluate buyer behavior and persist product-specific coaching in the log."""

    COUNTER_ONCE = "counter_once"
    STANDARD = "standard"

    def get_previous_coaching(
        self,
        log_path: str | Path,
        product: str,
    ) -> dict[str, str] | None:
        log_path = Path(log_path)
        if not log_path.exists():
            return None

        sessions = log_path.read_text(encoding="utf-8").split(
            "## Negotiation session "
        )
        for session in reversed(sessions):
            product_line = next(
                (
                    line.removeprefix("- Product: ")
                    for line in session.splitlines()
                    if line.startswith("- Product: ")
                ),
                None,
            )
            if product_line != product:
                continue

            marker = "### Parent agent evaluation"
            if marker not in session:
                continue
            evaluation = session.split(marker, 1)[1]
            feedback = self._evaluation_value(evaluation, "Feedback")
            strategy = self._evaluation_value(
                evaluation,
                "Next-session strategy",
            )
            if feedback and strategy in {self.COUNTER_ONCE, self.STANDARD}:
                return {"feedback": feedback, "strategy": strategy}
        return None

    def evaluate_session(
        self,
        *,
        result: dict[str, Any],
        product: str,
        initial_offer: float,
        max_price: float,
        max_rounds: int,
    ) -> dict[str, str]:
        buyer_actions = [
            message.get("message_type")
            for message in result["history"]
            if message.get("from_agent") == "buyer_agent"
        ]
        agreed_price = result.get("agreed_price_per_unit")
        accepted_first_counter = (
            result["status"] == "agreed"
            and "ACCEPT" in buyer_actions
            and "COUNTER" not in buyer_actions
            and agreed_price is not None
            and agreed_price < max_price
            and max_rounds > 1
        )

        if accepted_first_counter:
            savings = max_price - agreed_price
            feedback = (
                f"For {product}, you accepted the seller's first counter at "
                f"${agreed_price:.2f} per unit, ${savings:.2f} below your "
                "authorized maximum. In the next negotiation for this product, "
                "try one midpoint counter before accepting when another round "
                "is available."
            )
            strategy = self.COUNTER_ONCE
        elif result["status"] == "agreed":
            feedback = (
                f"For {product}, you reached an agreement at "
                f"${agreed_price:.2f} per unit after opening at "
                f"${initial_offer:.2f}. Continue to keep every offer within "
                "the authorized maximum and use seller concessions to guide "
                "your next move."
            )
            strategy = self.STANDARD
        else:
            feedback = (
                f"For {product}, this session ended without an agreement. "
                "Keep the authorized maximum as a hard limit and use the "
                "seller's stated counter and reasoning to shape a measured "
                "response in the next negotiation."
            )
            strategy = self.STANDARD

        return {"feedback": feedback, "strategy": strategy}

    def log_evaluation(
        self,
        log_path: str | Path,
        evaluation: dict[str, str],
    ) -> None:
        with Path(log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(
                "\n### Parent agent evaluation\n\n"
                f"- Feedback: {evaluation['feedback']}\n"
                f"- Next-session strategy: `{evaluation['strategy']}`\n"
            )

    @staticmethod
    def _evaluation_value(evaluation: str, label: str) -> str | None:
        prefix = f"- {label}: "
        return next(
            (
                line.removeprefix(prefix).strip().strip("`")
                for line in evaluation.splitlines()
                if line.startswith(prefix)
            ),
            None,
        )
