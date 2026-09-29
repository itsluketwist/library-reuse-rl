"""Response processing: sandboxed evaluation, code metrics and stop reasons.

Parsing responses (reasoning, answer, code blocks and library usage) is done by the lure
package, in reward/.
"""

from src.response.evaluate import evaluate_record, execute_code, resolve_eval_python
from src.response.metrics import code_metrics
from src.response.pipeline import ProcessedResponse, process_response
from src.response.stop_reason import map_stop_reason, stop_reason_from_tokens


__all__ = [
    "ProcessedResponse",
    "code_metrics",
    "evaluate_record",
    "execute_code",
    "map_stop_reason",
    "process_response",
    "resolve_eval_python",
    "stop_reason_from_tokens",
]
