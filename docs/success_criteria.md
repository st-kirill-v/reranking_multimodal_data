# Success Criteria

A question is considered successfully processed when all of the following hold:

- the question is processed without runtime error;
- retrieved pages exist;
- the VLM returns a non-empty answer;
- latency is recorded;
- prediction is saved;
- metrics are computed;
- if the answer is not found, the model returns exactly `NOT FOUND`.

These criteria are enforced by `scripts/run_experiment.py` when it converts raw evaluator output
into `predictions.jsonl`.
