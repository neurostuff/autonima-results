# E6 Codex run notes: <project>

- Protocol: `review.yaml`, copied from the registered protocol or E4 full-text workspace
  config (the root workspace README names the exception).
- Protocol SHA-256: <computed during setup>
- Skills package commit: `f3a68422262350ee3ddcc36fc89f416695f10d10`
- Harness: Codex CLI 0.155.0-alpha.16.3 (record actual version at run time)
- Judge transport: for Portkey arms, `run_judge.sh` selects the role model/effort and
  `portkey_codex.sh` explicitly supplies routing for every fresh `codex exec` session;
  native agent spawning is disabled.
- Models: abstract/full-text `gpt-6-luna`; extraction/selection `gpt-6-astra`;
  orchestrator `gpt-6-astra`
- Effort: abstract low; full-text medium; extraction low; selection medium;
  orchestrator high
- Search date and result count:
- Full-text source coverage:
- Batch sizes and retries:
- Pilot decisions and protocol changes:
- Run notes/deviations:
- Codex transcript audit (run after project work ends or pauses; command and report path):
- NiMARE version:
