# Changelog

## [Unreleased]

### Added

- Non-streaming `Client.Chat` response enrichment: manifest `response_paths` with OpenAI-style fallbacks via `EnrichNonstreamChatResponse` and `internal/protocol` JSON-path helpers (Rust/Python parity).

## [v0.5.1] - 2026-03-12

### Fixed

- **Version governance correction**: Retracted accidental `v0.0.1` in `go.mod`.
- **Release baseline clarity**: Canonical runtime baseline remains `v0.5.x` for continuing iterations.

### Notes

- If you previously consumed `v0.0.1`, upgrade to `v0.5.1` or newer.
- This release does not change public API semantics; it closes versioning inconsistency.

### Changed (2026-03-11, GO-001 Constitution Compliance)

- **ARCH-002**: Manifest-driven streaming decoder — `StreamingDecoderFormat()`, `NewDecoderWithFormat()` support openai_sse and anthropic_sse.
- **ARCH-003**: Unified retry/fallback semantics — exported `IsRetryableCode`/`IsFallbackableCode`; E2002 aligned per constitution.
- **TEST-001**: Removed portable path violation — dropped `D:/` hardcoded path; added `../../../../tests/compliance` candidate.
- **P1**: Fixed duplicate `response["error"]` branch in compliance classify; downgraded go.mod to Go 1.21.
- **Docs**: Refactored README.md and README_CN.md to align with ai-lib-rust, ai-lib-python, ai-lib-ts structure.

### Added

- Full repository reset for Go runtime rearchitecture baseline.
- Protocol-driven manifest models and loader (`internal/protocol`).
- Unified public client surface for chat/stream/embeddings/batch/stt/tts/rerank.
- Advanced capability APIs and client methods for mcp/computer_use/reasoning/video.
- Manifest capability gate for advanced features (undeclared capability returns `E1005`).
- Provider-aware error classification (`error.code/type` + HTTP status) with fallback matrix helpers.
- Transport preflight validations for endpoint path/method and required identifiers/messages.
- Manifest-driven error classification hook (`error_classification`) wired into runtime error parser.
- `FallbackClient` expanded to full `Client` surface with health-aware circuit breaker failover (including stream connect failover).
- Generic SSE stream decoder and retry execution wrapper.
- Baseline tests for chat/stream/embeddings, advanced capability routes, error mapping, and preflight behavior.
- Compliance test runner now executes all required ai-protocol categories directly from fixtures:
  - protocol loading
  - error classification
  - message building
  - stream decode / event mapping / tool accumulation
  - request parameter mapping
  - retry decision
