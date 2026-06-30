package ailib

import "testing"

// ARCH-003 fixture: quota exhaustion is fallbackable but not retryable on the same provider.
func TestIsRetryableCode_E2002NotRetryable(t *testing.T) {
	if IsRetryableCode(ErrQuotaExhausted) {
		t.Fatalf("%s must not be retryable per ARCH-003", ErrQuotaExhausted)
	}
	if !IsFallbackableCode(ErrQuotaExhausted) {
		t.Fatalf("%s must remain fallbackable", ErrQuotaExhausted)
	}
}
