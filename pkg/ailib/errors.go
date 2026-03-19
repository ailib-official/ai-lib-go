// Package ailib error model aligned with AI-Protocol.
// 错误码体系，遵循 E1001-E9999 规范。
package ailib

import "fmt"

const (
	ErrInvalidRequest = "E1001"
	ErrAuthentication = "E1002"
	ErrPermission     = "E1003"
	ErrNotFound       = "E1004"
	ErrUnsupported    = "E1005"

	ErrRateLimited    = "E2001"
	ErrQuotaExhausted = "E2002"

	ErrServerError = "E3001"
	ErrOverloaded  = "E3002"
	ErrTimeout     = "E3003"

	ErrConflict  = "E4001"
	ErrCancelled = "E4002"

	ErrUnknown = "E9999"
)

type APIError struct {
	Code       string `json:"code"`
	StatusCode int    `json:"status_code"`
	Message    string `json:"message"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("[%s] status=%d message=%s", e.Code, e.StatusCode, e.Message)
}

// IsRetryableCode returns true if the error code is retryable per ARCH-003.
// Canonical implementation for cross-runtime consistency.
func IsRetryableCode(code string) bool {
	switch code {
	case ErrRateLimited, ErrServerError, ErrOverloaded, ErrTimeout, ErrConflict:
		return true
	default:
		return false
	}
}

// IsFallbackableCode returns true if the error code is fallbackable per ARCH-003.
// Canonical implementation for cross-runtime consistency.
func IsFallbackableCode(code string) bool {
	switch code {
	case ErrAuthentication, ErrRateLimited, ErrQuotaExhausted, ErrServerError, ErrOverloaded, ErrTimeout:
		return true
	default:
		return false
	}
}

func classifyStatus(status int) string {
	switch status {
	case 400:
		return ErrInvalidRequest
	case 401:
		return ErrAuthentication
	case 403:
		return ErrPermission
	case 404:
		return ErrNotFound
	case 409:
		return ErrConflict
	case 413:
		return ErrUnsupported
	case 429:
		return ErrRateLimited
	case 500:
		return ErrServerError
	case 503, 529:
		return ErrOverloaded
	case 502:
		return ErrOverloaded
	case 504:
		return ErrTimeout
	default:
		return ErrUnknown
	}
}
