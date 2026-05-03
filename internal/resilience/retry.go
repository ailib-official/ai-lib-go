// Package resilience provides retry policy utilities.
// 重试策略模块，提供指数退避与可重试判定。
package resilience

import (
	"context"
	"errors"
	"math"
	"time"
)

type Policy struct {
	MaxAttempts int
	MinDelay    time.Duration
	MaxDelay    time.Duration
}

func DefaultPolicy() Policy {
	return Policy{
		MaxAttempts: 3,
		MinDelay:    time.Second,
		MaxDelay:    20 * time.Second,
	}
}

type TryFunc func(context.Context) error

// Execute runs fn with retries; use ExecuteAttempts when reporting micro_retry_count.
func Execute(ctx context.Context, p Policy, fn TryFunc, retryable func(error) bool) error {
	_, err := ExecuteAttempts(ctx, p, fn, retryable)
	return err
}

// ExecuteAttempts runs fn with retries and returns how many times fn was invoked (>=1 on error).
func ExecuteAttempts(ctx context.Context, p Policy, fn TryFunc, retryable func(error) bool) (attempts int, err error) {
	if p.MaxAttempts <= 0 {
		p.MaxAttempts = 1
	}
	if p.MinDelay <= 0 {
		p.MinDelay = 200 * time.Millisecond
	}
	if p.MaxDelay < p.MinDelay {
		p.MaxDelay = p.MinDelay
	}

	var lastErr error
	for attempt := 0; attempt < p.MaxAttempts; attempt++ {
		attempts = attempt + 1
		if err := fn(ctx); err != nil {
			lastErr = err
			if attempt == p.MaxAttempts-1 || (retryable != nil && !retryable(err)) {
				return attempts, err
			}
			delay := backoff(p, attempt)
			select {
			case <-ctx.Done():
				return attempts, ctx.Err()
			case <-time.After(delay):
			}
			continue
		}
		return attempts, nil
	}
	if lastErr == nil {
		return attempts, errors.New("retry exhausted")
	}
	return attempts, lastErr
}

func backoff(p Policy, attempt int) time.Duration {
	m := math.Pow(2, float64(attempt))
	d := time.Duration(float64(p.MinDelay) * m)
	if d > p.MaxDelay {
		return p.MaxDelay
	}
	return d
}
