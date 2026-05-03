package resilience

import (
	"context"
	"errors"
	"testing"
)

func TestExecuteAttemptsFirstTrySuccess(t *testing.T) {
	p := Policy{MaxAttempts: 3, MinDelay: 0, MaxDelay: 0}
	n, err := ExecuteAttempts(context.Background(), p, func(context.Context) error {
		return nil
	}, nil)
	if err != nil || n != 1 {
		t.Fatalf("got attempts=%d err=%v", n, err)
	}
}

func TestExecuteAttemptsRetriesThenSuccess(t *testing.T) {
	p := Policy{MaxAttempts: 3, MinDelay: 0, MaxDelay: 0}
	var calls int
	n, err := ExecuteAttempts(context.Background(), p, func(context.Context) error {
		calls++
		if calls < 2 {
			return errors.New("transient")
		}
		return nil
	}, func(err error) bool { return err != nil })
	if err != nil || n != 2 {
		t.Fatalf("got attempts=%d err=%v calls=%d", n, err, calls)
	}
}
