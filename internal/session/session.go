package session

import (
	"context"
	"strings"
)

type contextKey string

const sessionIDKey contextKey = "session_id"

func WithID(ctx context.Context, id string) context.Context {
	id = strings.TrimSpace(id)
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, sessionIDKey, id)
}

func IDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	id, _ := ctx.Value(sessionIDKey).(string)
	return strings.TrimSpace(id)
}
