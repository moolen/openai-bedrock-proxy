package usage

import "sync"

type Totals struct {
	Requests         int
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type InMemoryTracker struct {
	mu      sync.Mutex
	limit   int
	order   []string
	overall Totals
	session map[string]Totals
}

func NewInMemoryTracker(limit int) *InMemoryTracker {
	if limit < 0 {
		limit = 0
	}
	return &InMemoryTracker{
		limit:   limit,
		order:   make([]string, 0, limit),
		session: make(map[string]Totals, limit),
	}
}

func (t *InMemoryTracker) Record(sessionID string, promptTokens int, completionTokens int, totalTokens int) {
	if totalTokens == 0 {
		totalTokens = promptTokens + completionTokens
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	t.overall.Requests++
	t.overall.PromptTokens += promptTokens
	t.overall.CompletionTokens += completionTokens
	t.overall.TotalTokens += totalTokens

	if sessionID == "" || t.limit == 0 {
		return
	}

	current, ok := t.session[sessionID]
	if !ok {
		if len(t.order) >= t.limit {
			oldest := t.order[0]
			t.order = t.order[1:]
			delete(t.session, oldest)
		}
		t.order = append(t.order, sessionID)
	}

	current.Requests++
	current.PromptTokens += promptTokens
	current.CompletionTokens += completionTokens
	current.TotalTokens += totalTokens
	t.session[sessionID] = current
}

func (t *InMemoryTracker) Overall() Totals {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.overall
}

func (t *InMemoryTracker) Session(sessionID string) (Totals, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	total, ok := t.session[sessionID]
	return total, ok
}
