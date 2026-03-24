package conversation

import (
	"sync"
	"time"
)

type InMemoryStore struct {
	mu      sync.Mutex
	limit   int
	order   []string
	records map[string]Record
}

func NewInMemoryStore(limit int) *InMemoryStore {
	if limit < 0 {
		limit = 0
	}
	return &InMemoryStore{
		limit:   limit,
		order:   make([]string, 0, limit),
		records: make(map[string]Record, limit),
	}
}

func (store *InMemoryStore) Get(responseID string) (Record, bool) {
	store.mu.Lock()
	defer store.mu.Unlock()
	record, ok := store.records[responseID]
	return record, ok
}

func (store *InMemoryStore) Save(record Record) {
	if store.limit == 0 {
		return
	}
	store.mu.Lock()
	defer store.mu.Unlock()

	if _, ok := store.records[record.ResponseID]; ok {
		store.records[record.ResponseID] = record
		return
	}

	if len(store.order) >= store.limit {
		oldest := store.order[0]
		store.order = store.order[1:]
		delete(store.records, oldest)
	}

	store.records[record.ResponseID] = record
	store.order = append(store.order, record.ResponseID)
}

func RecordFromResponse(responseID, modelID string, snapshot Request) Record {
	return Record{
		ResponseID: responseID,
		ModelID:    modelID,
		Messages:   append([]Message(nil), snapshot.Messages...),
		CreatedAt:  time.Now().UTC(),
	}
}
