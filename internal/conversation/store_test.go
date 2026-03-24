package conversation

import "testing"

func TestInMemoryStoreEvictsOldestRecord(t *testing.T) {
	store := NewInMemoryStore(2)
	store.Save(Record{ResponseID: "r1", ModelID: "model"})
	store.Save(Record{ResponseID: "r2", ModelID: "model"})
	store.Save(Record{ResponseID: "r3", ModelID: "model"})

	if _, ok := store.Get("r1"); ok {
		t.Fatalf("expected r1 to be evicted")
	}
	if _, ok := store.Get("r2"); !ok {
		t.Fatalf("expected r2 to remain")
	}
	if _, ok := store.Get("r3"); !ok {
		t.Fatalf("expected r3 to remain")
	}
}

func TestInMemoryStoreLookupMissReturnsFalse(t *testing.T) {
	store := NewInMemoryStore(1)

	if _, ok := store.Get("missing"); ok {
		t.Fatalf("expected lookup miss to return false")
	}
}

func TestInMemoryStoreDoesNotExposeMutableSnapshot(t *testing.T) {
	store := NewInMemoryStore(1)

	original := Record{
		ResponseID: "r1",
		ModelID:    "model",
		Messages: []Message{
			{Role: "user", Text: "hi"},
		},
	}

	store.Save(original)
	original.Messages[0].Text = "mutated"

	stored, ok := store.Get("r1")
	if !ok {
		t.Fatalf("expected record to be stored")
	}
	if stored.Messages[0].Text != "hi" {
		t.Fatalf("expected stored record to be immutable after Save")
	}

	stored.Messages[0].Text = "changed"
	roundTrip, ok := store.Get("r1")
	if !ok {
		t.Fatalf("expected record to remain stored")
	}
	if roundTrip.Messages[0].Text != "hi" {
		t.Fatalf("expected Get to return a copy of stored messages")
	}
}
