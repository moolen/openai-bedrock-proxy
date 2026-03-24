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
