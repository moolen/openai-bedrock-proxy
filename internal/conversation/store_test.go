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
			{
				Role: "user",
				Blocks: []Block{
					{Type: BlockTypeText, Text: "hi"},
					{
						Type: BlockTypeToolResult,
						ToolResult: &ToolResult{
							CallID: "call_1",
							Output: map[string]any{"ok": true},
						},
					},
				},
			},
		},
	}

	store.Save(original)
	original.Messages[0].Blocks[0].Text = "mutated"
	original.Messages[0].Blocks[1].ToolResult.CallID = "mutated_call"
	original.Messages[0].Blocks[1].ToolResult.Output.(map[string]any)["ok"] = false

	stored, ok := store.Get("r1")
	if !ok {
		t.Fatalf("expected record to be stored")
	}
	if stored.Messages[0].Blocks[0].Text != "hi" {
		t.Fatalf("expected stored record to be immutable after Save")
	}
	if stored.Messages[0].Blocks[1].ToolResult.CallID != "call_1" {
		t.Fatalf("expected tool result call id to remain unchanged after Save")
	}
	if stored.Messages[0].Blocks[1].ToolResult.Output.(map[string]any)["ok"] != true {
		t.Fatalf("expected nested tool result output to remain unchanged after Save")
	}

	stored.Messages[0].Blocks[0].Text = "changed"
	stored.Messages[0].Blocks[1].ToolResult.CallID = "changed_call"
	stored.Messages[0].Blocks[1].ToolResult.Output.(map[string]any)["ok"] = false
	roundTrip, ok := store.Get("r1")
	if !ok {
		t.Fatalf("expected record to remain stored")
	}
	if roundTrip.Messages[0].Blocks[0].Text != "hi" {
		t.Fatalf("expected Get to return a copy of stored messages")
	}
	if roundTrip.Messages[0].Blocks[1].ToolResult.CallID != "call_1" {
		t.Fatalf("expected Get to deep-clone nested tool result state")
	}
	if roundTrip.Messages[0].Blocks[1].ToolResult.Output.(map[string]any)["ok"] != true {
		t.Fatalf("expected Get to deep-clone nested output payloads")
	}
}

func TestRecordFromResponseClonesNestedBlocks(t *testing.T) {
	snapshot := Request{
		Messages: []Message{
			{
				Role: "assistant",
				Blocks: []Block{
					{Type: BlockTypeText, Text: "ok"},
					{
						Type: BlockTypeToolCall,
						ToolCall: &ToolCall{
							ID:        "call_1",
							Name:      "lookup",
							Arguments: `{"q":"x"}`,
						},
					},
					{
						Type: BlockTypeToolResult,
						ToolResult: &ToolResult{
							CallID: "call_1",
							Output: map[string]any{"items": []any{"a", "b"}},
						},
					},
				},
			},
		},
	}

	record := RecordFromResponse("resp_1", "model", snapshot)
	snapshot.Messages[0].Blocks[0].Text = "mutated"
	snapshot.Messages[0].Blocks[1].ToolCall.Name = "mutated_lookup"
	snapshot.Messages[0].Blocks[2].ToolResult.Output.(map[string]any)["items"].([]any)[0] = "changed"

	if record.Messages[0].Blocks[0].Text != "ok" {
		t.Fatalf("expected text block to be cloned, got %#v", record.Messages[0].Blocks[0])
	}
	if record.Messages[0].Blocks[1].ToolCall.Name != "lookup" {
		t.Fatalf("expected tool call to be cloned, got %#v", record.Messages[0].Blocks[1].ToolCall)
	}
	items := record.Messages[0].Blocks[2].ToolResult.Output.(map[string]any)["items"].([]any)
	if items[0] != "a" {
		t.Fatalf("expected nested output payload to be cloned, got %#v", record.Messages[0].Blocks[2].ToolResult.Output)
	}
}
