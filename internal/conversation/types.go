package conversation

import "time"

type Message struct {
	Role string
	Text string
}

type Request struct {
	System   []string
	Messages []Message
}

type Record struct {
	ResponseID string
	ModelID    string
	Messages   []Message
	CreatedAt  time.Time
}

type Store interface {
	Get(string) (Record, bool)
	Save(Record)
}
