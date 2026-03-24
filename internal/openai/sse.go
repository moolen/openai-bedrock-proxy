package openai

import (
	"encoding/json"
	"fmt"
	"io"
)

type flushWriter interface {
	io.Writer
	Flush()
}

func WriteEvent(w io.Writer, name string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, data)
	if err != nil {
		return err
	}

	if flusher, ok := w.(flushWriter); ok {
		flusher.Flush()
	}

	return nil
}
