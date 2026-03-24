package openai

import (
	"encoding/json"
	"fmt"
	"io"
)

func WriteEvent(w io.Writer, name string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, data)
	return err
}
