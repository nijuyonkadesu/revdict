package queryclient

import (
	"context"
	"encoding/json"
	"os/exec"
	"strings"
)

// Request mirrors --tui-query's expected JSON shape, which itself mirrors
// the revdict daemon's existing wire-protocol field names exactly.
type Request struct {
	Query        string `json:"query"`
	TopN         int    `json:"top_n"`
	Sort         string `json:"sort"`
	Category     string `json:"category"`
	Syllables    *int   `json:"syllables,omitempty"`
	PrimaryVowel string `json:"primary_vowel,omitempty"`
	RhymesWith   string `json:"rhymes_with,omitempty"`
	SoundsLike   string `json:"sounds_like,omitempty"`
	Meter        string `json:"meter,omitempty"`
}

// ResultRow mirrors one JSONL row --tui-query emits (identical to what
// --jsonl-query already emits).
type ResultRow struct {
	Headword   string   `json:"headword"`
	POS        string   `json:"pos"`
	Definition string   `json:"definition"`
	Stress     *string  `json:"stress"`
	Label      string   `json:"label"`
	Polarity   string   `json:"polarity"`
	Synonyms   []string `json:"synonyms"`
	Examples   []string `json:"examples"`
	Relevance  int      `json:"relevance"`
	IsExact    bool     `json:"is_exact"`
}

// Executor runs `revdict` with the given args and returns its stdout.
// Swapped for a fake in tests so no real subprocess is spawned.
type Executor interface {
	Run(ctx context.Context, args ...string) ([]byte, error)
}

type execExecutor struct{}

func (execExecutor) Run(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "revdict", args...).Output()
}

type Client struct {
	executor Executor
}

// New returns a Client that shells out to the real `revdict` binary on PATH.
func New() *Client {
	return &Client{executor: execExecutor{}}
}

// NewWithExecutor returns a Client backed by a caller-supplied Executor --
// used by tests to avoid spawning a real subprocess.
func NewWithExecutor(e Executor) *Client {
	return &Client{executor: e}
}

// Query builds the --tui-query JSON payload, runs it, and parses the JSONL
// response into rows. Cancelling ctx (e.g. because a newer keystroke
// superseded this request) aborts the in-flight subprocess.
func (c *Client) Query(ctx context.Context, req Request) ([]ResultRow, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	out, err := c.executor.Run(ctx, "--tui-query", string(payload))
	if err != nil {
		return nil, err
	}

	var rows []ResultRow
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		var row ResultRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, nil
}
