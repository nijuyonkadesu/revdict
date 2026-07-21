package queryclient

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type fakeExecutor struct {
	output  []byte
	err     error
	gotArgs []string
}

func (f *fakeExecutor) Run(ctx context.Context, args ...string) ([]byte, error) {
	f.gotArgs = args
	return f.output, f.err
}

func TestQueryBuildsCorrectJSONRequest(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)
	req := Request{Query: "happy", TopN: 30, Sort: "most_formal", Category: "all"}

	_, err := c.Query(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(fake.gotArgs) != 2 || fake.gotArgs[0] != "--tui-query" {
		t.Fatalf("expected [--tui-query, <json>], got %v", fake.gotArgs)
	}
	if !strings.Contains(fake.gotArgs[1], `"query":"happy"`) {
		t.Fatalf("expected query in JSON payload, got %s", fake.gotArgs[1])
	}
	if !strings.Contains(fake.gotArgs[1], `"sort":"most_formal"`) {
		t.Fatalf("expected sort in JSON payload, got %s", fake.gotArgs[1])
	}
}

func TestQueryOmitsUnsetPhoneticFields(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)
	req := Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all"}

	_, err := c.Query(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(fake.gotArgs[1], "rhymes_with") {
		t.Fatalf("expected rhymes_with omitted when unset, got %s", fake.gotArgs[1])
	}
}

func TestQueryDistinguishesUnsetSyllablesFromExplicitZero(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)
	zero := 0
	req := Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all", Syllables: &zero}

	_, err := c.Query(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(fake.gotArgs[1], `"syllables":0`) {
		t.Fatalf("expected explicit syllables:0 in payload, got %s", fake.gotArgs[1])
	}
}

func TestQueryParsesJSONLResponseRows(t *testing.T) {
	output := `{"headword":"joyful","pos":"adjective","definition":"feeling great happiness","stress":null,"label":"joy","polarity":"positive","synonyms":["glad"],"examples":[],"relevance":90,"is_exact":false}
`
	fake := &fakeExecutor{output: []byte(output)}
	c := NewWithExecutor(fake)

	rows, err := c.Query(context.Background(), Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	if rows[0].Headword != "joyful" || rows[0].Relevance != 90 {
		t.Fatalf("unexpected row: %+v", rows[0])
	}
	if rows[0].Stress != nil {
		t.Fatalf("expected nil stress, got %v", *rows[0].Stress)
	}
	if len(rows[0].Synonyms) != 1 || rows[0].Synonyms[0] != "glad" {
		t.Fatalf("unexpected synonyms: %v", rows[0].Synonyms)
	}
}

func TestQueryReturnsEmptyRowsForBlankOutput(t *testing.T) {
	fake := &fakeExecutor{output: []byte("")}
	c := NewWithExecutor(fake)

	rows, err := c.Query(context.Background(), Request{Query: "", TopN: 30, Sort: "relevance", Category: "all"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("expected 0 rows, got %d", len(rows))
	}
}

func TestQueryPropagatesExecutorError(t *testing.T) {
	// Mirrors real execExecutor behavior: revdict prints its diagnostic to
	// stdout and exits non-zero, so the Go error itself is just the generic
	// "exit status 1" -- the useful text lives in the captured output.
	fake := &fakeExecutor{
		output: []byte("revdict: error: Unknown sort mode: 'bogus'"),
		err:    errors.New("exit status 1"),
	}
	c := NewWithExecutor(fake)

	_, err := c.Query(context.Background(), Request{Query: "happy", TopN: 30, Sort: "bogus", Category: "all"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "revdict: error: Unknown sort mode: 'bogus'") {
		t.Fatalf("expected error to contain the revdict diagnostic, got %q", err.Error())
	}
}

func TestQueryFallsBackToExecutorErrorWhenOutputEmpty(t *testing.T) {
	// Covers cases where the subprocess fails before producing any stdout
	// (e.g. revdict not found on PATH) -- Query must still surface an error.
	fake := &fakeExecutor{
		output: []byte(""),
		err:    errors.New("exec: \"revdict\": executable file not found in $PATH"),
	}
	c := NewWithExecutor(fake)

	_, err := c.Query(context.Background(), Request{Query: "happy", TopN: 30, Sort: "relevance", Category: "all"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "executable file not found") {
		t.Fatalf("expected fallback to original executor error, got %q", err.Error())
	}
}
