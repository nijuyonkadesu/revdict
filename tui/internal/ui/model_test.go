package ui

import (
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
)

func testRows() []queryclient.ResultRow {
	return []queryclient.ResultRow{
		{Headword: "annoyance", POS: "noun", Definition: "a feeling of being bothered", Relevance: 92},
		{Headword: "irritation", POS: "noun", Definition: "a feeling of anger about something", Relevance: 88},
	}
}

func TestNewModelStartsWithSearchFocused(t *testing.T) {
	m := NewModel(testRows())
	if !m.input.Focused() {
		t.Fatal("expected search input to be focused on start")
	}
}

func TestTypingAppendsToQuery(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m = mm.(Model)
	if m.input.Value() != "h" {
		t.Fatalf("expected query 'h', got %q", m.input.Value())
	}
}

func TestDownMovesSelectionToNextResult(t *testing.T) {
	m := NewModel(testRows())
	m.selected = 0
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = mm.(Model)
	if m.selected != 1 {
		t.Fatalf("expected selected=1, got %d", m.selected)
	}
}

func TestDownAtLastResultStaysPut(t *testing.T) {
	m := NewModel(testRows())
	m.selected = 1
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = mm.(Model)
	if m.selected != 1 {
		t.Fatalf("expected selected to stay at 1, got %d", m.selected)
	}
}

func TestEscClearsNonEmptyQuery(t *testing.T) {
	m := NewModel(testRows())
	m.input.SetValue("something")
	mm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.input.Value() != "" {
		t.Fatalf("expected query cleared, got %q", m.input.Value())
	}
	if cmd != nil {
		msg := cmd()
		if _, isQuit := msg.(tea.QuitMsg); isQuit {
			t.Fatal("expected Esc on a non-empty query not to quit")
		}
	}
}

func TestSecondEscOnAlreadyEmptyQueryQuits(t *testing.T) {
	m := NewModel(testRows())
	m.input.SetValue("")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if cmd == nil {
		t.Fatal("expected a quit command when Esc pressed on an already-empty query")
	}
	if _, isQuit := cmd().(tea.QuitMsg); !isQuit {
		t.Fatal("expected tea.QuitMsg")
	}
}

func TestCtrlCAlwaysQuits(t *testing.T) {
	m := NewModel(testRows())
	m.input.SetValue("something")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd == nil {
		t.Fatal("expected a quit command")
	}
	if _, isQuit := cmd().(tea.QuitMsg); !isQuit {
		t.Fatal("expected tea.QuitMsg")
	}
}

func TestF2TogglesPreviewVisibility(t *testing.T) {
	m := NewModel(testRows())
	initial := m.previewVisible
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF2})
	m = mm.(Model)
	if m.previewVisible == initial {
		t.Fatal("expected previewVisible to flip")
	}
}

func TestViewIncludesHighlightedHeadwordAndWrappedPreview(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)
	out := m.View()
	if !strings.Contains(out, "annoyance") {
		t.Fatalf("expected view to contain the first result's headword, got: %s", out)
	}
}

func manyTestRows(n int) []queryclient.ResultRow {
	rows := make([]queryclient.ResultRow, n)
	for i := range rows {
		rows[i] = queryclient.ResultRow{Headword: fmt.Sprintf("word%d", i), POS: "noun"}
	}
	return rows
}

func TestVisibleRowRangeClampsToAvailableHeightOnAShortTerminal(t *testing.T) {
	m := NewModel(manyTestRows(30))
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 10})
	m = mm.(Model)
	start, end := m.visibleRowRange()
	if end-start > m.height-2 {
		t.Fatalf("expected visible range to fit within height-2 rows, got start=%d end=%d height=%d", start, end, m.height)
	}
	if end-start >= len(m.rows) {
		t.Fatalf("expected visible range to be a strict subset of 30 rows on a 10-row terminal, got %d rows", end-start)
	}
}

func TestVisibleRowRangeKeepsSelectionInsideTheWindowWhenScrolledDown(t *testing.T) {
	m := NewModel(manyTestRows(30))
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 10})
	m = mm.(Model)
	m.selected = 25
	start, end := m.visibleRowRange()
	if m.selected < start || m.selected >= end {
		t.Fatalf("expected selected=25 to fall within visible range [%d,%d)", start, end)
	}
}

func TestVisibleRowRangeShowsAllRowsWhenTheyFit(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)
	start, end := m.visibleRowRange()
	if start != 0 || end != len(m.rows) {
		t.Fatalf("expected the full 2-row set to be visible on a tall terminal, got start=%d end=%d", start, end)
	}
}

// TestResultsListRowIsTruncatedNotWrappedOnANarrowResultsColumn guards
// against the results list's manual rendering loop feeding an overlong row
// string into lipgloss's Width().Render(), which word-wraps (rather than
// truncates) any line wider than the column -- silently turning one logical
// row into multiple physical lines and defeating visibleRowRange's height
// clamp (the exact terminal-corruption failure mode visibleRowRange and
// tea.WithAltScreen() exist to prevent). A single overlong row must render
// to exactly one physical line, truncated with an ellipsis.
func TestResultsListRowIsTruncatedNotWrappedOnANarrowResultsColumn(t *testing.T) {
	rows := []queryclient.ResultRow{
		{Headword: "counterrevolutionary", POS: "adjective", Definition: "opposing a revolution"},
	}
	m := NewModel(rows) // previewVisible defaults to true
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 40, Height: 24})
	m = mm.(Model)
	out := m.View()

	// With previewVisible=true and width=40, the results column is
	// listWidth = 40/2 = 20 runes -- narrower than the ~34-rune formatted
	// row ("  counterrevolutionary (adjective)"), so this row must be
	// truncated to fit.
	const listWidth = 20

	if !strings.Contains(out, "…") {
		t.Fatalf("expected the overlong row to be truncated with an ellipsis, got:\n%s", out)
	}
	if strings.Contains(out, "counterrevolutionary (adjective)") {
		t.Fatalf("expected the overlong row to be truncated, but the full untruncated text appears in the view:\n%s", out)
	}

	// Directly check for a wrapped continuation: isolate each physical
	// line's results-column slice (the first listWidth runes -- the
	// preview column, if any, starts at rune index >= listWidth) and
	// confirm none of them is just the bare headword or POS on its own
	// line, which is what lipgloss's word-wrap would produce pre-fix.
	for i, line := range strings.Split(out, "\n") {
		runes := []rune(line)
		col := line
		if len(runes) >= listWidth {
			col = string(runes[:listWidth])
		}
		trimmed := strings.TrimSpace(col)
		if trimmed == "counterrevolutionary" || trimmed == "(adjective)" {
			t.Fatalf("line %d's results column looks like a wrapped continuation of the row rather than a single truncated line: %q\nfull view:\n%s", i, col, out)
		}
	}
}
