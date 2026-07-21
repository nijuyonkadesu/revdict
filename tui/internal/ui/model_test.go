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
