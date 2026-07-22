package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestNewPanelStartsOnSortField(t *testing.T) {
	p := newPanelState(NewFilterState())
	if p.focusedField != fieldSort {
		t.Fatalf("expected initial focus on sort field, got %d", p.focusedField)
	}
}

func TestTabAdvancesThroughAllSevenFields(t *testing.T) {
	p := newPanelState(NewFilterState())
	seen := []int{p.focusedField}
	for i := 0; i < 6; i++ {
		p = p.handleKey(tea.KeyMsg{Type: tea.KeyTab})
		seen = append(seen, p.focusedField)
	}
	want := []int{fieldSort, fieldCategory, fieldSyllables, fieldPrimaryVowel, fieldRhymesWith, fieldSoundsLike, fieldMeter}
	for i, w := range want {
		if seen[i] != w {
			t.Fatalf("field order mismatch at %d: got %d, want %d", i, seen[i], w)
		}
	}
	// Tab wraps back to the first field.
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	if p.focusedField != fieldSort {
		t.Fatalf("expected wraparound to fieldSort, got %d", p.focusedField)
	}
}

func TestDownMovesSortSelection(t *testing.T) {
	p := newPanelState(NewFilterState())
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if p.sortSelected != 1 {
		t.Fatalf("expected sortSelected=1, got %d", p.sortSelected)
	}
	if p.toFilterState().Sort != sortModes[1] {
		t.Fatalf("expected filter state sort=%s, got %s", sortModes[1], p.toFilterState().Sort)
	}
}

func TestCategoryFieldNavigatesIndependentlyOfSort(t *testing.T) {
	p := newPanelState(NewFilterState())
	p.focusedField = fieldCategory
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if p.categorySelected != 1 {
		t.Fatalf("expected categorySelected=1, got %d", p.categorySelected)
	}
	if p.toFilterState().Category != categories[1] {
		t.Fatalf("expected filter state category=%s, got %s", categories[1], p.toFilterState().Category)
	}
}

func TestSyllablesFieldOnlyAcceptsDigits(t *testing.T) {
	p := newPanelState(NewFilterState())
	p.focusedField = fieldSyllables
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'2'}})
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	if p.syllablesText != "2" {
		t.Fatalf("expected non-digit rejected, syllablesText=%q", p.syllablesText)
	}
	fs := p.toFilterState()
	if fs.Syllables == nil || *fs.Syllables != 2 {
		t.Fatalf("expected Syllables=2, got %v", fs.Syllables)
	}
}

func TestMeterFieldOnlyAcceptsSlashAndX(t *testing.T) {
	p := newPanelState(NewFilterState())
	p.focusedField = fieldMeter
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'q'}})
	p = p.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	if p.meterText != "/x" {
		t.Fatalf("expected invalid char rejected, meterText=%q", p.meterText)
	}
}

func TestTabInMainScreenOpensThePanel(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = mm.(Model)
	if m.screen != screenPanel {
		t.Fatalf("expected screen=screenPanel, got %v", m.screen)
	}
}

func TestEscInPanelClosesItAndReturnsToSearch(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = mm.(Model)

	// Move the sort selection while the panel is open, so closing the panel
	// can be verified to actually apply the panel's edited state (via
	// toFilterState()) to m.filters, not just flip m.screen back.
	mm, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = mm.(Model)

	mm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.screen != screenSearch {
		t.Fatalf("expected screen=screenSearch after Esc, got %v", m.screen)
	}
	if m.filters.Sort != sortModes[1] {
		t.Fatalf("expected Esc to apply the panel's edited sort selection (%s) to m.filters, got %q", sortModes[1], m.filters.Sort)
	}
}
