package ui

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
)

// copyFunc is called with the selected headword on Enter and reports
// whether the copy succeeded. Task 7 wires this to the real clipboard
// mechanism (clipboard.Copy, which can fail e.g. with no clipboard
// utility available); tests and this task's own construction path use a
// no-op default so this file has no clipboard dependency of its own.
type copyFunc func(headword string) error

type Model struct {
	input          textinput.Model
	preview        viewport.Model
	rows           []queryclient.ResultRow
	selected       int
	previewVisible bool
	width, height  int
	statusMessage  string
	onCopy         copyFunc
}

func NewModel(rows []queryclient.ResultRow) Model {
	ti := textinput.New()
	ti.Focus()
	vp := viewport.New(0, 0)
	return Model{
		input:          ti,
		preview:        vp,
		rows:           rows,
		previewVisible: true,
		onCopy:         func(string) error { return nil },
	}
}

func (m Model) Init() tea.Cmd {
	return m.input.Focus()
}

func (m Model) selectedRow() (queryclient.ResultRow, bool) {
	if len(m.rows) == 0 || m.selected < 0 || m.selected >= len(m.rows) {
		return queryclient.ResultRow{}, false
	}
	return m.rows[m.selected], true
}

func (m *Model) refreshPreview() {
	row, ok := m.selectedRow()
	if !ok {
		m.preview.SetContent("")
		return
	}
	previewWidth := m.width / 2
	if previewWidth < 1 {
		previewWidth = 1
	}
	text := fmt.Sprintf("%s\n\n%s", row.Headword, row.Definition)
	wrapped := lipgloss.NewStyle().Width(previewWidth).Render(text)
	m.preview.SetContent(wrapped)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.preview.Width = msg.Width / 2
		m.preview.Height = msg.Height - 4
		m.refreshPreview()
		return m, nil

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyEsc:
			if m.input.Value() == "" {
				return m, tea.Quit
			}
			m.input.SetValue("")
			return m, nil

		case tea.KeyEnter:
			if row, ok := m.selectedRow(); ok {
				if err := m.onCopy(row.Headword); err != nil {
					m.statusMessage = "Copy failed: " + err.Error()
				} else {
					m.statusMessage = "Copied: " + row.Headword
				}
			}
			return m, nil

		case tea.KeyUp:
			if m.selected > 0 {
				m.selected--
				m.refreshPreview()
			}
			return m, nil

		case tea.KeyDown:
			if m.selected < len(m.rows)-1 {
				m.selected++
				m.refreshPreview()
			}
			return m, nil

		case tea.KeyF2:
			m.previewVisible = !m.previewVisible
			return m, nil
		}

		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		return m, cmd
	}

	return m, nil
}

// visibleRowRange returns the [start, end) slice of m.rows that fits within
// the terminal's available height, centered on the current selection. The
// results list is rendered as a manual loop (not bubbles/list), so nothing
// else clamps it -- without this, a 30-row result set on a short terminal
// would render taller than the screen and corrupt the display (bubbletea's
// renderer does not auto-scroll the root View).
func (m Model) visibleRowRange() (int, int) {
	maxVisible := m.height - 2 // input line + status line
	if maxVisible < 1 {
		maxVisible = 1
	}
	if len(m.rows) <= maxVisible {
		return 0, len(m.rows)
	}
	start := m.selected - maxVisible/2
	if start < 0 {
		start = 0
	}
	end := start + maxVisible
	if end > len(m.rows) {
		end = len(m.rows)
		start = end - maxVisible
		if start < 0 {
			start = 0
		}
	}
	return start, end
}

// truncateToWidth truncates s to at most width runes, replacing the last
// rune with an ellipsis when truncation occurs. It operates on runes (not
// bytes) so it's safe for non-ASCII headwords in the corpus (e.g. loanwords
// like "café"/"naïve").
func truncateToWidth(s string, width int) string {
	runes := []rune(s)
	if len(runes) <= width {
		return s
	}
	if width <= 1 {
		return string(runes[:width])
	}
	return string(runes[:width-1]) + "…"
}

func (m Model) View() string {
	listWidth := m.width
	if m.previewVisible {
		listWidth = m.width / 2
	}

	var b []string
	start, end := m.visibleRowRange()
	for i := start; i < end; i++ {
		row := m.rows[i]
		marker := "  "
		if i == m.selected {
			marker = "> "
		}
		line := fmt.Sprintf("%s%s (%s)", marker, row.Headword, row.POS)
		if listWidth > 0 {
			line = truncateToWidth(line, listWidth)
		}
		b = append(b, line)
	}
	resultsView := lipgloss.JoinVertical(lipgloss.Left, b...)

	var body string
	if m.previewVisible {
		left := lipgloss.NewStyle().Width(listWidth).Render(resultsView)
		right := lipgloss.NewStyle().Width(m.width - listWidth).Render(m.preview.View())
		body = lipgloss.JoinHorizontal(lipgloss.Top, left, right)
	} else {
		body = resultsView
	}

	status := m.statusMessage
	return lipgloss.JoinVertical(lipgloss.Left, m.input.View(), status, body)
}
