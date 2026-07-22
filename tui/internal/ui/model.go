package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
)

// copyFunc is called with the selected headword on Enter and reports
// whether the copy succeeded. Task 7 wires this to the real clipboard
// mechanism (clipboard.Copy, which can fail e.g. with no clipboard
// utility available); tests and this task's own construction path use a
// no-op default so this file has no clipboard dependency of its own.
type copyFunc func(headword string) error

type screenID int

const (
	screenSearch screenID = iota
	screenPanel
	screenHelp
)

type Model struct {
	input          textinput.Model
	preview        viewport.Model
	rows           []queryclient.ResultRow
	selected       int
	previewVisible bool
	width, height  int
	statusMessage  string
	statusIsError  bool // true when statusMessage holds a query-error banner (set by queryErrorMsg); the next successful queryResultMsg clears it. Enter's copy feedback ("Copied: X"/"Copy failed: ...") sets this false so an unrelated, later-resolving query can never silently wipe a copy confirmation the user just triggered.
	onCopy         copyFunc
	client         *queryclient.Client
	filters        FilterState
	cancelInFlight context.CancelFunc
	screen         screenID
	panel          panelState
}

// FilterState holds the currently-active sort/category/phonetic filters.
// The zero value is NOT valid -- always construct via NewFilterState.
type FilterState struct {
	Sort         string
	Category     string
	Syllables    *int
	PrimaryVowel string
	RhymesWith   string
	SoundsLike   string
	Meter        string
}

func NewFilterState() FilterState {
	return FilterState{Sort: "relevance", Category: "all"}
}

func (f FilterState) toRequest(query string) queryclient.Request {
	return queryclient.Request{
		Query: query, TopN: 30, Sort: f.Sort, Category: f.Category,
		Syllables: f.Syllables, PrimaryVowel: f.PrimaryVowel,
		RhymesWith: f.RhymesWith, SoundsLike: f.SoundsLike, Meter: f.Meter,
	}
}

// summary renders a persistent, one-line synopsis of the currently active
// sort/category/phonetic filters (e.g. "sort:relevance  cat:all"). Unlike
// Model.statusMessage (a transient copy-confirmation/query-error banner that
// clears or changes), this line always reflects the current FilterState and
// is shown even when the filter panel is closed, per the design spec.
func (f FilterState) summary() string {
	parts := []string{"sort:" + f.Sort, "cat:" + f.Category}
	if f.Syllables != nil {
		parts = append(parts, fmt.Sprintf("syl:%d", *f.Syllables))
	}
	if f.PrimaryVowel != "" {
		parts = append(parts, "vowel:"+f.PrimaryVowel)
	}
	if f.RhymesWith != "" {
		parts = append(parts, "rhymes:"+f.RhymesWith)
	}
	if f.SoundsLike != "" {
		parts = append(parts, "soundslike:"+f.SoundsLike)
	}
	if f.Meter != "" {
		parts = append(parts, "meter:"+f.Meter)
	}
	return strings.Join(parts, "  ")
}

type debounceFiredMsg struct{ query string }
type queryResultMsg struct {
	query string
	rows  []queryclient.ResultRow
}
type queryErrorMsg struct {
	query string
	err   error
}

const debounceDelay = 100 * time.Millisecond

func debounceCmd(query string) tea.Cmd {
	return tea.Tick(debounceDelay, func(time.Time) tea.Msg {
		return debounceFiredMsg{query: query}
	})
}

func runQueryCmd(ctx context.Context, client *queryclient.Client, query string, req queryclient.Request) tea.Cmd {
	return func() tea.Msg {
		rows, err := client.Query(ctx, req)
		if err != nil {
			return queryErrorMsg{query: query, err: err}
		}
		return queryResultMsg{query: query, rows: rows}
	}
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
		filters:        NewFilterState(),
	}
}

func NewLiveModel(client *queryclient.Client) Model {
	m := NewModel(nil)
	m.client = client
	m.filters = NewFilterState()
	return m
}

// SetCopyFunc overrides the clipboard behavior invoked on Enter. Exported
// so cmd/revdict-tui can wire in the real clipboard.Copy without this
// package needing to import the clipboard package itself (keeping
// dependency direction one-way: cmd -> ui -> queryclient, never ui ->
// clipboard).
func (m *Model) SetCopyFunc(f func(string) error) {
	m.onCopy = f
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

	var b strings.Builder
	b.WriteString(row.Headword)
	if row.Stress != nil && *row.Stress != "" {
		b.WriteString("\n" + *row.Stress)
	}
	b.WriteString("\n\n")
	b.WriteString(row.Definition)
	if len(row.Synonyms) > 0 {
		b.WriteString("\n\nSynonyms: " + strings.Join(row.Synonyms, ", "))
	}
	if len(row.Examples) > 0 {
		b.WriteString("\n\nExamples:")
		for _, ex := range row.Examples {
			b.WriteString("\n- " + ex)
		}
	}

	wrapped := lipgloss.NewStyle().Width(previewWidth).Render(b.String())
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

	case debounceFiredMsg:
		if msg.query != m.input.Value() {
			return m, nil
		}
		if m.client == nil {
			return m, nil
		}
		ctx, cancel := context.WithCancel(context.Background())
		if m.cancelInFlight != nil {
			m.cancelInFlight()
		}
		m.cancelInFlight = cancel
		return m, runQueryCmd(ctx, m.client, msg.query, m.filters.toRequest(msg.query))

	case queryResultMsg:
		if msg.query != m.input.Value() {
			return m, nil
		}
		m.rows = msg.rows
		m.selected = 0
		if m.statusIsError {
			m.statusMessage = ""
			m.statusIsError = false
		}
		m.refreshPreview()
		return m, nil

	case queryErrorMsg:
		if msg.query != m.input.Value() {
			return m, nil
		}
		m.statusMessage = msg.err.Error()
		m.statusIsError = true
		return m, nil

	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}

		if m.screen == screenPanel {
			if msg.Type == tea.KeyEsc {
				m.filters = m.panel.toFilterState()
				m.screen = screenSearch
				return m, debounceCmd(m.input.Value())
			}
			m.panel = m.panel.handleKey(msg)
			return m, nil
		}

		if m.screen == screenHelp {
			if msg.Type == tea.KeyEsc {
				m.screen = screenSearch
			}
			return m, nil
		}

		switch msg.Type {
		case tea.KeyTab:
			m.panel = newPanelState(m.filters)
			m.screen = screenPanel
			return m, nil

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
				m.statusIsError = false
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

		case tea.KeyF1:
			m.screen = screenHelp
			return m, nil

		case tea.KeyCtrlR:
			m.filters.Sort = nextSortMode(m.filters.Sort)
			return m, debounceCmd(m.input.Value())
		}

		var inputCmd tea.Cmd
		m.input, inputCmd = m.input.Update(msg)
		return m, tea.Batch(inputCmd, debounceCmd(m.input.Value()))
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
	maxVisible := m.height - 3 // input line + filter summary line + status line
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
	if m.screen == screenPanel {
		return m.panel.View()
	}

	if m.screen == screenHelp {
		return helpText
	}

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

	filterSummary := m.filters.summary()
	if m.width > 0 {
		filterSummary = truncateToWidth(filterSummary, m.width)
	}
	status := m.statusMessage
	return lipgloss.JoinVertical(lipgloss.Left, m.input.View(), filterSummary, status, body)
}
