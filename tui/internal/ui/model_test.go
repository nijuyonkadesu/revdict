package ui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/cursor"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
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

func TestViewShowsThePersistentFilterSummaryLine(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)
	out := m.View()
	if !strings.Contains(out, "sort:relevance") || !strings.Contains(out, "cat:all") {
		t.Fatalf("expected the view to show the persistent filter summary, got: %s", out)
	}
}

func TestCtrlCQuitsEvenWhileThePanelIsOpen(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = mm.(Model)
	if m.screen != screenPanel {
		t.Fatal("expected the panel to be open")
	}
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd == nil {
		t.Fatal("expected a quit command")
	}
	if _, isQuit := cmd().(tea.QuitMsg); !isQuit {
		t.Fatal("expected tea.QuitMsg")
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
	if end-start > m.height-3 {
		t.Fatalf("expected visible range to fit within height-3 rows, got start=%d end=%d height=%d", start, end, m.height)
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

func TestFilterSummaryLineIsTruncatedOnANarrowTerminal(t *testing.T) {
	m := NewModel(testRows())
	m.filters = FilterState{
		Sort: "relevance", Category: "all",
		RhymesWith: "a-very-long-rhymes-with-value-that-would-otherwise-wrap-the-terminal",
	}
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 30, Height: 24})
	m = mm.(Model)
	out := m.View()
	for _, line := range strings.Split(out, "\n") {
		if lipgloss.Width(line) > 30 {
			t.Fatalf("expected every line to fit within width 30, got a %d-wide line: %q", lipgloss.Width(line), line)
		}
	}
}

type fakeExecutor struct {
	calls [][]string
	ctxs  []context.Context
	err   error // when set, Run returns this error instead of the canned success payload
}

func (f *fakeExecutor) Run(ctx context.Context, args ...string) ([]byte, error) {
	f.calls = append(f.calls, args)
	f.ctxs = append(f.ctxs, ctx)
	if f.err != nil {
		return nil, f.err
	}
	return []byte(`{"headword":"annoyance","pos":"noun","definition":"a feeling","stress":null,"label":"joy","polarity":"positive","synonyms":[],"examples":[],"relevance":92,"is_exact":false}` + "\n"), nil
}

// findDebounceFiredMsg extracts a debounceFiredMsg from the result of
// invoking a tea.Cmd. Production code batches the textinput's own command
// (e.g. its cursor-blink re-arm) together with the debounce command via
// tea.Batch, which the real bubbletea runtime unwraps and dispatches
// independently before Update ever sees it. Tests that call a Cmd directly
// (bypassing the runtime) must therefore unwrap a tea.BatchMsg themselves to
// find the sub-command they care about.
func findDebounceFiredMsg(t *testing.T, msg tea.Msg) debounceFiredMsg {
	t.Helper()
	if debounce, ok := msg.(debounceFiredMsg); ok {
		return debounce
	}
	if batch, ok := msg.(tea.BatchMsg); ok {
		for _, sub := range batch {
			if sub == nil {
				continue
			}
			if debounce, ok := sub().(debounceFiredMsg); ok {
				return debounce
			}
		}
	}
	t.Fatalf("expected debounceFiredMsg (directly or within a tea.BatchMsg), got %T", msg)
	return debounceFiredMsg{}
}

// findSpinnerTickMsg extracts a spinner.TickMsg from the result of invoking
// a tea.Cmd, mirroring findDebounceFiredMsg above: production code batches
// the spinner's own tick command together with the debounce (and, for plain
// typing, the textinput's) command via tea.Batch, so a test calling the Cmd
// directly must unwrap a tea.BatchMsg to find the sub-command it cares
// about. Non-matching subs are still invoked (so their real delays elapse,
// same as findDebounceFiredMsg does for its non-matching subs) but their
// results are discarded, and the failure is only reported once the whole
// batch has been searched.
func findSpinnerTickMsg(t *testing.T, msg tea.Msg) spinner.TickMsg {
	t.Helper()
	if tick, ok := msg.(spinner.TickMsg); ok {
		return tick
	}
	if batch, ok := msg.(tea.BatchMsg); ok {
		for _, sub := range batch {
			if sub == nil {
				continue
			}
			if tick, ok := sub().(spinner.TickMsg); ok {
				return tick
			}
		}
	}
	t.Fatalf("expected spinner.TickMsg (directly or within a tea.BatchMsg), got %T", msg)
	return spinner.TickMsg{}
}

func TestTypingSchedulesADebouncedQuery(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)

	mm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m = mm.(Model)
	if cmd == nil {
		t.Fatal("expected a debounce command to be scheduled")
	}

	msg := cmd()
	debounce := findDebounceFiredMsg(t, msg)
	if debounce.query != "h" {
		t.Fatalf("expected debounce for query 'h', got %q", debounce.query)
	}
}

func TestStaleDebounceIsIgnoredIfQueryChangedSince(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("current")

	mm, cmd := m.Update(debounceFiredMsg{query: "stale"})
	m = mm.(Model)
	if cmd != nil {
		t.Fatal("expected no query dispatched for a stale debounce message")
	}
}

func TestFreshDebounceDispatchesAQuery(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("annoyance")

	mm, cmd := m.Update(debounceFiredMsg{query: "annoyance"})
	m = mm.(Model)
	if cmd == nil {
		t.Fatal("expected a query command to be dispatched")
	}
	msg := cmd()
	result, ok := msg.(queryResultMsg)
	if !ok {
		t.Fatalf("expected queryResultMsg, got %T", msg)
	}
	if len(result.rows) != 1 || result.rows[0].Headword != "annoyance" {
		t.Fatalf("unexpected rows: %v", result.rows)
	}
}

func TestQueryResultMsgReplacesRows(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)

	mm, _ := m.Update(queryResultMsg{query: "", rows: []queryclient.ResultRow{{Headword: "new-word"}}})
	m = mm.(Model)
	if len(m.rows) != 1 || m.rows[0].Headword != "new-word" {
		t.Fatalf("expected rows replaced with query result, got %v", m.rows)
	}
}

func TestStaleQueryResultIsIgnoredIfSupersededByNewerQuery(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("hex")

	mm, _ := m.Update(queryResultMsg{query: "he", rows: []queryclient.ResultRow{{Headword: "stale-word"}}})
	m = mm.(Model)
	if len(m.rows) != 0 {
		t.Fatalf("expected a stale result (query 'he' while input is 'hex') to be ignored, got rows=%v", m.rows)
	}

	mm, _ = m.Update(queryResultMsg{query: "hex", rows: []queryclient.ResultRow{{Headword: "fresh-word"}}})
	m = mm.(Model)
	if len(m.rows) != 1 || m.rows[0].Headword != "fresh-word" {
		t.Fatalf("expected the fresh result (query matches current input) to be applied, got rows=%v", m.rows)
	}
}

func TestSuccessfulQueryResultClearsAStaleErrorMessage(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("annoyance")

	mm, _ := m.Update(queryErrorMsg{query: "annoyance", err: errors.New("revdict: error: boom")})
	m = mm.(Model)
	if m.statusMessage == "" {
		t.Fatal("expected an error status message to be set")
	}

	mm, _ = m.Update(queryResultMsg{query: "annoyance", rows: []queryclient.ResultRow{{Headword: "annoyance"}}})
	m = mm.(Model)
	if m.statusMessage != "" {
		t.Fatalf("expected the stale error message to be cleared on a subsequent successful query, got %q", m.statusMessage)
	}
}

func TestSuccessfulQueryResultDoesNotClearACopyConfirmationMessage(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.rows = []queryclient.ResultRow{{Headword: "annoyance"}}
	m.selected = 0
	m.input.SetValue("annoyance")

	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = mm.(Model)
	if m.statusMessage != "Copied: annoyance" {
		t.Fatalf("expected a copy confirmation, got %q", m.statusMessage)
	}

	mm, _ = m.Update(queryResultMsg{query: "annoyance", rows: []queryclient.ResultRow{{Headword: "annoyance"}}})
	m = mm.(Model)
	if m.statusMessage != "Copied: annoyance" {
		t.Fatalf("expected the copy confirmation to survive an unrelated successful query result, got %q", m.statusMessage)
	}
}

func TestF1OpensHelpScreen(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF1})
	m = mm.(Model)
	if m.screen != screenHelp {
		t.Fatalf("expected screen=screenHelp, got %v", m.screen)
	}
}

func TestEscClosesHelpScreen(t *testing.T) {
	m := NewModel(nil)
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF1})
	m = mm.(Model)
	mm, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.screen != screenSearch {
		t.Fatalf("expected screen=screenSearch after Esc, got %v", m.screen)
	}
}

func TestCtrlRCyclesSortMode(t *testing.T) {
	m := NewModel(nil)
	m.filters = NewFilterState()
	if m.filters.Sort != "relevance" {
		t.Fatalf("expected initial sort=relevance, got %s", m.filters.Sort)
	}
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlR})
	m = mm.(Model)
	if m.filters.Sort != "alpha" {
		t.Fatalf("expected sort cycled to alpha, got %s", m.filters.Sort)
	}
}

func TestCtrlRWrapsAroundAfterLastSortMode(t *testing.T) {
	m := NewModel(nil)
	m.filters = FilterState{Sort: "most_lyrical", Category: "all"}
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlR})
	m = mm.(Model)
	if m.filters.Sort != "relevance" {
		t.Fatalf("expected wraparound to relevance, got %s", m.filters.Sort)
	}
}

func TestEnterCallsOnCopyWithSelectedHeadword(t *testing.T) {
	m := NewModel([]queryclient.ResultRow{{Headword: "annoyance"}})
	var copied string
	m.onCopy = func(h string) error { copied = h; return nil }
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if copied != "annoyance" {
		t.Fatalf("expected onCopy called with 'annoyance', got %q", copied)
	}
}

func TestEnterSurfacesACopyFailureInStatusMessage(t *testing.T) {
	m := NewModel([]queryclient.ResultRow{{Headword: "annoyance"}})
	m.onCopy = func(h string) error { return errors.New("no clipboard utility found") }
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = mm.(Model)
	if !strings.Contains(m.statusMessage, "no clipboard utility found") {
		t.Fatalf("expected copy failure in status message, got %q", m.statusMessage)
	}
}

func TestRefreshPreviewIncludesSynonymsExamplesAndStress(t *testing.T) {
	stress := "HAP-py"
	rows := []queryclient.ResultRow{
		{
			Headword: "happy", Definition: "feeling joy",
			Stress:   &stress,
			Synonyms: []string{"glad", "cheerful"},
			Examples: []string{"a happy childhood"},
		},
	}
	m := NewModel(rows)
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)
	out := m.preview.View()
	for _, want := range []string{"HAP-py", "glad", "cheerful", "a happy childhood"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected preview to contain %q, got: %s", want, out)
		}
	}
}

func TestHelpScreenClampsToAvailableHeightOnAShortTerminal(t *testing.T) {
	m := NewModel(testRows())
	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyF1})
	m = mm.(Model)
	mm, _ = m.Update(tea.WindowSizeMsg{Width: 80, Height: 10})
	m = mm.(Model)
	out := m.View()
	lines := strings.Split(out, "\n")
	if len(lines) > 10 {
		t.Fatalf("expected help view clamped to 10 lines, got %d", len(lines))
	}
	if !strings.Contains(out, "revdict-tui -- keyboard shortcuts") {
		t.Fatalf("expected the title (top-anchored) to still be visible, got: %s", out)
	}
}

func TestNewerDebounceCancelsThePreviousInFlightQuery(t *testing.T) {
	fake := &fakeExecutor{}
	client := queryclient.NewWithExecutor(fake)
	m := NewLiveModel(client)
	m.input.SetValue("he")

	mm, cmd1 := m.Update(debounceFiredMsg{query: "he"})
	m = mm.(Model)
	cmd1() // invoke so fakeExecutor captures the context; result discarded

	if len(fake.ctxs) != 1 {
		t.Fatalf("expected 1 captured context after the first dispatch, got %d", len(fake.ctxs))
	}
	firstCtx := fake.ctxs[0]

	m.input.SetValue("hex")
	mm, _ = m.Update(debounceFiredMsg{query: "hex"})
	m = mm.(Model)

	if firstCtx.Err() != context.Canceled {
		t.Fatalf("expected the first query's context to be cancelled once superseded by a newer one, got err=%v", firstCtx.Err())
	}
}

// TestRunQueryCmdSuppressesErrorsWhenContextIsCancelled guards against a
// real usage bug: cancelling a superseded query's context (as
// m.cancelInFlight does on every newer debounce) kills the real
// exec.CommandContext subprocess with SIGKILL, which surfaces as a generic
// "signal: killed" error from the executor. That's expected, routine
// cancellation -- not a genuine query failure -- so it must not be surfaced
// to the user as a queryErrorMsg.
func TestRunQueryCmdSuppressesErrorsWhenContextIsCancelled(t *testing.T) {
	fake := &fakeExecutor{err: errors.New("signal: killed")}
	client := queryclient.NewWithExecutor(fake)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	cmd := runQueryCmd(ctx, client, "x", queryclient.Request{Query: "x", TopN: 30, Sort: "relevance", Category: "all"})
	msg := cmd()
	if msg != nil {
		t.Fatalf("expected a cancelled query's error to be suppressed (nil message), got %T: %v", msg, msg)
	}
}

// TestRunQueryCmdSurfacesARealErrorWhenContextIsNotCancelled guards against
// the fix above over-suppressing: a genuine executor error on a live
// (non-cancelled) context must still reach the user as a queryErrorMsg.
func TestRunQueryCmdSurfacesARealErrorWhenContextIsNotCancelled(t *testing.T) {
	fake := &fakeExecutor{err: errors.New("revdict: index not found")}
	client := queryclient.NewWithExecutor(fake)
	ctx := context.Background()

	cmd := runQueryCmd(ctx, client, "x", queryclient.Request{Query: "x", TopN: 30, Sort: "relevance", Category: "all"})
	msg := cmd()
	errMsg, ok := msg.(queryErrorMsg)
	if !ok {
		t.Fatalf("expected a genuine error on a live context to surface as queryErrorMsg, got %T: %v", msg, msg)
	}
	if !strings.Contains(errMsg.err.Error(), "revdict: index not found") {
		t.Fatalf("expected the original error text to be preserved, got %q", errMsg.err.Error())
	}
}

func TestNewModelSetsAStaticAlwaysVisibleCursor(t *testing.T) {
	m := NewModel(testRows())
	if m.input.Cursor.Mode() != cursor.CursorStatic {
		t.Fatalf("expected a static (always-visible) cursor mode, got %v", m.input.Cursor.Mode())
	}
}

// TestSelectedRowStyleIsBoldAndReversed guards the results-list selection
// highlight added alongside the panel's focus/selection highlighting: the
// style must actually carry bold+reverse-video, not just exist as an
// unconfigured zero-value lipgloss.Style (which would render as a no-op and
// leave the selected row visually identical to every other row). Reverse
// video (rather than a hardcoded color) is used so the highlight adapts to
// whatever colors the user's terminal theme has set, instead of a fixed
// 256-color-palette entry that theme switching can't touch.
func TestSelectedRowStyleIsBoldAndReversed(t *testing.T) {
	if !selectedRowStyle.GetBold() {
		t.Fatal("expected selectedRowStyle to be bold")
	}
	if !selectedRowStyle.GetReverse() {
		t.Fatalf("expected selectedRowStyle to use reverse video (theme-adaptive highlight), got Reverse=%v", selectedRowStyle.GetReverse())
	}
}

// TestErrorStatusStyleIsBoldAndReversed guards the error-banner highlight
// the same way TestSelectedRowStyleIsBoldAndReversed guards the results-list
// highlight: the style must actually carry bold+reverse-video, not just
// exist as an unconfigured zero-value lipgloss.Style, otherwise an error
// would render visually identical to ordinary status text (e.g. "Copied:
// X") -- defeating the user's requirement that errors never blend in.
func TestErrorStatusStyleIsBoldAndReversed(t *testing.T) {
	if !errorStatusStyle.GetBold() {
		t.Fatal("expected errorStatusStyle to be bold")
	}
	if !errorStatusStyle.GetReverse() {
		t.Fatalf("expected errorStatusStyle to use reverse video (theme-adaptive highlight), got Reverse=%v", errorStatusStyle.GetReverse())
	}
}

func TestTypingShowsAQueryingIndicatorUntilResultsArrive(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)

	mm, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m = mm.(Model)
	if !m.querying {
		t.Fatal("expected querying=true immediately after typing")
	}
	if !strings.Contains(m.View(), "Searching...") {
		t.Fatalf("expected the view to show a querying indicator, got: %s", m.View())
	}

	mm, _ = m.Update(queryResultMsg{query: "h", rows: []queryclient.ResultRow{{Headword: "happy"}}})
	m = mm.(Model)
	if m.querying {
		t.Fatal("expected querying=false once a matching result arrives")
	}
	if strings.Contains(m.View(), "Searching...") {
		t.Fatalf("expected the querying indicator to disappear once results arrive, got: %s", m.View())
	}
}

func TestQueryErrorAlsoClearsTheQueryingIndicator(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)
	m.input.SetValue("x")
	m.querying = true

	mm, _ := m.Update(queryErrorMsg{query: "x", err: errors.New("boom")})
	m = mm.(Model)
	if m.querying {
		t.Fatal("expected querying=false once an error arrives for the current query")
	}
}

func TestSupersededQueryKeepsTheIndicatorShowingUntilTheNewerOneSettles(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)
	m.input.SetValue("he")
	m.querying = true

	// a stale result for an already-superseded query must not clear it
	mm, _ := m.Update(queryResultMsg{query: "h", rows: []queryclient.ResultRow{{Headword: "stale"}}})
	m = mm.(Model)
	if !m.querying {
		t.Fatal("expected querying to remain true -- this result is for a stale, superseded query")
	}

	// the result for the CURRENT text clears it
	mm, _ = m.Update(queryResultMsg{query: "he", rows: []queryclient.ResultRow{{Headword: "hex"}}})
	m = mm.(Model)
	if m.querying {
		t.Fatal("expected querying=false once the current query's result arrives")
	}
}

func TestEscClearingTheQueryAlsoClearsTheQueryingIndicator(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)
	m.input.SetValue("he")
	m.querying = true

	mm, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = mm.(Model)
	if m.querying {
		t.Fatal("expected querying=false after Esc clears the query -- the pending debounce for the old text will be ignored as stale and will never clear it otherwise")
	}
	if strings.Contains(m.View(), "Searching...") {
		t.Fatalf("expected no querying indicator after Esc clears the query, got: %s", m.View())
	}
}

func TestSpinnerTicksWhileQueryingAndStopsOnceSettled(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)

	mm, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m = mm.(Model)
	if cmd == nil {
		t.Fatal("expected a command to be returned")
	}
	tickMsg := findSpinnerTickMsg(t, cmd())
	mm, cmd = m.Update(tickMsg)
	m = mm.(Model)
	if cmd == nil {
		t.Fatal("expected the spinner to keep ticking while querying")
	}

	// once settled (a matching result arrives), further ticks must not
	// perpetuate the animation loop
	mm, _ = m.Update(queryResultMsg{query: "h", rows: []queryclient.ResultRow{{Headword: "happy"}}})
	m = mm.(Model)
	_, cmd = m.Update(tickMsg)
	if cmd != nil {
		t.Fatal("expected the spinner to stop ticking once querying is false")
	}
}

func TestQueryErrorIsStyledDistinctlyAndAutoDismisses(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)
	m.input.SetValue("x")
	mm, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = mm.(Model)

	mm, cmd := m.Update(queryErrorMsg{query: "x", err: errors.New("revdict: error: boom")})
	m = mm.(Model)
	if !strings.Contains(m.View(), "boom") {
		t.Fatalf("expected the exact error text to appear in the view, got: %s", m.View())
	}
	if cmd == nil {
		t.Fatal("expected a dismiss timer command to be scheduled")
	}
	dismissMsg, ok := cmd().(dismissErrorMsg)
	if !ok {
		t.Fatalf("expected dismissErrorMsg, got %T", cmd())
	}

	mm, _ = m.Update(dismissMsg)
	m = mm.(Model)
	if m.statusMessage != "" || m.statusIsError {
		t.Fatalf("expected the error to auto-dismiss, got statusMessage=%q statusIsError=%v", m.statusMessage, m.statusIsError)
	}
}

func TestANewerErrorsDismissTimerDoesNotWipeAnEvenNewerError(t *testing.T) {
	client := queryclient.NewWithExecutor(&fakeExecutor{})
	m := NewLiveModel(client)
	m.input.SetValue("x")

	mm, cmd1 := m.Update(queryErrorMsg{query: "x", err: errors.New("first error")})
	m = mm.(Model)
	firstDismiss := cmd1().(dismissErrorMsg)

	mm, _ = m.Update(queryErrorMsg{query: "x", err: errors.New("second error")})
	m = mm.(Model)

	// the FIRST error's dismiss timer fires late -- it must not clear the SECOND error
	mm, _ = m.Update(firstDismiss)
	m = mm.(Model)
	if m.statusMessage != "second error" {
		t.Fatalf("expected the second error to survive the first error's stale dismiss timer, got %q", m.statusMessage)
	}
}
