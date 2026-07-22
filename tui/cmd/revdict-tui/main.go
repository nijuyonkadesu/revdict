package main

import (
	"fmt"
	"os"
	"os/exec"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nijuyonkadesu/revdict/tui/internal/clipboard"
	"github.com/nijuyonkadesu/revdict/tui/internal/queryclient"
	"github.com/nijuyonkadesu/revdict/tui/internal/ui"
)

func main() {
	if _, err := exec.LookPath("revdict"); err != nil {
		fmt.Fprintln(os.Stderr, "revdict-tui: 'revdict' not found on PATH -- install it first, see the repo README")
		os.Exit(1)
	}

	model := ui.NewLiveModel(queryclient.New())
	model.SetCopyFunc(clipboard.Copy)

	p := tea.NewProgram(model, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "revdict-tui:", err)
		os.Exit(1)
	}
}
