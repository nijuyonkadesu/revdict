package clipboard

import (
	"encoding/base64"
	"fmt"
	"os"
)

// Copy writes text to the terminal's clipboard via the OSC 52 escape
// sequence -- this works over SSH/tmux (reaching the local machine's
// clipboard, not the remote host's), the same mechanism this project's
// existing fzf-based picker already uses (see picker.py's clipboard
// handling and the README's "Clipboard copy on Enter" section).
func Copy(text string) error {
	encoded := base64.StdEncoding.EncodeToString([]byte(text))
	_, err := fmt.Fprintf(os.Stderr, "\x1b]52;c;%s\x07", encoded)
	return err
}
