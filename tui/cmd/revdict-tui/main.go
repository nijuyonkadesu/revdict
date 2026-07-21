package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	if _, err := exec.LookPath("revdict"); err != nil {
		fmt.Fprintln(os.Stderr, "revdict-tui: 'revdict' not found on PATH -- install it first, see the repo README")
		os.Exit(1)
	}
	fmt.Println("revdict-tui: skeleton OK, revdict found on PATH")
}
