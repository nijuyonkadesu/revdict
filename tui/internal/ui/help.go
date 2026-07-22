package ui

const helpText = `revdict-tui -- keyboard shortcuts

Search screen:
  (typing)      type into the query box (live search, debounced)
  Up / Down     move the highlighted result
  Enter         copy the highlighted candidate to the clipboard
  Esc           clear the query; press again on an empty query to quit
  Ctrl-C        quit immediately
  Tab           open the filter/sort/category panel
  Ctrl-R        quick-cycle the sort mode
  F2            toggle the preview pane
  F1            this help screen

Filter panel:
  Tab / Shift-Tab   move between fields
  Up / Down         move within the sort/category lists
  (typing)          fill the focused text field
  Esc               close the panel and re-run the query

Query syntax (typed directly into the search box):
  blue*         starts with "blue"
  *bird         ends with "bird"
  bl????rd      starts with "bl", ends with "rd", 4 letters between
  ?????         any 5-letter word
  //fuljyo      anagram/unscramble
  -abcd         excludes these letters
  +abcd         built only from these letters
  bl*:snow      starts with "bl" AND related in meaning to "snow"
  **winter**    multi-word phrases containing the whole word "winter"
  expand:nasa   phrases whose initials spell "nasa"
`
