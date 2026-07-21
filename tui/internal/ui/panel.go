package ui

import (
	"strconv"
	"strings"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"
)

var sortModes = []string{
	"relevance", "alpha", "alpha_desc", "shortest", "longest",
	"most_common", "least_common", "most_formal", "oldest", "most_modern", "most_lyrical",
}

var categories = []string{"all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old"}

var arpabetVowels = []string{
	"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
	"IH", "IY", "OW", "OY", "UH", "UW",
}

const (
	fieldSort = iota
	fieldCategory
	fieldSyllables
	fieldPrimaryVowel
	fieldRhymesWith
	fieldSoundsLike
	fieldMeter
	fieldCount
)

type panelState struct {
	focusedField     int
	sortSelected     int
	categorySelected int
	syllablesText    string
	primaryVowelText string
	rhymesWithText   string
	soundsLikeText   string
	meterText        string
}

func newPanelState(initial FilterState) panelState {
	p := panelState{}
	for i, s := range sortModes {
		if s == initial.Sort {
			p.sortSelected = i
		}
	}
	for i, c := range categories {
		if c == initial.Category {
			p.categorySelected = i
		}
	}
	if initial.Syllables != nil {
		p.syllablesText = strconv.Itoa(*initial.Syllables)
	}
	p.primaryVowelText = initial.PrimaryVowel
	p.rhymesWithText = initial.RhymesWith
	p.soundsLikeText = initial.SoundsLike
	p.meterText = initial.Meter
	return p
}

func acceptRuneForField(field int, r rune) bool {
	switch field {
	case fieldSyllables:
		return unicode.IsDigit(r)
	case fieldPrimaryVowel:
		return unicode.IsLetter(r)
	case fieldRhymesWith, fieldSoundsLike:
		return unicode.IsLetter(r) || r == '-' || r == '\''
	case fieldMeter:
		return r == '/' || r == 'x'
	}
	return false
}

func (p panelState) handleKey(msg tea.KeyMsg) panelState {
	switch msg.Type {
	case tea.KeyTab:
		p.focusedField = (p.focusedField + 1) % fieldCount
		return p
	case tea.KeyShiftTab:
		p.focusedField = (p.focusedField - 1 + fieldCount) % fieldCount
		return p
	case tea.KeyUp:
		if p.focusedField == fieldSort && p.sortSelected > 0 {
			p.sortSelected--
		} else if p.focusedField == fieldCategory && p.categorySelected > 0 {
			p.categorySelected--
		}
		return p
	case tea.KeyDown:
		if p.focusedField == fieldSort && p.sortSelected < len(sortModes)-1 {
			p.sortSelected++
		} else if p.focusedField == fieldCategory && p.categorySelected < len(categories)-1 {
			p.categorySelected++
		}
		return p
	case tea.KeyBackspace:
		switch p.focusedField {
		case fieldSyllables:
			p.syllablesText = trimLastRune(p.syllablesText)
		case fieldPrimaryVowel:
			p.primaryVowelText = trimLastRune(p.primaryVowelText)
		case fieldRhymesWith:
			p.rhymesWithText = trimLastRune(p.rhymesWithText)
		case fieldSoundsLike:
			p.soundsLikeText = trimLastRune(p.soundsLikeText)
		case fieldMeter:
			p.meterText = trimLastRune(p.meterText)
		}
		return p
	case tea.KeyRunes:
		for _, r := range msg.Runes {
			if !acceptRuneForField(p.focusedField, r) {
				continue
			}
			switch p.focusedField {
			case fieldSyllables:
				p.syllablesText += string(r)
			case fieldPrimaryVowel:
				p.primaryVowelText += string(unicode.ToUpper(r))
			case fieldRhymesWith:
				p.rhymesWithText += string(r)
			case fieldSoundsLike:
				p.soundsLikeText += string(r)
			case fieldMeter:
				p.meterText += string(r)
			}
		}
		return p
	}
	return p
}

func trimLastRune(s string) string {
	runes := []rune(s)
	if len(runes) == 0 {
		return s
	}
	return string(runes[:len(runes)-1])
}

func (p panelState) toFilterState() FilterState {
	fs := FilterState{
		Sort: sortModes[p.sortSelected], Category: categories[p.categorySelected],
		PrimaryVowel: p.primaryVowelText, RhymesWith: p.rhymesWithText,
		SoundsLike: p.soundsLikeText, Meter: p.meterText,
	}
	if p.syllablesText != "" {
		if n, err := strconv.Atoi(p.syllablesText); err == nil {
			fs.Syllables = &n
		}
	}
	return fs
}

func (p panelState) View() string {
	var b strings.Builder
	b.WriteString("Sort:     " + radioLine(sortModes, p.sortSelected) + "\n")
	b.WriteString("Category: " + radioLine(categories, p.categorySelected) + "\n")
	b.WriteString("Syllables: [" + p.syllablesText + "]  Primary vowel: [" + p.primaryVowelText + "]\n")
	b.WriteString("Rhymes with: [" + p.rhymesWithText + "]  Sounds like: [" + p.soundsLikeText + "]\n")
	b.WriteString("Meter: [" + p.meterText + "]\n")
	return b.String()
}

func radioLine(options []string, selected int) string {
	var b strings.Builder
	for i, opt := range options {
		marker := "( )"
		if i == selected {
			marker = "(*)"
		}
		b.WriteString(marker + " " + opt + "  ")
	}
	return b.String()
}
