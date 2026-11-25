package main

import (
	"strings"
	"testing"
)

func TestLiveBindingLimitsAndRevocation(t *testing.T) {
	cfg := `api-keys:
  gemini:
    - base-url: https://generativelanguage.googleapis.com
      keys:
        - api-key: placeholder
          weight: 1
      models:
        - name: gemini-embedding-2
          alias: gemini-embedding-2
`
	if b, e := liveBinding([]byte(cfg), "gemini-embedding-2"); e != nil || b.Key != "placeholder" {
		t.Fatal("expected one configured key")
	}
	for _, v := range []string{strings.ReplaceAll(cfg, "weight: 1", "weight: 0"), strings.ReplaceAll(cfg, "weight: 1", "disabled: true"), strings.ReplaceAll(cfg, "base-url:", "disabled: true\n      base-url:"), strings.ReplaceAll(cfg, "googleapis.com", "invalid"), strings.ReplaceAll(cfg, "alias: gemini-embedding-2", "alias: private-other"), strings.ReplaceAll(cfg, "models:", "excluded-models: ['gemini-*']\n      models:")} {
		if _, e := liveBinding([]byte(v), "gemini-embedding-2"); e == nil {
			t.Fatal("inactive or altered binding accepted")
		}
	}
}
