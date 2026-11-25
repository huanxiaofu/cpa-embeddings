package main

import (
	"os"
	"path/filepath"
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

// 原生型号清单只来自 CPA 配置的 api-keys.gemini.models[].name，不再硬编码三个型号。
func geminiFixture(models ...string) string {
	var b strings.Builder
	b.WriteString("api-keys:\n  gemini:\n    - base-url: https://generativelanguage.googleapis.com\n      keys:\n        - api-key: unit-gemini\n          weight: 1\n      models:\n")
	for _, m := range models {
		b.WriteString("        - name: " + m + "\n          alias: " + m + "\n")
	}
	return b.String()
}

// 型号名只能是向量族的单段标识符，不得拼出聊天动作或任意 Google 路径。
func TestGeminiModelName(t *testing.T) {
	for _, m := range []string{"gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview", "gemini-embedding-exp-03-07", "text-embedding-004", "text-embedding-3-small"} {
		if !geminiModelName(m) {
			t.Fatal("valid model rejected", m)
		}
	}
	for _, m := range []string{"", "gemini-embedding-", "text-embedding-", "gemini-1.5-flash", "models/gemini-embedding-001", "gemini-embedding-001/", "/gemini-embedding-001", "gemini-embedding-001:embedContent", "gemini-embedding-001:generateContent", "gemini-embedding-../chat", "gemini-embedding-.hidden", "gemini-embedding--", "text-embedding-a b", "text-embedding-A004"} {
		if geminiModelName(m) {
			t.Fatal("unsafe model accepted", m)
		}
	}
}

func TestValidGeminiModelFollowsConfig(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.yaml")
	declared := geminiFixture("gemini-embedding-001", "text-embedding-004", "gemini-embedding-exp-03-07")
	if e := os.WriteFile(p, []byte(declared), 0600); e != nil {
		t.Fatal(e)
	}
	t.Setenv("CPR_EMBED_CONFIG", p)
	for _, m := range []string{"gemini-embedding-001", "text-embedding-004", "gemini-embedding-exp-03-07"} {
		if !validGeminiModel(m) {
			t.Fatal("declared model rejected", m)
		}
	}
	// 未登记、聊天型号或试图拼出路径/动作的名字都不能路由到原生端点。
	for _, m := range []string{"", "gemini-embedding-2", "gemini-1.5-flash", "text-embedding-005", "models/gemini-embedding-001", "gemini-embedding-001:embedContent", "gemini-embedding-../chat"} {
		if validGeminiModel(m) {
			t.Fatal("unexpected model accepted", m)
		}
	}
	// 原生不做别名转发：alias 与 name 不同的条目不再视为可用。
	alias := strings.Replace(declared, "alias: text-embedding-004", "alias: private-embed", 1)
	if e := os.WriteFile(p, []byte(alias), 0600); e != nil {
		t.Fatal(e)
	}
	if validGeminiModel("text-embedding-004") || validGeminiModel("private-embed") {
		t.Fatal("aliased native model accepted")
	}
	// 非原生 base 的组不参与。
	other := strings.Replace(declared, "https://generativelanguage.googleapis.com", "https://example.invalid", 1)
	if e := os.WriteFile(p, []byte(other), 0600); e != nil {
		t.Fatal(e)
	}
	if validGeminiModel("text-embedding-004") {
		t.Fatal("non-native base accepted")
	}
	// 配置缺失时不能凭空接纳。
	t.Setenv("CPR_EMBED_CONFIG", filepath.Join(t.TempDir(), "absent.yaml"))
	if validGeminiModel("text-embedding-004") {
		t.Fatal("missing configuration accepted")
	}
}

func TestLiveBindingUsesConfiguredModels(t *testing.T) {
	cfg := geminiFixture("text-embedding-004")
	if b, e := liveBinding([]byte(cfg), "text-embedding-004"); e != nil || b.Key == "" {
		t.Fatal("configured model not bound", e)
	}
	for _, m := range []string{"gemini-embedding-2", "gemini-1.5-flash", "text-embedding-004/../chat", "text-embedding-004:embedContent"} {
		if _, e := liveBinding([]byte(cfg), m); e == nil {
			t.Fatal("unexpected binding", m)
		}
	}
}

// 上游入口同样只认配置，并且只放行向量动作（不会向 Google 发起请求）。
func TestUpstreamForModelGate(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.yaml")
	if e := os.WriteFile(p, []byte(geminiFixture("text-embedding-004")), 0600); e != nil {
		t.Fatal(e)
	}
	t.Setenv("CPR_EMBED_CONFIG", p)
	call, e := upstreamFor("text-embedding-004")
	if e != nil || call == nil {
		t.Fatal("declared model not resolvable", e)
	}
	// 只测拒绝分支：聊天等其它动作在构造 URL 前就被挡回，不会产生网络调用。
	for _, action := range []string{"generateContent", "streamGenerateContent", "countTextTokens", ""} {
		if _, s, e := call(action, []byte(`{}`)); s != 400 || e == nil {
			t.Fatal("unexpected action accepted", action, s)
		}
	}
	for _, m := range []string{"gemini-embedding-2", "gemini-1.5-flash", "text-embedding-004:embedContent", "models/text-embedding-004"} {
		if _, e := upstreamFor(m); e == nil {
			t.Fatal("unexpected upstream", m)
		}
	}
}
