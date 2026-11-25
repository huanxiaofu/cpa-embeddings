package main

import (
	"strings"
	"testing"
)

// 名称识别覆盖面：各厂商公开向量型号逐个放行，聊天/重排/图像生成与畸形名称拒绝。
func TestEmbeddingModelName(t *testing.T) {
	allow := []string{
		"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002", "ada-002",
		"text-embedding-004", "text-embedding-v4",
		"gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview",
		"qwen3-embedding-8b", "qwen3-vl-embedding-8b", "Qwen/Qwen3-Embedding-4B",
		"BAAI/bge-m3", "bge-large-zh-v1.5", "baai/bge-base-en-v1.5",
		"mistral-embed",
		"embed-v4.0", "embed-english-v3.0", "embed-multilingual-light-v3.0", "cohere-embed-v3-english",
		"jina-embeddings-v3", "jina-embeddings-v4", "jina-clip-v2",
		"voyage-3", "voyage-3.5-lite", "voyage-code-2",
		"nomic-embed-text-v1.5", "nomic-embed-vision-v1.5",
		"gte-large", "e5-large-v2", "multilingual-e5-large", "snowflake-arctic-embed-l",
		"mxbai-embed-large", "tongyi-embedding-vision-plus",
	}
	for _, name := range allow {
		if !embeddingModelName(name) {
			t.Errorf("向量型号被拒绝: %s", name)
		}
	}
	deny := []string{
		"", "gpt-4o", "gpt-4o-mini", "qwen-max", "qwen-vl-plus", "gemini-1.5-pro",
		"gemini-2.0-flash", "claude-3-5-sonnet", "mistral-large-latest", "deepseek-chat",
		// 重排模型即使带向量家族前缀也拒绝。
		"bge-reranker-v2-m3", "BAAI/bge-reranker-base", "jina-reranker-v2-base-multilingual",
		"qwen3-reranker-0.6b", "cohere-rerank-v3.5",
		// 图像/音视频生成。
		"dall-e-3", "gpt-image-1", "imagen-3.0-generate-002", "stable-diffusion-xl",
		"flux-pro", "sdxl-turbo", "veo-2", "sora", "whisper-1", "tts-1", "gpt-4o-transcribe",
		// 名称特征优先于家族前缀，且畸形写法不识别。
		"text-embedding-generate", "qwen3-embedding-chat", "text-embedding-004/../chat",
		"text embedding", "https://example.test/embed", "/bge-m3", "bge-m3/",
	}
	for _, name := range deny {
		if embeddingModelName(name) {
			t.Errorf("非向量型号被放行: %s", name)
		}
	}
}

// 显式输出模态优先：只有 embedding/embeddings 才放行，显式非向量或无法识别的
// 模态先拒绝；没有显式声明时返回 ok=false，交回名称识别。
func TestDeclaredModalities(t *testing.T) {
	for _, tc := range []struct {
		label          string
		in             []string
		allow, decided bool
	}{
		{"未声明", nil, false, false},
		{"空条目", []string{"", "  "}, false, false},
		{"embedding", []string{"embedding"}, true, true},
		{"embeddings 复数", []string{"embeddings"}, true, true},
		{"大小写与空白", []string{"  EmbEdding "}, true, true},
		{"chat", []string{"chat"}, false, true},
		{"text 属于对话输出", []string{"text"}, false, true},
		{"rerank", []string{"rerank"}, false, true},
		{"image", []string{"image"}, false, true},
		{"非向量优先", []string{"chat", "embedding"}, false, true},
		{"未知模态", []string{"hologram"}, false, true},
	} {
		allow, decided := declaredEmbedding(tc.in)
		if allow != tc.allow || decided != tc.decided {
			t.Errorf("%s: allow=%v decided=%v", tc.label, allow, decided)
		}
	}
}

// 自动裁决只看真实名称与显式模态；alias 不能反过来证明模型是向量模型。
func TestEmbeddingModelAllowed(t *testing.T) {
	if !embeddingModelAllowed(modelBinding{Name: "vendor-vector-v1", Modality: []string{"embedding"}}) {
		t.Error("显式 embedding 模态未放行")
	}
	if embeddingModelAllowed(modelBinding{Name: "text-embedding-3-small", Modality: []string{"chat"}}) {
		t.Error("显式 chat 模态未拒绝")
	}
	if embeddingModelAllowed(modelBinding{Name: "gpt-4o", Alias: "text-embedding-3-small"}) {
		t.Error("alias 伪造向量名被放行")
	}
	if !embeddingModelAllowed(modelBinding{Name: "BAAI/bge-m3"}) {
		t.Error("真实向量名未放行")
	}
}

const recFixture = `
plugins:
  configs:
    cpr-gemini-embeddings:
      upstream-protocols:
        明确指定: cohere
        写错协议: grpc
      model-overrides:
        混合渠道:
          人工放行: allow
          text-embedding-3-large: deny
          写错规则: maybe
api-keys:
  openai-compatibility:
    - name: 混合渠道
      base-url: https://embeddings.internal.test/v1
      keys: [{api-key: unit-mixed}]
      models:
        - {name: gpt-4o}
        - {name: text-embedding-3-small, alias: kb-embed}
        - {name: text-embedding-3-large}
        - {name: 人工放行}
        - {name: 写错规则}
        - {name: vendor-vector-v1, output-modalities: [embedding]}
        - {name: vendor-chat-v1, output-modalities: [chat]}
        - {name: text-embedding-v4, output-modalities: [embeddings]}
    - name: 明确指定
      base-url: https://api.cohere.com/v2
      keys: [{api-key: unit-cohere}]
      models: [{name: embed-v4.0, alias: co-doc}]
    - name: 写错协议
      base-url: https://api.cohere.com/v2
      keys: [{api-key: unit-bad}]
      models: [{name: embed-v4.0}]
    - name: official-cohere-embeddings
      base-url: https://api.cohere.com/v2
      keys: [{api-key: unit-official}]
      models: [{name: embed-english-v3.0, alias: official-doc}]
  gemini:
    - name: 原生渠道
      base-url: https://generativelanguage.googleapis.com
      keys: [{api-key: unit-gemini}]
      models: [{name: gemini-embedding-001}]
`

func recFind(t *testing.T, cfg *catalogConfig, group, model string) (groupBinding, modelBinding) {
	t.Helper()
	for _, g := range append(append([]groupBinding{}, cfg.API.Compat...), cfg.API.Gemini...) {
		for _, m := range g.Models {
			if g.Name == group && m.Name == model {
				return g, m
			}
		}
	}
	t.Fatalf("缺少组或模型: %s/%s", group, model)
	return groupBinding{}, modelBinding{}
}

// modelDecision 是主会话唯一入口：返回是否自动裁决、最终协议与是否放行。
func TestModelDecision(t *testing.T) {
	cfg, err := parseCatalog([]byte(recFixture))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		label, group, model string
		auto, enabled       bool
		mode                string
	}{
		{"默认协议按名称放行", "混合渠道", "text-embedding-3-small", true, true, "openai"},
		{"聊天模型拒绝", "混合渠道", "gpt-4o", true, false, "openai"},
		{"显式 embedding 模态放行", "混合渠道", "vendor-vector-v1", true, true, "openai"},
		{"显式 chat 模态拒绝", "混合渠道", "vendor-chat-v1", true, false, "openai"},
		{"复数 embeddings 放行", "混合渠道", "text-embedding-v4", true, true, "openai"},
		{"人工 allow 覆盖名称识别", "混合渠道", "人工放行", false, true, "openai"},
		{"人工 deny 覆盖名称识别", "混合渠道", "text-embedding-3-large", false, false, "openai"},
		{"非法规则取值按拒绝", "混合渠道", "写错规则", false, false, "openai"},
		{"显式 cohere 映射", "明确指定", "embed-v4.0", true, true, "cohere"},
		{"Cohere 官方命名", "official-cohere-embeddings", "embed-english-v3.0", true, true, "cohere"},
		{"非法协议映射", "写错协议", "embed-v4.0", true, false, ""},
		{"原生 Gemini base", "原生渠道", "gemini-embedding-001", true, true, "gemini"},
	} {
		g, m := recFind(t, cfg, tc.group, tc.model)
		auto, mode, enabled := cfg.modelDecision(g, m)
		if auto != tc.auto || mode != tc.mode || enabled != tc.enabled {
			t.Errorf("%s: auto=%v mode=%q enabled=%v", tc.label, auto, mode, enabled)
		}
	}
	// 人工规则按真实模型名索引：用 alias 写规则不会命中。
	g, m := recFind(t, cfg, "混合渠道", "text-embedding-3-small")
	if auto, _, enabled := cfg.modelDecision(g, modelBinding{Name: m.Alias, Alias: m.Alias}); !auto || enabled {
		t.Errorf("alias 被当作真实模型裁决: auto=%v enabled=%v", auto, enabled)
	}
}

// parseCatalog 只解码一次；compatBindingFromConfig 与读文件的 compatBindingFrom 语义一致。
func TestParseCatalogAndBindingFromConfig(t *testing.T) {
	cfg, err := parseCatalog([]byte(recFixture))
	if err != nil || cfg == nil {
		t.Fatal(err)
	}
	if cfg.protocol("明确指定") != "cohere" || cfg.protocol("混合渠道") != "openai" || cfg.protocol("写错协议") != "" {
		t.Fatal("协议映射未解析")
	}
	if len(cfg.API.Compat) != 4 || len(cfg.API.Gemini) != 1 {
		t.Fatalf("组数量错误: %d %d", len(cfg.API.Compat), len(cfg.API.Gemini))
	}
	// Gemini 组同样带 modelBinding 与显式模态字段。
	if cfg.API.Gemini[0].Models[0].Modality != nil {
		t.Fatal("未声明模态被赋默认值")
	}
	for _, tc := range []struct {
		model, base, key string
		cohere           bool
	}{
		{"kb-embed", "https://embeddings.internal.test/v1", "unit-mixed", false},
		{"text-embedding-3-large", "", "", false},
		{"人工放行", "https://embeddings.internal.test/v1", "unit-mixed", false},
		{"gpt-4o", "", "", false},
		{"co-doc", "https://api.cohere.com/v2", "unit-cohere", true},
		{"official-doc", "https://api.cohere.com/v2", "unit-official", true},
		{"embed-v4.0", "", "", false},
	} {
		target, err := compatBindingFromConfig(cfg, tc.model)
		if tc.base == "" {
			if err == nil {
				t.Errorf("%s: 被错误绑定 %+v", tc.model, target)
			}
			continue
		}
		if err != nil || target.Base != tc.base || target.Key != tc.key || target.Cohere != tc.cohere {
			t.Errorf("%s: %v %+v", tc.model, err, target)
		}
		fromRaw, err := compatBindingFrom([]byte(recFixture), tc.model)
		if err != nil || fromRaw != target {
			t.Errorf("%s: 两条入口不一致 %v %+v", tc.model, err, fromRaw)
		}
	}
	// nil 配置不能凭空放行。
	if _, err := compatBindingFromConfig(nil, "kb-embed"); err == nil {
		t.Error("nil 配置被接受")
	}
}

// 人工 allow 只放宽模型识别，不越过凭据与地址等安全约束。
func TestOverrideDoesNotBypassSafety(t *testing.T) {
	const override = `
plugins:
  configs:
    cpr-gemini-embeddings:
      model-overrides:
        私有渠道:
          vendor-vector-v1: allow
`
	for _, tc := range []struct{ label, extra, keys string }{
		{"组代理策略", "      proxy-url: http://127.0.0.1:9\n", "[{api-key: unit-a}]"},
		{"组附加头", "      headers: {X-Test: '1'}\n", "[{api-key: unit-a}]"},
		{"多个可用凭据", "", "[{api-key: unit-a}, {api-key: unit-b}]"},
		{"无凭据", "", "[]"},
		{"http 地址", "", "[{api-key: unit-a}]"},
	} {
		base := "https://api.example.test/v1"
		if tc.label == "http 地址" {
			base = "http://api.example.test/v1"
		}
		raw := override + compatCfg(compatGroupYAML("私有渠道", base, tc.extra,
			"[{name: vendor-vector-v1}]", tc.keys))
		cfg, err := parseCatalog([]byte(raw))
		if err != nil {
			t.Fatal(tc.label, err)
		}
		// 决策层放行，绑定层仍必须拒绝。
		if _, _, enabled := cfg.modelDecision(cfg.API.Compat[0], cfg.API.Compat[0].Models[0]); !enabled {
			t.Errorf("%s: 人工规则未生效", tc.label)
		}
		if target, err := compatBindingFromConfig(cfg, "vendor-vector-v1"); err == nil {
			t.Errorf("%s: 越过安全约束 %+v", tc.label, target)
		}
	}
}

// 多文档配置与空内容无法判定以哪份为准，一律拒绝。
func TestParseCatalogRejectsMultiDocument(t *testing.T) {
	if _, err := parseCatalog(nil); err == nil {
		t.Error("空配置被接受")
	}
	if _, err := parseCatalog([]byte(recFixture + "---\napi-keys: {}\n")); err == nil {
		t.Error("多文档配置被接受")
	}
	// 全局代理在绑定层拒绝，不退回直连。
	cfg, err := parseCatalog([]byte("proxy-url: http://127.0.0.1:9\n" + recFixture))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compatBindingFromConfig(cfg, "kb-embed"); err == nil {
		t.Error("全局代理策略被忽略")
	}
}

// 混合组按模型逐个识别：组内聊天模型不影响向量模型的绑定。
func TestMixedGroupBinding(t *testing.T) {
	raw := compatCfg(compatGroupYAML("mixed", "https://api.example.test/v1", "",
		"[{name: gpt-4o}, {name: text-embedding-3-small, alias: kb-embed}, {name: bge-reranker-v2-m3}]",
		"[{api-key: unit-mixed}]"))
	cfg, err := parseCatalog([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	target, err := compatBindingFromConfig(cfg, "kb-embed")
	if err != nil || target.Name != "text-embedding-3-small" {
		t.Fatalf("混合组向量模型未绑定: %v %+v", err, target)
	}
	for _, m := range []string{"gpt-4o", "bge-reranker-v2-m3"} {
		if _, err := compatBindingFromConfig(cfg, m); err == nil {
			t.Errorf("%s: 混合组内非向量模型被放行", m)
		}
	}
	// 显式非向量模态即使名称像向量也拒绝。
	raw = compatCfg(compatGroupYAML("mixed", "https://api.example.test/v1", "",
		"[{name: text-embedding-3-small, output-modalities: [chat, embedding]}]",
		"[{api-key: unit-mixed}]"))
	cfg, err = parseCatalog([]byte(raw))
	if err != nil || cfg.API.Compat[0].Models[0].Modality == nil {
		t.Fatalf("模态未解析: %v", err)
	}
	if target, err := compatBindingFromConfig(cfg, "text-embedding-3-small"); err == nil {
		t.Errorf("显式非向量模态被放行: %+v", target)
	}
	// 单值模态写法也要能解析。
	if !strings.Contains(recFixture, "output-modalities: [embedding]") {
		t.Fatal("fixture 漂移")
	}
	raw = compatCfg(compatGroupYAML("single", "https://api.example.test/v1", "",
		"[{name: vendor-vector-v1, output-modalities: embedding}]", "[{api-key: unit-a}]"))
	if cfg, err := parseCatalog([]byte(raw)); err != nil || len(cfg.API.Compat[0].Models[0].Modality) != 1 {
		t.Fatalf("单值模态未解析: %v", err)
	}
}
