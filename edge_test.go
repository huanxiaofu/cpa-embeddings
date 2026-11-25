package main

import (
	"strings"
	"testing"
)

func TestEdgeCatalog(t *testing.T) {
	const creds = "[{api-key: synthetic-fixture}]"
	cfg := compatCfg(compatGroupYAML("embedder-embeddings", "https://provider.example/v1", "", "[{name: vendor-name, alias: my-vector}]", creds), compatGroupYAML("cohere-embeddings", "https://api.cohere.com/v2", "", "[{name: embed-v4.0, alias: co-doc}]", creds), compatGroupYAML("ordinary-chat", "https://chat.example/v1", "", "[{name: chat-only}]", creds))
	edge, e := renderEdge([]byte(cfg), "cpa:8317")
	if e != nil || !strings.Contains(edge, `["my-vector"]=true`) || !strings.Contains(edge, `["co-doc"]=true`) || !strings.Contains(edge, "http://cpa:8317/v1/responses") {
		t.Fatal("catalog render")
	}
	for _, n := range []string{"vendor-name", "synthetic-fixture", "chat-only", "provider.example", "CPA_EMBED_MODELS"} {
		if strings.Contains(edge, n) {
			t.Fatal("unexpected entry", n)
		}
	}
	if _, e := renderEdge([]byte(cfg), "cpa:8317;other"); e == nil {
		t.Fatal("invalid host accepted")
	}
	target, e := compatBindingFrom([]byte(cfg), "co-doc")
	if e != nil || !target.Cohere {
		t.Fatal("cohere protocol not selected")
	}
	compatConfig(t, cfg)
	if _, s, _ := dispatchEmbedding("chat-only", []byte(`{"input":"hello"}`)); s != 400 {
		t.Fatal("chat model not rejected")
	}
	if _, e := compatBindingFrom([]byte("requests:\n  proxy-url: http://gateway.example\n"+cfg), "my-vector"); e == nil {
		t.Fatal("global proxy silently bypassed")
	}
}

func TestExistingGroupWithoutRename(t *testing.T) {
	s := `plugins:
  configs:
    cpr-gemini-embeddings:
      upstream-protocols: {已有渠道: openai}
` + strings.ReplaceAll(compatFixture, "siliconflow-embeddings", "已有渠道")
	if _, e := compatBindingFrom([]byte(s), compatTestModel); e != nil {
		t.Fatal(e)
	}
	out, e := renderEdge([]byte(s), "localhost:8317")
	if e != nil || !strings.Contains(out, compatTestModel) {
		t.Fatal("missing opt-in")
	}
}
