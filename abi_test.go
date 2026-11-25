package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestABIRoute(t *testing.T) {
	for _, tc := range []struct {
		model, format string
		values        []string
		stream, want  bool
	}{
		{"gemini-embedding-001", abiFormat, []string{"embeddings"}, false, true},
		{"gemini-embedding-2", abiFormat, []string{"embeddings"}, false, true},
		{"gemini-embedding-2-preview", abiFormat, []string{"embeddings"}, false, true},
		{"gemini-embedding-001", abiFormat, nil, false, false},
		{"gemini-embedding-001", abiFormat, []string{"chat"}, false, false},
		{"gemini-embedding-001", abiFormat, []string{"embeddings", "chat"}, false, false},
		{"gemini-embedding-001", abiFormat, []string{"embeddings"}, true, false},
		{"gemini-embedding-001", "openai", []string{"embeddings"}, false, false},
		{"models/gemini-embedding-001", abiFormat, []string{"embeddings"}, false, true},
		{"gemini-embedding-2-other", abiFormat, []string{"embeddings"}, false, true},
	} {
		raw, _ := json.Marshal(map[string]any{"RequestedModel": tc.model, "SourceFormat": tc.format, "Stream": tc.stream, "Headers": map[string][]string{"x-cpr-operation": tc.values}})
		env := abiDispatch("model.route", raw, nil)
		r, ok := env.Result.(abiRoute)
		if !ok || r.Handled != tc.want || (r.TargetKind == "self") != tc.want {
			t.Errorf("%+v: %+v", tc, env)
		}
	}
}

func TestABIExecutorWire(t *testing.T) {
	raw := []byte(`{"Model":"gemini-embedding-001","SourceFormat":"openai-response","Headers":{"X-Cpr-Operation":["embeddings"]},"Payload":"bmV3","OriginalRequest":"b2xk"}`)
	env := abiDispatch("executor.execute", raw, func(model string, body []byte) ([]byte, int, error) {
		if model != "gemini-embedding-001" || string(body) != "new" {
			t.Fatal("wrong executor input")
		}
		return []byte(`{"usage":{"input_tokens":4}}`), 200, nil
	})
	got, _ := json.Marshal(env)
	var decoded struct {
		OK     bool
		Result struct{ Payload []byte }
	}
	if json.Unmarshal(got, &decoded) != nil || !decoded.OK || string(decoded.Result.Payload) != `{"usage":{"input_tokens":4}}` {
		t.Fatal(string(got))
	}
	env = abiDispatch("executor.execute", raw, func(string, []byte) ([]byte, int, error) { return []byte("sensitive"), 429, errors.New("sensitive") })
	got, _ = json.Marshal(env)
	if env.OK || env.Error.Status != 429 || bytes.Contains(got, []byte("sensitive")) {
		t.Fatal(string(got))
	}
}

func TestABIRegistration(t *testing.T) {
	for _, method := range []string{"plugin.register", "plugin.reconfigure"} {
		cfg, _ := json.Marshal(map[string][]byte{"config_yaml": []byte("enabled: true\npriority: 10\n")})
		env := abiDispatch(method, cfg, nil)
		got, _ := json.Marshal(env.Result)
		var reg struct {
			Capabilities map[string]any
			Metadata     struct{ Name string }
		}
		want := map[string]any{"model_router": true, "executor": true, "executor_model_scope": "static", "executor_input_formats": []any{"openai-response"}, "executor_output_formats": []any{"openai-response"}}
		if !env.OK || json.Unmarshal(got, &reg) != nil || reg.Metadata.Name != abiPluginID || !reflect.DeepEqual(reg.Capabilities, want) {
			t.Fatal(method, string(got))
		}
		if abiDispatch(method, []byte(`{"config_yaml":"enabled: true"}`), nil).OK {
			t.Fatal("accepted non-base64 config")
		}
	}
	if abiDispatch("model.router", []byte(`{}`), nil).OK {
		t.Fatal("model.router is not an RPC method")
	}
	if got := abiDispatch("executor.identifier", nil, nil).Result; !reflect.DeepEqual(got, map[string]string{"identifier": abiPluginID}) {
		t.Fatal(got)
	}
}

func TestABIReconfigure(t *testing.T) {
	defer abiDisabled.Store(false)
	r := abiIngress{SourceFormat: abiFormat, Headers: map[string][]string{"X-Cpr-Operation": {"embeddings"}}}
	for _, step := range []struct {
		yaml      string
		ok, route bool
	}{{"enabled: false", true, false}, {"enabled: [", false, false}, {"enabled: true", true, true}} {
		raw, _ := json.Marshal(map[string][]byte{"config_yaml": []byte(step.yaml)})
		if abiDispatch("plugin.reconfigure", raw, nil).OK != step.ok || r.accepts("gemini-embedding-2") != step.route {
			t.Fatal(step)
		}
	}
}
