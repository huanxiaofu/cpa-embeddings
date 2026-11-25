package main

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const compatTestModel = "BAAI/bge-m3"
const compatFixture = `api-keys:
  openai-compatibility:
    - name: siliconflow
      base-url: https://api.siliconflow.cn/v1
      models: [{name: BAAI/bge-m3, alias: BAAI/bge-m3}]
      keys: [{api-key: synthetic, weight: 1}]
`

func compatConfig(t *testing.T, text string) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "fixture.yaml")
	if e := os.WriteFile(p, []byte(text), 0600); e != nil {
		t.Fatal(e)
	}
	t.Setenv("CPR_EMBED_CONFIG", p)
}
func TestCompatBinding(t *testing.T) {
	compatConfig(t, compatFixture)
	if !compatAllowed(compatTestModel) {
		t.Fatal("missing synthetic binding")
	}
	for _, pair := range [][2]string{{"weight: 1", "weight: 0"}, {"weight: 1", "disabled: true"}, {"weight: 1", "weight: 1, prefix: private"}, {"weight: 1", "weight: 1, headers: {X: Y}"}, {"weight: 1", "weight: 1, proxy-url: broken"}, {"weight: 1", "weight: 1, excluded-models: ['*']"}, {", weight: 1", ""}, {"alias: BAAI/bge-m3", "alias: ''"}, {"siliconflow.cn", "example.test"}, {"weight: 1}", "weight: 1}, {api-key: another, weight: 1}"}} {
		compatConfig(t, strings.Replace(compatFixture, pair[0], pair[1], 1))
		if compatAllowed(compatTestModel) {
			t.Errorf("accepted %q", pair[1])
		}
	}
}

type compatTransport func(*http.Request) (*http.Response, error)

func (f compatTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestCompatExchange(t *testing.T) {
	compatConfig(t, compatFixture)
	const good = `{"data":[{"index":0,"embedding":[1,-2]}],"usage":{"prompt_tokens":3,"total_tokens":4}}`
	reply := good
	status := 200
	calls := 0
	tr := compatTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		var q map[string]any
		if r.URL.String() != compatBase+"/embeddings" || r.Method != "POST" || r.Header.Get("Authorization") != "Bearer synthetic" || json.NewDecoder(r.Body).Decode(&q) != nil || q["encoding_format"] != "float" || q["model"] != compatTestModel || len(q) != 3 {
			t.Fatal("unexpected wire request")
		}
		return &http.Response{StatusCode: status, Header: http.Header{"Location": {"https://example.test"}}, Body: io.NopCloser(strings.NewReader(reply))}, nil
	})
	run := func(body string) (string, int, error) {
		b, s, e := compatRun(compatTestModel, []byte(body), tr)
		return string(b), s, e
	}
	out, s, e := run(`{"input":["text"],"encoding_format":"base64","url":"ignored"}`)
	if e != nil || s != 200 || !strings.Contains(out, `"AACAPwAAAMA="`) || !strings.Contains(out, `"total_tokens":4`) || !strings.Contains(out, compatTestModel) {
		t.Fatal(out, s, e)
	}
	for _, pair := range [][2]string{{"1,-2", "null"}, {"1,-2", "1e100"}, {"1,-2", ""}, {"\"index\":0", "\"index\":1"}, {"\"index\":0,", ""}, {"prompt_tokens", "unused"}, {"total_tokens\":4", "total_tokens\":0"}} {
		reply = strings.Replace(good, pair[0], pair[1], 1)
		if _, s, e := run(`{"input":"text"}`); s != 502 || e == nil {
			t.Error("invalid reply accepted", reply)
		}
	}
	reply = good
	if _, s, _ := run(`{"input":["a","b"]}`); s != 502 {
		t.Error("short batch accepted")
	}
	if _, s, _ := run(`{"input":"a"}`); s != 200 {
		t.Error("float response failed")
	}
	status = 302
	reply = "sensitive"
	before := calls
	if out, s, e := run(`{"input":"x"}`); calls != before+1 || s != 502 || e == nil || strings.Contains(e.Error(), reply) || out != "" {
		t.Fatal("redirect not scrubbed")
	}
	before = calls
	for _, b := range []string{`{"input":null}`, `{"input":[null]}`, `{"input":"a","stream":true}`, `{"input":"a","encoding_format":null}`, `{"input":[` + strings.Repeat(`"a",`, 32) + `"a"]}`, `{"input":"` + strings.Repeat("x", embMaxInput+1) + `"}`} {
		if _, s, _ := run(b); s != 400 && s != 413 {
			t.Error("invalid input accepted")
		}
	}
	if before != calls {
		t.Fatal("invalid input sent upstream")
	}
}

func TestCompatBatchDimensionsAndLimits(t *testing.T) {
	compatConfig(t, compatFixture)
	const batch = `{"data":[{"index":0,"embedding":[0.25]},{"index":1,"embedding":[-1]}],"usage":{"prompt_tokens":2,"total_tokens":2}}`
	for _, tc := range []struct {
		reply  string
		status int
	}{
		{batch, 200}, {strings.Replace(batch, `"index":1`, `"index":0`, 1), 502}, {strings.Replace(batch, `[-1]`, `[-1,2]`, 1), 502}, {strings.Repeat(" ", compatLimit) + batch, 502},
	} {
		tr := compatTransport(func(r *http.Request) (*http.Response, error) {
			var q struct {
				Dimensions int
				Input      []string
			}
			if json.NewDecoder(r.Body).Decode(&q) != nil || q.Dimensions != 1 || len(q.Input) != 2 || q.Input[0] != "first" || q.Input[1] != "second" {
				t.Fatal("incorrect batch")
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(tc.reply))}, nil
		})
		b, s, e := compatRun(compatTestModel, []byte(`{"dimensions":1,"input":["first","second"]}`), tr)
		if s != tc.status || (e == nil) != (s == 200) {
			t.Fatalf("unexpected result: %d %v", s, e)
		}
		if s == 200 && (!strings.Contains(string(b), `"embedding":[0.25]`) || !strings.Contains(string(b), `"embedding":[-1]`)) {
			t.Fatal("incorrect float vectors")
		}
	}
}
