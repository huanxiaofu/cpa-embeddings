package main

import (
	"encoding/json"
	"testing"
)

func TestEmbeddingAdapter(t *testing.T) {
	const model = "gemini-embedding-2"
	for _, tc := range []struct {
		name, input, response string
		counts                []string
		wantStatus, wantTotal int
	}{
		{"single", `"a"`, `{"embedding":{"values":[1]},"usageMetadata":{"promptTokenCount":2}}`, nil, 200, 2},
		{"batch", `["a","b"]`, `{"embeddings":[{"values":[1]},{"values":[2]}],"usageMetadata":{"promptTokenCount":3}}`, nil, 200, 3},
		{"partial", `["a","b"]`, `{"embeddings":[{"values":[1],"usageMetadata":{"promptTokenCount":2}},{"values":[2]}]}`, []string{`{"totalTokens":4}`}, 200, 6},
		{"missing", `"a"`, `{"embedding":{"values":[1]}}`, []string{`{}`}, 502, 0},
		{"null value", `"a"`, `{"embedding":{"values":[null]}}`, nil, 502, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			call := func(action string, p []byte) ([]byte, int, error) {
				calls++
				if calls == 1 {
					want := "embedContent"
					if tc.input[0] == '[' {
						want = "batchEmbedContents"
					}
					if action != want {
						t.Fatal(action)
					}
					return []byte(tc.response), 200, nil
				}
				if action != "countTokens" || calls-2 >= len(tc.counts) {
					t.Fatal(action)
				}
				if tc.name == "partial" && string(p) != `{"contents":[{"parts":[{"text":"b"}]}]}` {
					t.Fatal(string(p))
				}
				return []byte(tc.counts[calls-2]), 200, nil
			}
			b, s, err := embRun(model, []byte(`{"dimensions":1,"encoding_format":"base64","prompt_tokens":99,"input":`+tc.input+`}`), func(m string) (embCall, error) {
				if m != model {
					t.Fatal(m)
				}
				return call, nil
			})
			if s != tc.wantStatus || (err == nil) != (s == 200) || calls != 1+len(tc.counts) {
				t.Fatal(s, err, calls)
			}
			if s != 200 {
				return
			}
			var r struct {
				Usage map[string]int
				Data  []struct {
					Index     int
					Embedding string
				}
			}
			if json.Unmarshal(b, &r) != nil || r.Usage["total_tokens"] != tc.wantTotal || r.Usage["prompt_tokens"] != tc.wantTotal || len(r.Data) < 1 || r.Data[0].Embedding != "AACAPw==" {
				t.Fatal(string(b))
			}
			if tc.input[0] == '[' && (len(r.Data) != 2 || r.Data[1].Index != 1 || r.Data[1].Embedding != "AAAAQA==") {
				t.Fatal(string(b))
			}
		})
	}
}
func TestEmbeddingRejectsInput(t *testing.T) {
	for _, body := range []string{`{}`, `{"input":[]}`, `{"input":[null]}`, `{"input":"a","dimensions":3073}`, `{"input":"a","model":"other"}`, `{"input":"a","stream":true}`, `{"input":"a","encoding_format":null}`} {
		_, s, err := embRun("gemini-embedding-001", []byte(body), nil)
		if s != 400 || err == nil {
			t.Fatal(body, s)
		}
	}
}
