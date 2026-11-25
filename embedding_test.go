package main

import (
	"encoding/json"
	"strconv"
	"strings"
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

// 不同型号的默认维度和可调上限不同：宽度只按响应逐项校验，不写死 3072。
func TestEmbeddingWidthRules(t *testing.T) {
	const model = "text-embedding-004"
	vec := func(n int) string {
		s := make([]string, n)
		for i := range s {
			s[i] = "0.5"
		}
		return "[" + strings.Join(s, ",") + "]"
	}
	for _, tc := range []struct {
		name, dims, input, response string
		wantStatus, wantTotal       int
	}{
		{"默认维度由响应决定", "", `"a"`, `{"embedding":{"values":` + vec(1024) + `},"usageMetadata":{"promptTokenCount":1}}`, 200, 1},
		{"维度与请求一致", `,"dimensions":2`, `"a"`, `{"embedding":{"values":` + vec(2) + `},"usageMetadata":{"promptTokenCount":1}}`, 200, 1},
		{"维度与请求不符", `,"dimensions":3`, `"a"`, `{"embedding":{"values":` + vec(2) + `},"usageMetadata":{"promptTokenCount":1}}`, 502, 0},
		{"空向量", "", `"a"`, `{"embedding":{"values":[]},"usageMetadata":{"promptTokenCount":1}}`, 502, 0},
		{"批量长度不一致", "", `["a","b"]`, `{"embeddings":[{"values":` + vec(2) + `},{"values":` + vec(3) + `}],"usageMetadata":{"promptTokenCount":5}}`, 502, 0},
		{"响应超出上限", "", `"a"`, `{"embedding":{"values":` + vec(embMaxDim+1) + `},"usageMetadata":{"promptTokenCount":1}}`, 502, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got string
			call := func(action string, p []byte) ([]byte, int, error) {
				got = string(p)
				return []byte(tc.response), 200, nil
			}
			b, s, err := embRun(model, []byte(`{"input":`+tc.input+tc.dims+`}`), func(m string) (embCall, error) {
				if m != model {
					t.Fatal(m)
				}
				return call, nil
			})
			// 请求里没有 dimensions 时不得凭空补一个默认值。
			if strings.Contains(got, "outputDimensionality") != strings.Contains(tc.dims, "dimensions") {
				t.Fatal(got)
			}
			if s != tc.wantStatus || (err == nil) != (s == 200) {
				t.Fatal(s, err)
			}
			if s != 200 {
				return
			}
			var r struct {
				Usage map[string]int
				Data  []struct {
					Embedding []float32
				}
			}
			if json.Unmarshal(b, &r) != nil || r.Usage["total_tokens"] != tc.wantTotal || len(r.Data) != 1 || len(r.Data[0].Embedding) == 0 {
				t.Fatal(string(b))
			}
			want := 1024
			if tc.dims != "" {
				want, _ = strconv.Atoi(strings.TrimPrefix(tc.dims, `,"dimensions":`))
			}
			if len(r.Data[0].Embedding) != want {
				t.Fatal(len(r.Data[0].Embedding), want)
			}
		})
	}
}

func TestEmbeddingRejectsInput(t *testing.T) {
	for _, tc := range []struct{ model, body string }{
		{"gemini-embedding-001", `{}`},
		{"gemini-embedding-001", `{"input":[]}`},
		{"gemini-embedding-001", `{"input":[null]}`},
		{"gemini-embedding-001", `{"input":"a","dimensions":0}`},
		{"gemini-embedding-001", `{"input":"a","dimensions":` + strconv.Itoa(embMaxDim+1) + `}`},
		{"gemini-embedding-001", `{"input":"a","dimensions":-1}`},
		{"gemini-embedding-001", `{"input":"a","model":"other"}`},
		{"gemini-embedding-001", `{"input":"a","stream":true}`},
		{"gemini-embedding-001", `{"input":"a","encoding_format":null}`},
		// 只能走向量族型号：聊天型号、带路径或动作冒号的名字一律不接纳。
		{"gemini-1.5-flash", `{"input":"a"}`},
		{"models/gemini-embedding-001", `{"input":"a"}`},
		{"gemini-embedding-001:embedContent", `{"input":"a"}`},
		{"gemini-embedding-001/../chat", `{"input":"a"}`},
		{"", `{"input":"a"}`},
	} {
		_, s, err := embRun(tc.model, []byte(tc.body), nil)
		if s != 400 || err == nil {
			t.Fatal(tc.model, tc.body, s)
		}
	}
}
