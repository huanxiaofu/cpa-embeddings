package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"unicode/utf8"
)

const embMaxInput = 256 << 10

type embPart struct {
	Text string `json:"text"`
}
type embContent struct {
	Parts []embPart `json:"parts"`
}
type embRequest struct {
	Model     string     `json:"model"`
	Content   embContent `json:"content"`
	Dimension int        `json:"outputDimensionality,omitempty"`
}
type embUsage struct {
	Count json.RawMessage `json:"promptTokenCount"`
}
type embVector struct {
	Values []json.RawMessage `json:"values"`
	Usage  embUsage          `json:"usageMetadata"`
}
type embReply struct {
	Single *embVector  `json:"embedding"`
	Batch  []embVector `json:"embeddings"`
	Usage  embUsage    `json:"usageMetadata"`
}
type embItem struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}
type embCall = func(string, []byte) ([]byte, int, error)

// 只向上返回固定错误，不透传上游正文、凭证或错误信息。
func embFetch(call embCall, action string, req, dst any, limit int) (int, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return 500, errors.New("cannot encode embedding request")
	}
	b, s, err := call(action, payload)
	if err != nil || s < 200 || s >= 300 {
		switch s {
		case 400, 413, 429, 503, 504:
		default:
			s = 502
		}
		return s, errors.New("embedding upstream unavailable")
	}
	if len(b) > limit || json.Unmarshal(b, dst) != nil {
		return 502, errors.New("invalid embedding upstream response")
	}
	return 200, nil
}

// 空字段表示需要补查；显式无效的计数不能视作成功。
func embCount(b json.RawMessage) (int, bool) {
	if b == nil {
		return 0, true
	}
	var n int
	ok := json.Unmarshal(b, &n) == nil && n > 0
	return n, ok
}

func runEmbedding(model string, body []byte) ([]byte, int, error) {
	return embRun(model, body, upstreamFor)
}

func embRun(model string, body []byte, resolve func(string) (embCall, error)) ([]byte, int, error) {
	bad := errors.New("invalid embedding request")
	invalid := errors.New("invalid embedding upstream response")
	switch model {
	case "gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview":
	default:
		return nil, 400, bad
	}
	// 允许 JSON 转义带来的膨胀，但在解析前也限制请求大小。
	if len(body) > 6*embMaxInput+4096 {
		return nil, 413, bad
	}
	var in map[string]json.RawMessage
	if !utf8.Valid(body) || json.Unmarshal(body, &in) != nil {
		return nil, 400, bad
	}
	if b, ok := in["model"]; ok {
		var m string
		if json.Unmarshal(b, &m) != nil || m != model {
			return nil, 400, bad
		}
	}
	if b, ok := in["stream"]; ok && string(b) != "false" {
		return nil, 400, bad
	}
	dim := 0
	if b, ok := in["dimensions"]; ok && (json.Unmarshal(b, &dim) != nil || dim < 1 || dim > 3072) {
		return nil, 400, bad
	}
	format := "float"
	if b, ok := in["encoding_format"]; ok && (string(b) == "null" || json.Unmarshal(b, &format) != nil || (format != "float" && format != "base64")) {
		return nil, 400, bad
	}
	raw := in["input"]
	if len(raw) == 0 {
		return nil, 400, bad
	}
	batch := raw[0] == '['
	var texts []string
	if batch {
		if json.Unmarshal(raw, &texts) != nil {
			return nil, 400, bad
		}
	} else {
		var s string
		if json.Unmarshal(raw, &s) != nil {
			return nil, 400, bad
		}
		texts = []string{s}
	}
	if len(texts) == 0 || len(texts) > 32 {
		return nil, 400, bad
	}
	reqs := make([]embRequest, len(texts))
	size := 0
	for i, s := range texts {
		size += len(s)
		if size > embMaxInput {
			return nil, 413, bad
		}
		if s == "" {
			return nil, 400, bad
		}
		reqs[i] = embRequest{"models/" + model, embContent{[]embPart{{s}}}, dim}
	}
	call, err := resolve(model)
	if err != nil || call == nil {
		return nil, 502, errors.New("embedding upstream unavailable")
	}
	var rep embReply
	var req any = reqs[0]
	action := "embedContent"
	if batch {
		action = "batchEmbedContents"
		req = struct {
			Requests []embRequest `json:"requests"`
		}{reqs}
	}
	s, err := embFetch(call, action, req, &rep, len(reqs)*(256<<10))
	if err != nil {
		return nil, s, err
	}
	vs := rep.Batch
	if !batch {
		if rep.Single == nil {
			return nil, 502, invalid
		}
		vs = []embVector{*rep.Single}
	}
	if len(vs) != len(reqs) {
		return nil, 502, invalid
	}
	expected := dim
	if expected == 0 {
		expected = 3072
	}
	items := make([]embItem, len(vs))
	counts := make([]int, len(vs))
	total, ok := embCount(rep.Usage.Count)
	if !ok {
		return nil, 502, invalid
	}
	for i, v := range vs {
		if len(v.Values) != expected {
			return nil, 502, invalid
		}
		counts[i], ok = embCount(v.Usage.Count)
		if !ok {
			return nil, 502, invalid
		}
		vals := make([]float32, expected)
		for j, b := range v.Values {
			if bytes.Equal(b, []byte("null")) || json.Unmarshal(b, &vals[j]) != nil {
				return nil, 502, invalid
			}
		}
		var value any = vals
		if format == "base64" {
			buf := make([]byte, 4*len(vals))
			for j, f := range vals {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(f))
			}
			value = base64.StdEncoding.EncodeToString(buf)
		}
		items[i] = embItem{"embedding", i, value}
	}
	// 批量总数优先；没有总数时，只为缺少计数的文本单独查询。
	sum := 0
	missing := 0
	for i, n := range counts {
		if n == 0 {
			missing++
			if total != 0 {
				continue
			}
			var r struct {
				Count json.RawMessage `json:"totalTokens"`
			}
			s, err = embFetch(call, "countTokens", struct {
				Contents []embContent `json:"contents"`
			}{[]embContent{reqs[i].Content}}, &r, 64<<10)
			if err != nil {
				return nil, s, err
			}
			n, ok = embCount(r.Count)
			if !ok || n == 0 {
				return nil, 502, invalid
			}
		}
		if sum > math.MaxInt-n {
			return nil, 502, invalid
		}
		sum += n
	}
	if total == 0 {
		total = sum
	} else if total-sum < missing || (missing == 0 && total != sum) {
		return nil, 502, invalid
	}
	out, err := json.Marshal(map[string]any{"object": "list", "model": model, "data": items, "usage": map[string]int{"prompt_tokens": total, "total_tokens": total}})
	if err != nil {
		return nil, 500, errors.New("cannot encode embedding response")
	}
	return out, 200, nil
}
