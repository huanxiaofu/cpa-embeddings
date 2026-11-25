package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"os"
	"time"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

const compatBase = "https://api.siliconflow.cn/v1"
const compatLimit = 8 << 20

var errCompatBinding = errors.New("no unambiguous SiliconFlow binding")

func compatModel(m string) bool {
	switch m {
	case "Qwen/Qwen3-VL-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B", "BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "BAAI/bge-large-en-v1.5", "Pro/BAAI/bge-m3":
		return true
	}
	return false
}

func compatBinding(model string) (keyBinding, error) {
	if !compatModel(model) {
		return keyBinding{}, errCompatBinding
	}
	path := os.Getenv("CPR_EMBED_CONFIG")
	if path == "" {
		path = "config.yaml"
	}
	raw, e := os.ReadFile(path)
	if e != nil {
		return keyBinding{}, errCompatBinding
	}
	var cfg struct {
		Proxy string `yaml:"proxy-url"`
		API   struct {
			Groups []groupBinding `yaml:"openai-compatibility"`
		} `yaml:"api-keys"`
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	if dec.Decode(&cfg) != nil || cfg.Proxy != "" || dec.Decode(new(yaml.Node)) != io.EOF {
		return keyBinding{}, errCompatBinding
	}
	// 无法忠实执行的排除规则或代理策略一律拒绝，不退回直连。
	blocked := func(k keyBinding) bool {
		return k.Disabled || k.Weight != nil && *k.Weight <= 0 || k.Proxy != "" || k.Prefix != "" || len(k.Headers) > 0 || len(k.Exclude) > 0
	}
	var result keyBinding
	for _, g := range cfg.API.Groups {
		for _, m := range g.Models {
			if m.Alias != model && !(m.Alias == "" && m.Name == model) {
				continue
			}
			if result.Key != "" || m.Alias != model || m.Name != model || g.Base != compatBase || blocked(g.keyBinding) || g.Key != "" {
				return keyBinding{}, errCompatBinding
			}
			for _, k := range g.Keys {
				if k.Disabled || k.Weight != nil && *k.Weight <= 0 {
					continue
				}
				if blocked(k) || k.Weight == nil || k.Key == "" || result.Key != "" {
					return keyBinding{}, errCompatBinding
				}
				result = k
			}
			if result.Key == "" {
				return keyBinding{}, errCompatBinding
			}
		}
	}
	if result.Key == "" {
		return result, errCompatBinding
	}
	return result, nil
}
func compatAllowed(model string) bool { _, e := compatBinding(model); return e == nil }

func runCompatEmbedding(model string, body []byte) ([]byte, int, error) {
	return compatRun(model, body, &http.Transport{DisableKeepAlives: true})
}
func compatRun(model string, body []byte, tr http.RoundTripper) ([]byte, int, error) {
	bad := errors.New("invalid embedding request")
	invalid := errors.New("invalid embedding upstream response")
	if !compatModel(model) {
		return nil, 400, bad
	}
	if len(body) > 6*embMaxInput+4096 {
		return nil, 413, bad
	}
	var in map[string]json.RawMessage
	if !utf8.Valid(body) || json.Unmarshal(body, &in) != nil {
		return nil, 400, bad
	}
	var m string
	if b := in["model"]; b != nil && (json.Unmarshal(b, &m) != nil || m != model) {
		return nil, 400, bad
	}
	if b := in["stream"]; b != nil && string(b) != "false" {
		return nil, 400, bad
	}
	format := "float"
	if b := in["encoding_format"]; b != nil && (string(b) == "null" || json.Unmarshal(b, &format) != nil || format != "float" && format != "base64") {
		return nil, 400, bad
	}
	input := in["input"]
	var texts []string
	if len(input) > 0 && input[0] == '"' {
		var s string
		if json.Unmarshal(input, &s) != nil {
			return nil, 400, bad
		}
		texts = []string{s}
	} else if json.Unmarshal(input, &texts) != nil {
		return nil, 400, bad
	}
	if len(texts) < 1 || len(texts) > 32 {
		return nil, 400, bad
	}
	size := 0
	for _, s := range texts {
		if s == "" {
			return nil, 400, bad
		}
		size += len(s)
	}
	if size > embMaxInput {
		return nil, 413, bad
	}
	req := map[string]any{"model": model, "encoding_format": "float", "input": input}
	dim := 0
	if b := in["dimensions"]; b != nil {
		if json.Unmarshal(b, &dim) != nil || dim <= 0 {
			return nil, 400, bad
		}
		req["dimensions"] = dim
	}
	k, e := compatBinding(model)
	if e != nil {
		return nil, 502, e
	}
	// 仅重建白名单字段；所有重定向和上游错误正文都在此截断。
	call := func(_ string, payload []byte) ([]byte, int, error) {
		r, e := http.NewRequest(http.MethodPost, compatBase+"/embeddings", bytes.NewReader(payload))
		if e != nil {
			return nil, 502, invalid
		}
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Authorization", "Bearer "+k.Key)
		c := http.Client{Transport: tr, Timeout: 25 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		resp, e := c.Do(r)
		if e != nil {
			return nil, 502, invalid
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, resp.StatusCode, invalid
		}
		b, e := io.ReadAll(io.LimitReader(resp.Body, compatLimit+1))
		return b, 200, e
	}
	var rep struct {
		Data []struct {
			Index  *int       `json:"index"`
			Vector []*float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			Prompt int `json:"prompt_tokens"`
			Total  int `json:"total_tokens"`
		} `json:"usage"`
	}
	status, e := embFetch(call, "", req, &rep, compatLimit)
	if e != nil {
		return nil, status, e
	}
	if len(rep.Data) != len(texts) || rep.Usage.Prompt <= 0 || rep.Usage.Total < rep.Usage.Prompt {
		return nil, 502, invalid
	}
	items := make([]embItem, len(texts))
	for i, d := range rep.Data {
		if dim == 0 {
			dim = len(d.Vector)
		}
		if d.Index == nil || *d.Index != i || dim == 0 || len(d.Vector) != dim {
			return nil, 502, invalid
		}
		for _, v := range d.Vector {
			if v == nil {
				return nil, 502, invalid
			}
		}
		var value any = d.Vector
		if format == "base64" {
			buf := make([]byte, 4*dim)
			for j, v := range d.Vector {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(*v))
			}
			value = base64.StdEncoding.EncodeToString(buf)
		}
		items[i] = embItem{"embedding", i, value}
	}
	out, e := json.Marshal(map[string]any{"object": "list", "model": model, "usage": rep.Usage, "data": items})
	return out, 200, e
}
