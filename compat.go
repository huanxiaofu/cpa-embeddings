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
	"net/url"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// OpenAI-compatible 上游只从 CPA api-keys.openai-compatibility 的显式专用组读取：
// 组名必须以 -embeddings 结尾，避免把聊天、重排等渠道误当成向量上游。
const compatSuffix = "-embeddings"
const compatLimit = 8 << 20

var (
	errCompatBinding = errors.New("no unambiguous OpenAI-compatible embedding binding")
	errCompatBad     = errors.New("invalid embedding request")
	errCompatInvalid = errors.New("invalid embedding upstream response")
)

// compatTarget 是唯一确定的一次上游调用绑定：向上游发送的真实模型名、实际 HTTPS base
// 和单条凭据。客户端看到的模型名（alias 或 name）只用于匹配和回写响应。
type compatTarget struct {
	Name   string
	Base   string
	Key    string
	Cohere bool
}

func compatConfigPath() string {
	if p := os.Getenv("CPR_EMBED_CONFIG"); p != "" {
		return p
	}
	return "config.yaml"
}

// 常用上游（OpenAI、DashScope 兼容模式、Mistral、私有兼容部署）都允许带 HTTPS 路径，
// 但禁止凭据、查询串和片段；返回去掉尾部斜杠的 base。
func compatBase(raw string) (string, bool) {
	b := strings.TrimRight(strings.TrimSpace(raw), "/")
	if b == "" {
		return "", false
	}
	u, err := url.Parse(b)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.Opaque != "" ||
		u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
		return "", false
	}
	return b, true
}

// CPA 的可用性规则：停用或权重不大于 0 的组/凭据不参与；权重未填按默认 1。
func compatUsable(k keyBinding) bool {
	return !k.Disabled && (k.Weight == nil || *k.Weight > 0)
}

// 无法忠实执行的前缀、代理、附加头和排除规则一律拒绝，不静默忽略或改走其他渠道。
func compatUnsupported(k keyBinding) bool {
	return k.Prefix != "" || k.Proxy != "" || len(k.Headers) > 0 || len(k.Exclude) > 0
}

// compatBinding 解析出客户端模型名对应的唯一上游绑定。
// 找不到可用组、同名模型跨组冲突或多个可用凭据都返回 error，不做猜测。
func compatBinding(model string) (compatTarget, error) {
	if model == "" {
		return compatTarget{}, errCompatBinding
	}
	raw, err := os.ReadFile(compatConfigPath())
	if err != nil {
		return compatTarget{}, errCompatBinding
	}
	return compatBindingFrom(raw, model)
}

// 已有渠道无须改名：插件实例可显式写入组名到协议的映射。
type channelOptions struct {
	Plugins struct {
		Configs map[string]struct {
			Protocols map[string]string `yaml:"upstream-protocols"`
		} `yaml:"configs"`
	} `yaml:"plugins"`
}

func (c channelOptions) protocol(name string) string {
	if v, ok := c.Plugins.Configs[abiPluginID].Protocols[name]; ok {
		if v == "openai" || v == "cohere" {
			return v
		}
		return ""
	}
	if name == "cohere-embeddings" || strings.HasSuffix(name, "-cohere-embeddings") {
		return "cohere"
	}
	if len(name) > len(compatSuffix) && strings.HasSuffix(name, compatSuffix) {
		return "openai"
	}
	return ""
}
func compatBindingFrom(raw []byte, model string) (compatTarget, error) {
	var cfg struct {
		channelOptions `yaml:",inline"`
		Requests       struct {
			Proxy string `yaml:"proxy-url"`
		} `yaml:"requests"`
		Proxy string `yaml:"proxy-url"`
		API   struct {
			Groups []groupBinding `yaml:"openai-compatibility"`
		} `yaml:"api-keys"`
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	// 无法忠实执行的全局代理策略或多文档配置一律拒绝，不退回直连。
	if dec.Decode(&cfg) != nil || cfg.Proxy != "" || cfg.Requests.Proxy != "" || dec.Decode(new(yaml.Node)) != io.EOF {
		return compatTarget{}, errCompatBinding
	}
	var found []compatTarget
	for _, g := range cfg.API.Groups {
		kind := cfg.protocol(g.Name)
		if kind == "" {
			continue
		}
		// 客户端名优先取 alias；未设置 alias 时直接使用真实 name。
		var real []string
		for _, m := range g.Models {
			if m.Name == "" {
				continue
			}
			if m.Alias != "" && m.Alias == model || m.Alias == "" && m.Name == model {
				real = append(real, m.Name)
			}
		}
		if len(real) == 0 {
			continue
		}
		// 停用的组不再提供该模型；其余情况必须能唯一解析，避免挑错渠道。
		if !compatUsable(g.keyBinding) {
			continue
		}
		if len(real) != 1 || compatUnsupported(g.keyBinding) {
			return compatTarget{}, errCompatBinding
		}
		keys := make([]keyBinding, 0, len(g.Keys)+1)
		if g.Key != "" {
			keys = append(keys, g.keyBinding)
		}
		keys = append(keys, g.Keys...)
		usable := 0
		for _, k := range keys {
			if k.Key == "" || !compatUsable(k) {
				continue
			}
			if compatUnsupported(k) {
				return compatTarget{}, errCompatBinding
			}
			usable++
		}
		if usable == 0 {
			continue
		}
		base, ok := compatBase(g.Base)
		if !ok {
			return compatTarget{}, errCompatBinding
		}
		for _, k := range keys {
			if k.Key == "" || !compatUsable(k) || compatUnsupported(k) {
				continue
			}
			found = append(found, compatTarget{Name: real[0], Base: base, Key: k.Key, Cohere: kind == "cohere"})
		}
	}
	if len(found) != 1 {
		return compatTarget{}, errCompatBinding
	}
	return found[0], nil
}

// compatAllowed 是共享入口：客户端模型名（alias 或 name）是否登记在显式专用向量组。
func compatAllowed(model string) bool {
	_, err := compatBinding(model)
	return err == nil
}

// compatModel 保留给现有路由与分发代码使用，语义与 compatAllowed 相同；
// 不再硬编码型号或上游域名。
func compatModel(model string) bool { return compatAllowed(model) }

func runCompatEmbedding(model string, body []byte) ([]byte, int, error) {
	return compatRun(model, body, &http.Transport{DisableKeepAlives: true})
}

func compatRun(model string, body []byte, tr http.RoundTripper) ([]byte, int, error) {
	target, err := compatBinding(model)
	if err != nil {
		return nil, 502, err
	}
	return compatWithTarget(model, body, target, tr)
}
func compatWithTarget(model string, body []byte, target compatTarget, tr http.RoundTripper) ([]byte, int, error) {
	if target.Cohere {
		return cohereRun(model, body, target, tr)
	}
	invalid := errCompatInvalid
	if len(body) > 6*embMaxInput+4096 {
		return nil, 413, errCompatBad
	}
	var in map[string]json.RawMessage
	if !utf8.Valid(body) || json.Unmarshal(body, &in) != nil {
		return nil, 400, errCompatBad
	}
	if b := in["model"]; b != nil {
		var m string
		if json.Unmarshal(b, &m) != nil || m != model {
			return nil, 400, errCompatBad
		}
	}
	if b := in["stream"]; b != nil && string(b) != "false" {
		return nil, 400, errCompatBad
	}
	// 客户端未指定编码格式时不向上游发送 encoding_format，避免上游不支持该字段。
	format := "float"
	if b := in["encoding_format"]; b != nil {
		if string(b) == "null" || json.Unmarshal(b, &format) != nil || format != "float" && format != "base64" {
			return nil, 400, errCompatBad
		}
	}
	input := in["input"]
	var texts []string
	if len(input) > 0 && input[0] == '"' {
		var s string
		if json.Unmarshal(input, &s) != nil {
			return nil, 400, errCompatBad
		}
		texts = []string{s}
	} else if json.Unmarshal(input, &texts) != nil {
		return nil, 400, errCompatBad
	}
	if len(texts) < 1 || len(texts) > 32 {
		return nil, 400, errCompatBad
	}
	size := 0
	for _, s := range texts {
		if s == "" {
			return nil, 400, errCompatBad
		}
		size += len(s)
	}
	if size > embMaxInput {
		return nil, 413, errCompatBad
	}
	dim := 0
	if b := in["dimensions"]; b != nil {
		if json.Unmarshal(b, &dim) != nil || dim <= 0 {
			return nil, 400, errCompatBad
		}
	}
	if in["input_type"] != nil {
		return nil, 400, errCompatBad
	}
	// 仅重建白名单字段：客户端自带的 URL、认证头等一律不透传，模型名换成真实 name。
	req := map[string]any{"model": target.Name, "input": input}
	// 编码只影响下游输出，不要求上游实现 encoding_format。
	if dim > 0 {
		req["dimensions"] = dim
	}
	endpoint := target.Base + "/embeddings"
	// 所有重定向和上游错误正文都在此截断，只向客户端返回固定错误。
	call := func(_ string, payload []byte) ([]byte, int, error) {
		r, e := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(payload))
		if e != nil {
			return nil, 502, invalid
		}
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Authorization", "Bearer "+target.Key)
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
			Index     *int            `json:"index"`
			Embedding json.RawMessage `json:"embedding"`
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
	// usage 必须是上游真实回报；缺失或倒退的计数不能当作成功，也不估算。
	if len(rep.Data) != len(texts) || rep.Usage.Prompt <= 0 || rep.Usage.Total < rep.Usage.Prompt {
		return nil, 502, invalid
	}
	// 上游可能乱序返回批量结果，按 index 复原；重复或越界的索引一律拒绝。
	vectors := make([][]float32, len(texts))
	seen := make([]bool, len(texts))
	want := dim
	for _, d := range rep.Data {
		if d.Index == nil || *d.Index < 0 || *d.Index >= len(texts) || seen[*d.Index] {
			return nil, 502, invalid
		}
		v, ok := compatVector(d.Embedding)
		if !ok || len(v) == 0 {
			return nil, 502, invalid
		}
		if want == 0 {
			want = len(v)
		} else if len(v) != want {
			return nil, 502, invalid
		}
		vectors[*d.Index] = v
		seen[*d.Index] = true
	}
	items := make([]embItem, len(vectors))
	for i, v := range vectors {
		if !seen[i] {
			return nil, 502, invalid
		}
		var value any = v
		if format == "base64" {
			buf := make([]byte, 4*len(v))
			for j, f := range v {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(f))
			}
			value = base64.StdEncoding.EncodeToString(buf)
		}
		items[i] = embItem{"embedding", i, value}
	}
	// 响应 model 保留客户端使用的 alias（或 name），便于客户端按自己的模型名核对。
	out, e := json.Marshal(map[string]any{"object": "list", "model": model, "usage": rep.Usage, "data": items})
	return out, 200, e
}

// 兼容两种常见 embedding 表示：float 数组，或 base64 编码的小端 float32。
// null 元素、空向量和非法 base64 都视为无效响应。
func compatVector(raw json.RawMessage) ([]float32, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return nil, false
	}
	if raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) != nil {
			return nil, false
		}
		buf, err := base64.StdEncoding.DecodeString(s)
		if err != nil || len(buf) == 0 || len(buf)%4 != 0 {
			return nil, false
		}
		v := make([]float32, len(buf)/4)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
			if math.IsNaN(float64(v[i])) || math.IsInf(float64(v[i]), 0) {
				return nil, false
			}
		}
		return v, true
	}
	var parts []*float32
	if json.Unmarshal(raw, &parts) != nil || len(parts) == 0 {
		return nil, false
	}
	v := make([]float32, len(parts))
	for i, p := range parts {
		if p == nil {
			return nil, false
		}
		v[i] = *p
	}
	return v, true
}
