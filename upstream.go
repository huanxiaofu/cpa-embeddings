package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// 仅读取 CPA 当前配置中的 Gemini 模型绑定，不缓存上游密钥，也不复制下游 Key。
type keyBinding struct {
	Key      string            `yaml:"api-key"`
	Disabled bool              `yaml:"disabled"`
	Weight   *int              `yaml:"weight"`
	Prefix   string            `yaml:"prefix"`
	Proxy    string            `yaml:"proxy-url"`
	Exclude  []string          `yaml:"excluded-models"`
	Headers  map[string]string `yaml:"headers"`
}

// modelBinding 是 CPA 组内登记的单个模型条目。向量识别只看真实 name 与显式
// output-modalities；alias 仅是客户端寻址名，不能反向证明模型属于向量族。
type modelBinding struct {
	Name     string   `yaml:"name"`
	Alias    string   `yaml:"alias"`
	Modality []string `yaml:"output-modalities"`
}

// UnmarshalYAML 同时接受 output-modalities 与 output_modalities，并容忍单值写法，
// 避免一处拼写差异让整份 CPA 配置解析失败。
func (m *modelBinding) UnmarshalYAML(node *yaml.Node) error {
	var raw struct {
		Name  string    `yaml:"name"`
		Alias string    `yaml:"alias"`
		Kebab yaml.Node `yaml:"output-modalities"`
		Snake yaml.Node `yaml:"output_modalities"`
	}
	if err := node.Decode(&raw); err != nil {
		return err
	}
	kebab, err := modalityList(&raw.Kebab)
	if err != nil {
		return err
	}
	snake, err := modalityList(&raw.Snake)
	if err != nil {
		return err
	}
	m.Name, m.Alias = raw.Name, raw.Alias
	m.Modality = append(kebab, snake...)
	return nil
}

// modalityList 把标量或序列形态的输出模态统一成切片；空值与 null 视为未声明。
func modalityList(node *yaml.Node) ([]string, error) {
	if node == nil || node.Kind == 0 || node.Tag == "!!null" {
		return nil, nil
	}
	if node.Kind == yaml.ScalarNode {
		var s string
		if err := node.Decode(&s); err != nil {
			return nil, err
		}
		if s = strings.TrimSpace(s); s == "" {
			return nil, nil
		}
		return []string{s}, nil
	}
	var list []string
	if err := node.Decode(&list); err != nil {
		return nil, err
	}
	return list, nil
}

type groupBinding struct {
	keyBinding `yaml:",inline"`
	Name       string         `yaml:"name"`
	Base       string         `yaml:"base-url"`
	Keys       []keyBinding   `yaml:"keys"`
	Models     []modelBinding `yaml:"models"`
}

const googleBase = "https://generativelanguage.googleapis.com"

// geminiModelName 只接受 Gemini 向量族的单段型号名：拒绝路径分隔符、动作冒号、
// 查询串、空白和非 ASCII 字符，避免拼出聊天动作或任意 Google URL 路径。
func geminiModelName(model string) bool {
	var tail string
	switch {
	case strings.HasPrefix(model, "gemini-embedding-"):
		tail = model[len("gemini-embedding-"):]
	case strings.HasPrefix(model, "text-embedding-"):
		tail = model[len("text-embedding-"):]
	default:
		return false
	}
	if tail == "" || len(model) > 96 || strings.HasPrefix(tail, "-") || strings.HasSuffix(tail, "-") ||
		strings.HasPrefix(tail, ".") || strings.HasSuffix(tail, ".") || strings.Contains(tail, "..") {
		return false
	}
	for _, c := range tail {
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '-', c == '.':
		default:
			return false
		}
	}
	return true
}

// geminiConfig 读取 CPA 配置原文，路径可由 CPR_EMBED_CONFIG 覆盖。
func geminiConfig() (string, []byte, error) {
	f := os.Getenv("CPR_EMBED_CONFIG")
	if f == "" {
		f = "config.yaml"
	}
	raw, e := os.ReadFile(f)
	if e != nil {
		return f, nil, errors.New("cannot load CPA configuration")
	}
	return f, raw, nil
}

func parseGemini(raw []byte) ([]groupBinding, error) {
	var cfg struct {
		API struct {
			Gemini []groupBinding `yaml:"gemini"`
		} `yaml:"api-keys"`
	}
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return nil, errors.New("invalid CPA configuration")
	}
	return cfg.API.Gemini, nil
}

// geminiDeclared 判断型号是否在原生 googleBase 组里按 name 显式登记。
// Native 不做别名转发，因此 alias 必须为空或与 name 相同。
func geminiDeclared(raw []byte, model string) bool {
	groups, err := parseGemini(raw)
	if err != nil {
		return false
	}
	for _, g := range groups {
		if strings.TrimRight(g.Base, "/") != googleBase {
			continue
		}
		for _, m := range g.Models {
			if m.Name == model && (m.Alias == model || m.Alias == "") {
				return true
			}
		}
	}
	return false
}

// validGeminiModel 供上层分派：型号形态合法，且已在 CPA 的 api-keys.gemini 中
// 按 name 显式登记。这里只回答"是否路由到 Google 原生端点"，凭据可用性仍由
// liveBinding 在真正调用上游时确定。
func validGeminiModel(model string) bool {
	if !geminiModelName(model) {
		return false
	}
	_, raw, err := geminiConfig()
	return err == nil && geminiDeclared(raw, model)
}

func liveBinding(raw []byte, model string) (keyBinding, error) {
	if !geminiModelName(model) {
		return keyBinding{}, errors.New("unsupported embedding model")
	}
	groups, e := parseGemini(raw)
	if e != nil {
		return keyBinding{}, e
	}
	return geminiBinding(groups, model)
}
func geminiBinding(groups []groupBinding, model string) (keyBinding, error) {
	var bk []keyBinding
	blocked := func(b keyBinding) bool {
		if b.Disabled || b.Weight != nil && *b.Weight <= 0 || b.Prefix != "" || len(b.Headers) > 0 {
			return true
		}
		for _, x := range b.Exclude {
			if hit, e := filepath.Match(x, model); e != nil || hit {
				return true
			}
		}
		return false
	}
	for _, g := range groups {
		if blocked(g.keyBinding) || strings.TrimRight(g.Base, "/") != googleBase {
			continue
		}
		allowed := false
		for _, m := range g.Models {
			if m.Name == model && (m.Alias == model || m.Alias == "") {
				allowed = true
				break
			}
		}
		if !allowed {
			continue
		}
		for _, k := range g.Keys {
			if k.Key == "" || blocked(k) {
				continue
			}
			if k.Proxy == "" {
				k.Proxy = g.Proxy
			}
			bk = append(bk, k)
		}
	}
	if len(bk) != 1 {
		return keyBinding{}, errors.New("need exactly one enabled matching Gemini credential")
	}
	return bk[0], nil
}
func upstreamFor(model string) (func(string, []byte) ([]byte, int, error), error) {
	// 型号必须形态合法且在 CPA 中原生登记，不再硬编码具体型号清单。
	if !geminiModelName(model) {
		return nil, errors.New("unsupported embedding model")
	}
	f, raw, e := geminiConfig()
	if e != nil {
		return nil, e
	}
	if !geminiDeclared(raw, model) {
		return nil, errors.New("unsupported embedding model")
	}
	binding, e := liveBinding(raw, model)
	if e != nil {
		return nil, e
	}
	// Docker 内部出口名称可以由管理员映射到同一出口对宿主开放的本地监听地址。
	var local struct {
		ProxyAliases map[string]string `yaml:"proxy_aliases"`
	}
	loc := filepath.Join(filepath.Dir(f), "cpr-embeddings.local.yaml")
	data, e := os.ReadFile(loc)
	if e == nil {
		if e = yaml.Unmarshal(data, &local); e != nil {
			return nil, errors.New("invalid local proxy mapping")
		}
	} else if !errors.Is(e, os.ErrNotExist) {
		return nil, errors.New("cannot read proxy mapping")
	}
	proxy := binding.Proxy
	if dest, ok := local.ProxyAliases[proxy]; ok && proxy != "" {
		if dest == "" {
			return nil, errors.New("empty proxy mapping")
		}
		proxy = dest
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil
	tr.DisableKeepAlives = true
	if proxy != "" {
		u, err := url.Parse(proxy)
		if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https" && u.Scheme != "socks5" && u.Scheme != "socks5h") {
			return nil, errors.New("unsupported egress proxy")
		}
		tr.Proxy = http.ProxyURL(u)
	}
	c := http.Client{Transport: tr, Timeout: 25 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	return func(action string, body []byte) ([]byte, int, error) {
		switch action {
		case "embedContent", "batchEmbedContents", "countTokens":
		default:
			return nil, 400, errors.New("unsupported operation")
		}
		// 型号名在进门前已校验为单段标识符，这里再挡一次，确保 URL 只可能拼出
		// 白名单动作，不会变成聊天或其它 Google 路径。
		if !geminiModelName(model) {
			return nil, 400, errors.New("unsupported operation")
		}
		req, e := http.NewRequest(http.MethodPost, googleBase+"/v1beta/models/"+model+":"+action, bytes.NewReader(body))
		if e != nil {
			return nil, 500, errors.New("request construction failed")
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-goog-api-key", binding.Key)
		r, e := c.Do(req)
		if e != nil {
			return nil, 502, errors.New("Gemini upstream connection failed")
		}
		defer r.Body.Close()
		if r.StatusCode != 200 {
			status := r.StatusCode
			if status < 400 || status > 599 {
				status = 502
			}
			return nil, status, fmt.Errorf("Gemini upstream HTTP %d", status)
		}
		const limit = 8 << 20
		b, e := io.ReadAll(io.LimitReader(r.Body, limit+1))
		if e != nil || len(b) > limit {
			return nil, 502, errors.New("invalid upstream response size")
		}
		return b, 200, nil
	}, nil
}
