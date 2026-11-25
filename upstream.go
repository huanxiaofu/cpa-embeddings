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
type groupBinding struct {
	keyBinding `yaml:",inline"`
	Base       string       `yaml:"base-url"`
	Keys       []keyBinding `yaml:"keys"`
	Models     []struct {
		Name  string `yaml:"name"`
		Alias string `yaml:"alias"`
	} `yaml:"models"`
}

const googleBase = "https://generativelanguage.googleapis.com"

func liveBinding(raw []byte, model string) (keyBinding, error) {
	var cfg struct {
		API struct {
			Gemini []groupBinding `yaml:"gemini"`
		} `yaml:"api-keys"`
	}
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return keyBinding{}, errors.New("invalid CPA configuration")
	}
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
	for _, g := range cfg.API.Gemini {
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
	switch model {
	case "gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview":
	default:
		return nil, errors.New("unsupported embedding model")
	}
	f := os.Getenv("CPR_EMBED_CONFIG")
	if f == "" {
		f = "config.yaml"
	}
	raw, e := os.ReadFile(f)
	if e != nil {
		return nil, errors.New("cannot load CPA configuration")
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
