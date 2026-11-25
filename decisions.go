package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
)

type rulesMap map[string]map[string]string

var panelMu sync.Mutex

func rulesPath() string {
	return filepath.Join(filepath.Dir(compatConfigPath()), "cpa-embeddings-state", "overrides.json")
}
func modeOK(s string) bool { return s == "allow" || s == "deny" || s == "auto" }

// 面板仅写独立的小型规则文件，不重写 CPA 渠道、凭据或其他插件配置。
func decisionSnapshot() (*catalogConfig, string, rulesMap, error) {
	raw, e := readConfigBytes()
	if e != nil {
		return nil, "", nil, e
	}
	cfg, e := parseCatalog(raw)
	if e != nil {
		return nil, "", nil, e
	}
	rules := rulesMap{}
	data, e := os.ReadFile(rulesPath())
	if e != nil && !os.IsNotExist(e) {
		return nil, "", nil, e
	}
	if e == nil && (len(data) > 1<<20 || json.Unmarshal(data, &rules) != nil || rules == nil) {
		return nil, "", nil, errors.New("invalid decisions")
	}
	item := cfg.Plugins.Configs[abiPluginID]
	if item.ModelOverrides == nil {
		item.ModelOverrides = make(rulesMap)
	}
	for ch, m := range rules {
		if m == nil {
			return nil, "", nil, errors.New("invalid decisions")
		}
		if item.ModelOverrides[ch] == nil {
			item.ModelOverrides[ch] = map[string]string{}
		}
		for n, v := range m {
			if !modeOK(v) {
				return nil, "", nil, errors.New("invalid decision")
			}
			if v == "auto" {
				delete(item.ModelOverrides[ch], n)
			} else {
				item.ModelOverrides[ch][n] = v
			}
		}
	}
	if cfg.Plugins.Configs != nil {
		cfg.Plugins.Configs[abiPluginID] = item
	} else {
		cfg.Plugins.Configs = map[string]pluginSettings{abiPluginID: item}
	}
	h := sha256.New()
	h.Write(raw)
	h.Write([]byte{0})
	h.Write(data)
	return cfg, hex.EncodeToString(h.Sum(nil)), rules, nil
}
func activeCatalog() (*catalogConfig, string, error) {
	cfg, rev, _, e := decisionSnapshot()
	return cfg, rev, e
}

type panelRow struct {
	Channel   string `json:"channel"`
	Name      string `json:"name"`
	Alias     string `json:"alias"`
	Automatic bool   `json:"automatic"`
	Mode      string `json:"mode"`
	Enabled   bool   `json:"enabled"`
	Protocol  string `json:"protocol"`
}

func panelListing(cfg *catalogConfig, rev string) []byte {
	rows := []panelRow{}
	for _, g := range cfg.API.Compat {
		for _, m := range g.Models {
			a := m.Alias
			if a == "" {
				a = m.Name
			}
			mode := "auto"
			if allow, ok := cfg.overrideRule(g.Name, m.Name); ok {
				if allow {
					mode = "allow"
				} else {
					mode = "deny"
				}
			}
			_, protocol, admitted := cfg.modelDecision(g, m)
			t, e := compatBindingFromConfig(cfg, a)
			enabled := admitted && compatUsable(g.keyBinding) && e == nil && t.Channel == g.Name && t.Name == m.Name && !nativeDeclared(cfg, a)
			rows = append(rows, panelRow{g.Name, m.Name, m.Alias, embeddingModelAllowed(m), mode, enabled, protocol})
		}
	}
	b, _ := json.Marshal(map[string]any{"models": rows, "revision": rev})
	return b
}
func panelError(s int) (int, []byte) { return s, []byte(`{"error":"model selection unavailable"}`) }
func panelModels(method string, body []byte) (int, []byte) {
	if method != "GET" && method != "POST" {
		return panelError(405)
	}
	panelMu.Lock()
	defer panelMu.Unlock()
	cfg, rev, rules, e := decisionSnapshot()
	if e != nil {
		return panelError(503)
	}
	if method == "GET" {
		return 200, panelListing(cfg, rev)
	}
	if len(body) > 4096 {
		return panelError(413)
	}
	var in struct{ Channel, Name, Mode, Revision string }
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if d.Decode(&in) != nil || d.Decode(new(any)) != io.EOF || !modeOK(in.Mode) || in.Channel == "" || in.Name == "" || in.Revision == "" {
		return panelError(400)
	}
	if in.Revision != rev {
		return panelError(409)
	}
	matches := 0
	for _, g := range cfg.API.Compat {
		if g.Name == in.Channel {
			for _, m := range g.Models {
				if m.Name == in.Name {
					matches++
				}
			}
		}
	}
	if matches != 1 {
		return panelError(400)
	}
	if rules[in.Channel] == nil {
		rules[in.Channel] = map[string]string{}
	}
	rules[in.Channel][in.Name] = in.Mode
	b, e := json.Marshal(rules)
	if e != nil || len(b) > 1<<20 {
		return panelError(413)
	}
	p := rulesPath()
	tmp, e := os.CreateTemp(filepath.Dir(p), ".models-*")
	if e != nil {
		return panelError(503)
	}
	defer os.Remove(tmp.Name())
	_, e = tmp.Write(b)
	if e == nil {
		e = tmp.Sync()
	}
	ce := tmp.Close()
	if e != nil || ce != nil {
		return panelError(503)
	}
	if os.Rename(tmp.Name(), p) != nil {
		return panelError(503)
	}
	cfg, rev, _, e = decisionSnapshot()
	if e != nil {
		return panelError(503)
	}
	return 200, panelListing(cfg, rev)
}
