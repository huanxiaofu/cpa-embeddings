package main

import (
	_ "embed"
	"errors"
	"fmt"
	"gopkg.in/yaml.v3"
	"net"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

//go:embed deploy/openresty.conf.in
var edgeTemplate string
var safeAlias = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_./:-]{0,254}$`)
var safeHost = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9.-]*$`)

func readConfigBytes() ([]byte, error) { return os.ReadFile(compatConfigPath()) }

// 由管理员从同一份 CPA 配置生成入口白名单；清单只含公开别名，不含凭据或上游地址。
func renderEdge(raw []byte, upstream string) (string, error) {
	host, port, err := net.SplitHostPort(upstream)
	num, e := strconv.Atoi(port)
	if err != nil || e != nil || strconv.Itoa(num) != port || num < 1 || num > 65535 || (!safeHost.MatchString(host) && net.ParseIP(host) == nil) {
		return "", errors.New("invalid CPA host:port")
	}
	var cfg struct {
		channelOptions `yaml:",inline"`
		API            struct {
			Gemini []groupBinding `yaml:"gemini"`
			Compat []groupBinding `yaml:"openai-compatibility"`
		} `yaml:"api-keys"`
	}
	if yaml.Unmarshal(raw, &cfg) != nil {
		return "", errors.New("invalid CPA configuration")
	}
	names := map[string]bool{}
	for _, g := range cfg.API.Gemini {
		for _, m := range g.Models {
			if geminiModelName(m.Name) {
				if _, err := liveBinding(raw, m.Name); err == nil {
					names[m.Name] = true
				}
			}
		}
	}
	for _, g := range cfg.API.Compat {
		if cfg.protocol(g.Name) == "" {
			continue
		}
		for _, m := range g.Models {
			name := m.Alias
			if name == "" {
				name = m.Name
			}
			if _, err := compatBindingFrom(raw, name); err != nil {
				continue
			}
			if geminiModelName(name) && geminiDeclared(raw, name) {
				return "", errors.New("native and compatible model alias conflict")
			}
			names[name] = true
		}
	}
	list := make([]string, 0, len(names))
	for n := range names {
		if !safeAlias.MatchString(n) {
			return "", errors.New("invalid public embedding model alias")
		}
		list = append(list, fmt.Sprintf("[%q]=true", n))
	}
	sort.Strings(list)
	// 空清单合法：最后一个模型撤销后可生成一个拒绝全部请求的入口。
	return strings.NewReplacer("-- CPA_EMBED_MODELS", strings.Join(list, ","), "CPA_UPSTREAM", upstream).Replace(edgeTemplate), nil
}
func edgeCLI() {
	if len(os.Args) != 4 || os.Args[1] != "render-edge" {
		fmt.Fprintln(os.Stderr, "usage: cpa-embeddings-config render-edge CONFIG.yaml CPA_HOST:PORT")
		os.Exit(2)
	}
	raw, e := os.ReadFile(os.Args[2])
	if e != nil {
		fmt.Fprintln(os.Stderr, "cannot read CPA configuration")
		os.Exit(1)
	}
	s, e := renderEdge(raw, os.Args[3])
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	fmt.Print(s)
}
