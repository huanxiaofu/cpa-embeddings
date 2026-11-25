package main

import (
	_ "embed"
	"errors"
	"fmt"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
)

//go:embed deploy/openresty.conf.in
var edgeTemplate string
var safeAlias = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_./:-]{0,254}$`)
var safeHost = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9.-]*$`)

func readConfigBytes() ([]byte, error) { return os.ReadFile(compatConfigPath()) }

// 一次性生成站点配置。模型选择由运行中的插件提供，不把模型或密钥烘焙进代理。
func renderEdge(raw []byte, upstream string) (string, error) {
	host, port, err := net.SplitHostPort(upstream)
	num, e := strconv.Atoi(port)
	if err != nil || e != nil || strconv.Itoa(num) != port || num < 1 || num > 65535 || (!safeHost.MatchString(host) && net.ParseIP(host) == nil) {
		return "", errors.New("invalid CPA host:port")
	}
	if _, err := parseCatalog(raw); err != nil {
		return "", err
	}
	return strings.ReplaceAll(edgeTemplate, "CPA_UPSTREAM", upstream), nil
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
