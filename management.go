package main

import (
	_ "embed"
	"encoding/json"
)

// 资源页只提供静态 HTML；读写模型选择分别经过 CPA 自带的管理鉴权。
//
//go:embed deploy/panel.html
var panelHTML []byte

const resourceRoot = "/v0/resource/plugins/" + abiPluginID
const rulesAPI = "/v0/management/cpa-embeddings/models"

func managementRPC(method string, raw []byte) abiEnvelope {
	if method == "management.register" {
		return abiOK(map[string]any{"resources": []map[string]string{{"Path": "/panel", "Menu": "向量模型选择", "Description": "混合渠道的自动识别与逐模型覆盖"}, {"Path": "/eligible"}}, "routes": []map[string]string{{"Method": "GET", "Path": rulesAPI}, {"Method": "POST", "Path": rulesAPI}}})
	}
	var req struct {
		Method, Path string
		Query        map[string][]string
		Body         []byte
	}
	if json.Unmarshal(raw, &req) != nil {
		return abiFail("invalid_request", "invalid request", 400)
	}
	status := 404
	mime := "application/json"
	var out []byte
	if !abiDisabled.Load() {
		switch req.Path {
		case resourceRoot + "/panel":
			if req.Method == "GET" {
				status, mime, out = 200, "text/html; charset=utf-8", panelHTML
			}
		case resourceRoot + "/eligible":
			if req.Method == "GET" {
				status = 403
				if ms := req.Query["model"]; len(ms) == 1 && safeAlias.MatchString(ms[0]) {
					cfg, _, e := activeCatalog()
					if e != nil {
						status = 503
					} else if _, _, ok := resolveCatalog(cfg, ms[0]); ok {
						status = 200
						out = []byte("eligible-v3")
					}
				}
			}
		case rulesAPI:
			status, out = panelModels(req.Method, req.Body)
		}
	}
	return abiOK(map[string]any{"StatusCode": status, "Body": out, "Headers": map[string][]string{"Content-Type": {mime}, "Cache-Control": {"no-store"}, "X-Content-Type-Options": {"nosniff"}, "Referrer-Policy": {"no-referrer"}}})
}
