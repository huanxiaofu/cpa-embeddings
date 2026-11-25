package main

import "net/http"

func nativeDeclared(cfg *catalogConfig, model string) bool {
	if !geminiModelName(model) {
		return false
	}
	for _, g := range cfg.API.Gemini {
		if nativeGeminiBase(g.Base) {
			for _, m := range g.Models {
				if m.Name == model && (m.Alias == "" || m.Alias == model) {
					return true
				}
			}
		}
	}
	return false
}

// 入口与执行器使用同一判定；即使在两者之间撤销了型号，也不会回退为聊天请求。
func resolveCatalog(cfg *catalogConfig, model string) (bool, compatTarget, bool) {
	native := nativeDeclared(cfg, model)
	target, e := compatBindingFromConfig(cfg, model)
	if native {
		_, ne := geminiBinding(cfg.API.Gemini, model)
		return true, compatTarget{}, ne == nil && e != nil
	}
	return false, target, e == nil
}
func dispatchEmbedding(model string, body []byte) ([]byte, int, error) {
	cfg, _, e := activeCatalog()
	if e != nil {
		return nil, 502, errCompatBinding
	}
	native, target, ok := resolveCatalog(cfg, model)
	if !ok {
		return nil, 400, errCompatBad
	}
	if native {
		return runEmbedding(model, body)
	}
	return compatWithTarget(model, body, target, &http.Transport{DisableKeepAlives: true})
}
