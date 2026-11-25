package main

import "net/http"

// 所有带标记请求在这里终结；拒绝未登记模型和跨协议重名，不尝试聊天执行器。
func dispatchEmbedding(model string, body []byte) ([]byte, int, error) {
	raw, err := readConfigBytes()
	if err != nil {
		return nil, 502, errCompatBinding
	}
	native := geminiModelName(model) && geminiDeclared(raw, model)
	target, err := compatBindingFrom(raw, model)
	if native && err != nil {
		return runEmbedding(model, body)
	}
	if !native && err == nil {
		return compatWithTarget(model, body, target, &http.Transport{DisableKeepAlives: true})
	}
	return nil, 400, errCompatBad
}
