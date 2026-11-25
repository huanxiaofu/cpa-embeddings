package main

// 两类上游共用 CPA 执行回执，由宿主按鉴权得到的客户端 Key 计量。
func dispatchEmbedding(model string, body []byte) ([]byte, int, error) {
	if compatModel(model) {
		return runCompatEmbedding(model, body)
	}
	return runEmbedding(model, body)
}
