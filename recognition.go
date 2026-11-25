package main

import "strings"

// 向量模型识别核心。
//
// 识别输入只有 CPA 里登记的真实模型名（models[].name）和显式 output-modalities；
// alias 只用于客户端寻址，不能反过来证明某个模型是向量模型。组名后缀也不再
// 授权整组：混合组里的聊天、重排、图像生成模型必须逐个被拒绝。
//
// 人工规则来自插件实例配置，按“渠道组名 -> 真实模型名 -> allow|deny”覆盖识别：
//
//	plugins:
//	  configs:
//	    cpr-gemini-embeddings:
//	      model-overrides:
//	        渠道组名:
//	          真实模型名: allow   # 或 deny
//
// 缺失即按识别结果自动裁决。人工规则只放宽或收紧模型识别，不越过凭据、地址、
// 代理、同名歧义等安全约束，那些仍由绑定解析执行。

// 显式非向量输出模态；出现时优先拒绝，避免同时声明 embedding 与 chat 的条目
// 被当作向量模型。
var nonVectorModalities = map[string]bool{
	"chat": true, "text": true, "completion": true, "completions": true,
	"response": true, "responses": true, "rerank": true, "ranking": true,
	"image": true, "images": true, "image_generation": true, "image-generation": true,
	"audio": true, "speech": true, "video": true, "moderation": true,
}

var vectorModalities = map[string]bool{
	"embedding": true, "embeddings": true, "embed": true,
}

// 明确不是向量模型的名称特征，优先于任何家族前缀。
var nonEmbeddingMarkers = []string{
	"rerank", "re-rank", "reranker", "cross-encoder", "crossencoder", "ranker",
	"chat", "completion", "moderation", "transcri", "whisper", "tts", "speech",
	"text-to-image", "image-generation", "imagegen", "diffusion", "dall-e",
	"dalle", "imagen", "sdxl", "midjourney", "flux", "generate", "veo", "sora",
}

// 各厂商公开向量模型族的固定名称前缀（大小写不敏感，先剥离 vendor/ 路径）。
var embeddingNamePrefixes = []string{
	"text-embedding-",        // OpenAI、DashScope、Gemini 文本向量
	"gemini-embedding-",      // Google 原生向量族
	"embed-",                 // Cohere 官方命名（embed-english-v3.0 等）
	"cohere-embed",           // Cohere 兼容/代理命名
	"bge-",                   // BAAI BGE 系列（bge-m3、bge-large-zh 等）
	"mistral-embed",          // Mistral
	"jina-embeddings-",       // Jina 文本向量
	"jina-clip-",             // Jina 多模态向量（支持文本输入）
	"voyage-",                // Voyage AI
	"nomic-embed",            // Nomic
	"gte-",                   // GTE 系列
	"e5-",                    // 微软 E5 系列
	"multilingual-e5",        // 多语言 E5
	"snowflake-arctic-embed", // Snowflake
	"mxbai-embed",            // MixedBread
	"tongyi-embedding",       // 通义向量
}

// 少数以精确名称登记的向量型号。
var embeddingExactNames = map[string]bool{
	"ada-002": true, // OpenAI 旧型号常以裸名登记
}

// embeddingModelName 只根据真实模型名判断是否属于向量族。没有命中任何固定家族
// 前缀的名称一律不识别，不用模糊关键词放行。
func embeddingModelName(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" || len(n) > 128 {
		return false
	}
	// 只接受单段模型标识符；空白、路径穿越与 URL 形态一律不识别。
	if strings.ContainsAny(n, " \t\r\n") || strings.Contains(n, "..") ||
		strings.Contains(n, "://") || strings.HasPrefix(n, "/") || strings.HasSuffix(n, "/") {
		return false
	}
	for _, bad := range nonEmbeddingMarkers {
		if strings.Contains(n, bad) {
			return false
		}
	}
	// 图像类名称只有同时带 embed 段时才可能是多模态向量模型。
	if strings.Contains(n, "image") && !strings.Contains(n, "embed") {
		return false
	}
	leaf := n
	if i := strings.LastIndex(leaf, "/"); i >= 0 {
		leaf = leaf[i+1:]
	}
	if leaf == "" {
		return false
	}
	if embeddingExactNames[leaf] {
		return true
	}
	for _, p := range embeddingNamePrefixes {
		if strings.HasPrefix(leaf, p) || strings.HasPrefix(n, p) {
			return true
		}
	}
	// Qwen 家族同时有聊天与向量型号，只有名称里带 embedding 段的才是向量模型
	// （qwen3-embedding-8b、qwen3-vl-embedding-8b 等）。
	return strings.HasPrefix(leaf, "qwen") && strings.Contains(leaf, "embedding")
}

// normalizeModality 归一化 output-modalities 里的单个取值。
func normalizeModality(raw string) string {
	m := strings.ToLower(strings.TrimSpace(raw))
	m = strings.ReplaceAll(m, "_", "-")
	m = strings.ReplaceAll(m, " ", "-")
	switch {
	case vectorModalities[m]:
		return "embedding"
	case nonVectorModalities[m] || nonVectorModalities[strings.ReplaceAll(m, "-", "_")]:
		return "non-vector"
	}
	return "unknown"
}

// declaredEmbedding 依据显式输出模态裁决。ok=false 表示没有可用的显式声明，需要
// 回落到名称识别；ok=true 时 allow 就是结论，显式非向量或无法识别的模态优先拒绝。
func declaredEmbedding(modalities []string) (allow bool, ok bool) {
	decided, vector := false, false
	for _, raw := range modalities {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		decided = true
		switch normalizeModality(raw) {
		case "embedding":
			vector = true
		default:
			return false, true
		}
	}
	return vector, decided
}

// embeddingModelAllowed 是自动裁决：显式模态优先，其次按真实名称识别。
func embeddingModelAllowed(m modelBinding) bool {
	if allow, ok := declaredEmbedding(m.Modality); ok {
		return allow
	}
	return embeddingModelName(m.Name)
}

// overrideRule 读取人工规则 plugins.configs.<插件>.model-overrides[channel][真实模型名]。
// 只有明确的 allow/deny 生效；取值写错时按拒绝处理，不能静默放行。
func (c channelOptions) overrideRule(channel, model string) (allow bool, ok bool) {
	if channel == "" || model == "" {
		return false, false
	}
	v, exists := c.Plugins.Configs[abiPluginID].ModelOverrides[channel][model]
	if !exists {
		return false, false
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "allow":
		return true, true
	case "deny":
		return false, true
	}
	return false, true
}

// modelDecision 汇总单个模型条目的最终裁决，供 edge/dispatch/管理面板复用。
//
//	auto    表示没有人工规则介入，结论完全来自识别
//	mode    是最终协议：显式合法映射优先，原生 Gemini base 记为 gemini，
//	        Cohere 官方命名记为 cohere，其余默认 openai；显式非法映射为空串
//	enabled 表示是否允许该真实模型作为向量模型对外提供
//
// 人工 allow/deny 只作用于模型识别；协议无法确定时仍拒绝，凭据、地址与代理策略
// 等安全约束继续由绑定解析执行。
func (c channelOptions) modelDecision(g groupBinding, m modelBinding) (auto bool, mode string, enabled bool) {
	mode = c.protocol(g.Name)
	if mode == "openai" && nativeGeminiBase(g.Base) {
		mode = "gemini"
	}
	allow, hasRule := c.overrideRule(g.Name, m.Name)
	if !hasRule {
		return true, mode, mode != "" && embeddingModelAllowed(m)
	}
	return false, mode, mode != "" && allow
}

// nativeGeminiBase 判断组地址是否就是 Google 原生生成式端点。
func nativeGeminiBase(base string) bool {
	return strings.TrimRight(strings.TrimSpace(base), "/") == googleBase
}
