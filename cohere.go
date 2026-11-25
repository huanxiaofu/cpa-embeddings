package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"
)

// Cohere v2 Embed 文本接口。官方契约：
//   - POST https://api.cohere.com/v2/embed
//   - https://docs.cohere.com/reference/embed
//   - 模型维度表：https://docs.cohere.com/docs/cohere-embed
//
// 请求只发送 texts、input_type、embedding_types:["float"] 和按规范裁剪的
// output_dimension；响应 embeddings.float[][] 由本插件统一输出为 OpenAI 形状，
// 用量取 meta.billed_units.input_tokens 的真实计数，不做字符估算。
const coherePath = "/embed"

// cohereInputTypes 是允许转发给该上游的 input_type。文档里的 "image" 属于图文
// 输入，本插件只处理文本，因此不接收；其他值由普通上游拒绝，也不能透传。
var cohereInputTypes = map[string]bool{
	"search_document": true,
	"search_query":    true,
	"classification":  true,
	"clustering":      true,
}

// cohereDimSpec 来自官方维度表：Fixed 大于 0 表示模型维度固定，此时上游不接受
// output_dimension；否则 Allowed 列出可调维度的合法取值。
type cohereDimSpec struct {
	Fixed   int
	Allowed []int
}

var cohereDims = map[string]cohereDimSpec{
	"embed-v4.0":                    {Allowed: []int{256, 512, 1024, 1536}},
	"embed-v5.0-pro":                {Allowed: []int{256, 512, 768, 1024, 1536, 2048}},
	"embed-v5.0-fast":               {Allowed: []int{256, 512, 768, 1024, 1536, 2048}},
	"embed-english-v3.0":            {Fixed: 1024},
	"embed-multilingual-v3.0":       {Fixed: 1024},
	"embed-english-light-v3.0":      {Fixed: 384},
	"embed-multilingual-light-v3.0": {Fixed: 384},
	"embed-english-v2.0":            {Fixed: 4096},
	"embed-multilingual-v2.0":       {Fixed: 768},
	"embed-english-light-v2.0":      {Fixed: 1024},
	"embed-multilingual-light-v2.0": {Fixed: 384},
}

// cohereOutputDimension 决定是否向上游发送 output_dimension：
// 返回 0 表示省略。模型未登记或取值不符合官方规范时 ok=false，由调用方拒绝请求，
// 避免上游忽略维度后返回与客户端预期不一致的向量。
func cohereOutputDimension(model string, dim int) (int, bool) {
	spec, known := cohereDims[model]
	if !known || dim <= 0 {
		return 0, false
	}
	if spec.Fixed > 0 {
		// 固定维度模型只允许与模型维度一致的请求，并且必须省略该参数。
		return 0, dim == spec.Fixed
	}
	for _, v := range spec.Allowed {
		if v == dim {
			return dim, true
		}
	}
	return 0, false
}

// cohereRun 执行一次 Cohere v2 Embed 调用。target 由 compatBinding 解析得到，
// 客户端模型名（alias 或 name）仅用于回写响应；base 必须是 HTTPS，凭据只放在
// Authorization 头里，重定向和上游错误正文都不透传。
func cohereRun(model string, body []byte, target compatTarget, tr http.RoundTripper) ([]byte, int, error) {
	bad, invalid := errCompatBad, errCompatInvalid
	if len(body) > 6*embMaxInput+4096 {
		return nil, 413, bad
	}
	var in map[string]json.RawMessage
	if !utf8.Valid(body) || json.Unmarshal(body, &in) != nil {
		return nil, 400, bad
	}
	// 白名单：只接受 OpenAI 向量请求里与 Cohere 语义一致的字段。客户端自带的
	// URL、认证头、truncate 等一律拒绝，不猜测上游支持范围。
	for k := range in {
		switch k {
		case "model", "input", "encoding_format", "dimensions", "input_type", "stream":
		default:
			return nil, 400, bad
		}
	}
	if b := in["model"]; b != nil {
		var m string
		if json.Unmarshal(b, &m) != nil || m != model {
			return nil, 400, bad
		}
	}
	if b := in["stream"]; b != nil && string(b) != "false" {
		return nil, 400, bad
	}
	// 上游固定索取 float；客户端要 base64 时由本插件按 OpenAI 约定编码。
	format := "float"
	if b := in["encoding_format"]; b != nil {
		if string(b) == "null" || json.Unmarshal(b, &format) != nil || format != "float" && format != "base64" {
			return nil, 400, bad
		}
	}
	// 检索入库默认 search_document；查询侧可显式选择 search_query。
	inputType := "search_document"
	if b := in["input_type"]; b != nil {
		if string(b) == "null" || json.Unmarshal(b, &inputType) != nil || !cohereInputTypes[inputType] {
			return nil, 400, bad
		}
	}
	dim := 0
	if b := in["dimensions"]; b != nil {
		if json.Unmarshal(b, &dim) != nil || dim <= 0 {
			return nil, 400, bad
		}
	}
	input := in["input"]
	var texts []string
	if len(input) > 0 && input[0] == '"' {
		var s string
		if json.Unmarshal(input, &s) != nil {
			return nil, 400, bad
		}
		texts = []string{s}
	} else if json.Unmarshal(input, &texts) != nil {
		return nil, 400, bad
	}
	if len(texts) < 1 || len(texts) > 32 {
		return nil, 400, bad
	}
	size := 0
	for _, s := range texts {
		if s == "" {
			return nil, 400, bad
		}
		size += len(s)
	}
	if size > embMaxInput {
		return nil, 413, bad
	}
	if target.Name == "" || target.Key == "" {
		return nil, 502, errCompatBinding
	}
	base := strings.TrimRight(strings.TrimSpace(target.Base), "/")
	u, err := url.Parse(base)
	if base == "" || err != nil || u.Scheme != "https" || u.Host == "" ||
		u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
		return nil, 502, errCompatBinding
	}
	// 维度按上游真实模型名核对官方维度表；客户端 alias 不参与判断，避免别名
	// 让不合规的 dimensions 混过校验后与返回宽度不一致。
	sendDim := 0
	if dim > 0 {
		var ok bool
		if sendDim, ok = cohereOutputDimension(target.Name, dim); !ok {
			return nil, 400, bad
		}
	}
	req := struct {
		Model           string   `json:"model"`
		Texts           []string `json:"texts"`
		InputType       string   `json:"input_type"`
		EmbeddingTypes  []string `json:"embedding_types"`
		OutputDimension int      `json:"output_dimension,omitempty"`
	}{target.Name, texts, inputType, []string{"float"}, sendDim}
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, 500, bad
	}
	if tr == nil {
		tr = &http.Transport{DisableKeepAlives: true}
	}
	c := http.Client{Transport: tr, Timeout: 25 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	r, err := http.NewRequest(http.MethodPost, base+coherePath, bytes.NewReader(payload))
	if err != nil {
		return nil, 502, invalid
	}
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+target.Key)
	resp, err := c.Do(r)
	if err != nil {
		return nil, 502, invalid
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		status := resp.StatusCode
		if status < 400 || status > 599 {
			status = 502
		}
		return nil, status, invalid
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, compatLimit+1))
	if err != nil || len(raw) > compatLimit {
		return nil, 502, invalid
	}
	var rep struct {
		Embeddings struct {
			// 用指针逐项判空；1e100 这类超出 float32 的值在解码阶段即失败。
			Float [][]*float32 `json:"float"`
		} `json:"embeddings"`
		Meta struct {
			BilledUnits struct {
				InputTokens *int `json:"input_tokens"`
			} `json:"billed_units"`
		} `json:"meta"`
	}
	if json.Unmarshal(raw, &rep) != nil {
		return nil, 502, invalid
	}
	rows := rep.Embeddings.Float
	tokens := rep.Meta.BilledUnits.InputTokens
	// usage 必须是上游真实回报；缺失、零或负数的计数不能当作成功。
	if len(rows) != len(texts) || tokens == nil || *tokens <= 0 {
		return nil, 502, invalid
	}
	// 客户端指定了维度时（含固定维度模型），响应宽度必须一致。
	want := sendDim
	if want == 0 {
		want = dim
	}
	items := make([]embItem, len(rows))
	for i, row := range rows {
		if len(row) == 0 || len(row) > embMaxDim {
			return nil, 502, invalid
		}
		if want == 0 {
			want = len(row)
		} else if len(row) != want {
			return nil, 502, invalid
		}
		vec := make([]float32, len(row))
		for j, p := range row {
			// null、NaN 和 ±Inf 都不是可用向量。
			if p == nil || math.IsNaN(float64(*p)) || math.IsInf(float64(*p), 0) {
				return nil, 502, invalid
			}
			vec[j] = *p
		}
		var value any = vec
		if format == "base64" {
			buf := make([]byte, 4*len(vec))
			for j, f := range vec {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(f))
			}
			value = base64.StdEncoding.EncodeToString(buf)
		}
		items[i] = embItem{Object: "embedding", Index: i, Embedding: value}
	}
	// 响应 model 保留客户端使用的 alias（或 name），usage 两项都取真实计费输入数。
	out, err := json.Marshal(map[string]any{
		"object": "list",
		"model":  model,
		"data":   items,
		"usage":  map[string]int{"prompt_tokens": *tokens, "total_tokens": *tokens},
	})
	if err != nil {
		return nil, 500, invalid
	}
	return out, 200, nil
}
