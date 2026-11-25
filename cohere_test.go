package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"testing"
)

const cohereTestBase = "https://api.cohere.com/v2"

// 使用带字段名的字面量，后续给 compatTarget 增加 Protocol 字段时测试仍可编译。
func cohereTestTarget() compatTarget {
	return compatTarget{Name: "embed-v4.0", Base: cohereTestBase, Key: "unit-cohere"}
}

type cohereTransport func(*http.Request) (*http.Response, error)

func (f cohereTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// cohereRow 生成 width 个 0.5 的 float 向量。
func cohereRow(width int) string {
	return "[" + strings.TrimSuffix(strings.Repeat("0.5,", width), ",") + "]"
}

// cohereReply 生成 rows 行等宽向量，tail 用于拼 meta、附加字段或制造畸形响应。
func cohereReply(rows, width int, tail string) string {
	list := make([]string, rows)
	for i := range list {
		list[i] = cohereRow(width)
	}
	return `{"embeddings":{"float":[` + strings.Join(list, ",") + `]}` + tail + `}`
}

const cohereMetaOK = `,"meta":{"billed_units":{"input_tokens":7}}`

// cohereStub 返回固定响应并记录调用次数；wire 非 nil 时解析上游请求体。
func cohereStub(t *testing.T, status int, reply string, calls *int, wire *map[string]any) http.RoundTripper {
	t.Helper()
	return cohereTransport(func(r *http.Request) (*http.Response, error) {
		*calls = *calls + 1
		if wire != nil {
			*wire = nil
			if json.NewDecoder(r.Body).Decode(wire) != nil {
				t.Fatal("上游请求体无法解码")
			}
		}
		return &http.Response{
			StatusCode: status,
			Header:     http.Header{"Location": {"https://evil.test/x"}},
			Body:       io.NopCloser(strings.NewReader(reply)),
		}, nil
	})
}

// 单条文本：URL、Bearer、白名单字段和真实 usage 都必须与 Cohere v2 契约一致。
func TestCohereRequestShape(t *testing.T) {
	wire := map[string]any{}
	calls := 0
	tr := cohereTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		if r.Method != http.MethodPost || r.URL.String() != cohereTestBase+"/embed" {
			t.Errorf("非预期上游请求: %s %s", r.Method, r.URL)
		}
		if r.Header.Get("Authorization") != "Bearer unit-cohere" || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("凭据或内容类型错误: %v", r.Header)
		}
		wire = nil
		if json.NewDecoder(r.Body).Decode(&wire) != nil {
			t.Fatal("上游请求体无法解码")
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(cohereReply(1, 2, cohereMetaOK)))}, nil
	})
	out, s, e := cohereRun("kb-cohere", []byte(`{"input":"hello"}`), cohereTestTarget(), tr)
	if s != 200 || e != nil || calls != 1 {
		t.Fatalf("状态=%d 错误=%v 调用=%d", s, e, calls)
	}
	if len(wire) != 4 || wire["model"] != "embed-v4.0" || wire["input_type"] != "search_document" ||
		wire["embedding_types"].([]any)[0] != "float" || wire["texts"].([]any)[0] != "hello" {
		t.Fatalf("请求字段未按契约重建: %v", wire)
	}
	if _, ok := wire["output_dimension"]; ok {
		t.Fatal("未请求维度时不应发送 output_dimension")
	}
	var rep struct {
		Object string `json:"object"`
		Model  string `json:"model"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			Prompt int `json:"prompt_tokens"`
			Total  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if json.Unmarshal(out, &rep) != nil || rep.Object != "list" || rep.Model != "kb-cohere" ||
		len(rep.Data) != 1 || rep.Data[0].Object != "embedding" || rep.Data[0].Index != 0 ||
		len(rep.Data[0].Embedding) != 2 || rep.Usage.Prompt != 7 || rep.Usage.Total != 7 {
		t.Fatalf("响应形状不符合 OpenAI 约定: %s", out)
	}
}

// 批量文本按顺序返回；encoding_format=base64 由本插件编码，上游仍只索取 float。
func TestCohereBatchAndBase64(t *testing.T) {
	wire := map[string]any{}
	calls := 0
	reply := `{"embeddings":{"float":[[1.5,-2],[0.25,0.75]]}` + cohereMetaOK + `}`
	tr := cohereStub(t, 200, reply, &calls, &wire)
	out, s, e := cohereRun("kb-cohere", []byte(`{"input":["first","second"],"encoding_format":"base64"}`), cohereTestTarget(), tr)
	if s != 200 || e != nil {
		t.Fatalf("状态=%d 错误=%v", s, e)
	}
	texts := wire["texts"].([]any)
	if len(texts) != 2 || texts[0] != "first" || texts[1] != "second" {
		t.Fatalf("批量文本顺序被改写: %v", texts)
	}
	if wire["embedding_types"].([]any)[0] != "float" || wire["encoding_format"] != nil {
		t.Fatalf("上游编码格式不应由客户端决定: %v", wire)
	}
	var rep struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if json.Unmarshal(out, &rep) != nil || len(rep.Data) != 2 {
		t.Fatalf("无法解析 base64 响应: %s", out)
	}
	want := [][]float32{{1.5, -2}, {0.25, 0.75}}
	for i, d := range rep.Data {
		buf, err := base64.StdEncoding.DecodeString(d.Embedding)
		if err != nil || len(buf) != 4*len(want[i]) {
			t.Fatalf("第 %d 项 base64 非法: %v", i, d.Embedding)
		}
		for j := range want[i] {
			got := math.Float32frombits(binary.LittleEndian.Uint32(buf[4*j:]))
			if got != want[i][j] {
				t.Fatalf("第 %d 项第 %d 位=%v want %v", i, j, got, want[i][j])
			}
		}
	}
}

// dimensions 只有符合官方维度表时才映射为 output_dimension：可调维度显式发送，
// 固定维度模型仅在请求值等于模型维度时省略该参数，其余一律拒绝且不访问上游。
func TestCohereDimensions(t *testing.T) {
	for _, tc := range []struct {
		label   string
		model   string
		dim     int
		sendDim int
		width   int
		ok      bool
	}{
		{"v4 可调维度", "embed-v4.0", 512, 512, 512, true},
		{"v5 可调维度", "embed-v5.0-fast", 768, 768, 768, true},
		{"v4 非法维度", "embed-v4.0", 333, 0, 0, false},
		{"v5 非法维度", "embed-v5.0-pro", 300, 0, 0, false},
		{"v3 固定维度省略参数", "embed-english-v3.0", 1024, 0, 1024, true},
		{"v3 与模型维度不符", "embed-english-v3.0", 512, 0, 0, false},
		{"light v3 固定维度", "embed-multilingual-light-v3.0", 384, 0, 384, true},
		{"v2 固定维度", "embed-english-v2.0", 4096, 0, 4096, true},
		{"未登记模型带维度", "embed-custom", 2, 0, 0, false},
	} {
		t.Run(tc.label, func(t *testing.T) {
			wire := map[string]any{}
			calls := 0
			target := compatTarget{Name: tc.model, Base: cohereTestBase, Key: "unit-cohere"}
			tr := cohereStub(t, 200, cohereReply(1, tc.width, cohereMetaOK), &calls, &wire)
			body := []byte(`{"input":"hello","dimensions":` + strconv.Itoa(tc.dim) + `}`)
			out, s, e := cohereRun("kb-cohere", body, target, tr)
			if !tc.ok {
				if s != 400 || e == nil || calls != 0 || out != nil {
					t.Fatalf("非法维度被接受: 状态=%d 错误=%v 调用=%d", s, e, calls)
				}
				return
			}
			if s != 200 || e != nil || calls != 1 {
				t.Fatalf("状态=%d 错误=%v 调用=%d", s, e, calls)
			}
			got, has := wire["output_dimension"]
			if tc.sendDim == 0 {
				if has {
					t.Fatalf("固定维度模型不应发送 output_dimension: %v", wire)
				}
			} else if !has || int(got.(float64)) != tc.sendDim {
				t.Fatalf("output_dimension=%v want %d", got, tc.sendDim)
			}
		})
	}
	// 未登记模型不指定维度时按上游默认处理，仅校验响应宽度一致。
	wire := map[string]any{}
	calls := 0
	target := compatTarget{Name: "embed-custom", Base: cohereTestBase, Key: "unit-cohere"}
	tr := cohereStub(t, 200, cohereReply(1, 2, cohereMetaOK), &calls, &wire)
	if _, s, e := cohereRun("kb-cohere", []byte(`{"input":"hello"}`), target, tr); s != 200 || e != nil {
		t.Fatalf("未登记模型不带维度应放行: 状态=%d 错误=%v", s, e)
	}
	if _, has := wire["output_dimension"]; has {
		t.Fatal("未登记模型不应推断维度")
	}
}

// input_type 只接受文档里的文本取值；客户端自带的认证、URL、stream 等字段不透传。
func TestCohereInputTypeAndWhitelist(t *testing.T) {
	wire := map[string]any{}
	calls := 0
	tr := cohereStub(t, 200, cohereReply(1, 2, cohereMetaOK), &calls, &wire)
	if _, s, e := cohereRun("kb-cohere", []byte(`{"input":"q","input_type":"search_query"}`), cohereTestTarget(), tr); s != 200 || e != nil {
		t.Fatalf("search_query 被拒绝: 状态=%d 错误=%v", s, e)
	}
	if wire["input_type"] != "search_query" {
		t.Fatalf("input_type 未转发: %v", wire)
	}
	// stream:false 属于客户端常规附带字段，可以接受；其余未知字段一律拒绝。
	for _, b := range []string{
		`{"input":"q","stream":false}`,
	} {
		before := calls
		if _, s, _ := cohereRun("kb-cohere", []byte(b), cohereTestTarget(), tr); s != 200 || calls != before+1 {
			t.Errorf("可接受字段被拒绝: %s", b)
		}
	}
	for _, b := range []string{
		`{"input":"q","input_type":"image"}`,
		`{"input":"q","input_type":"SearchQuery"}`,
		`{"input":"q","input_type":"SEMANTIC_SIMILARITY"}`,
		`{"input":"q","input_type":null}`,
		`{"input":"q","input_type":1}`,
		`{"input":"q","truncate":"END"}`,
		`{"input":"q","stream":true}`,
		`{"input":"q","url":"https://evil.test"}`,
		`{"input":"q","api-key":"leak"}`,
		`{"input":"q","user":"u1"}`,
		`{"input":"q","encoding_format":"int8"}`,
		`{"input":"q","encoding_format":null}`,
		`{"input":"q","model":"embed-english-v3.0"}`,
	} {
		before := calls
		if out, s, _ := cohereRun("kb-cohere", []byte(b), cohereTestTarget(), tr); s != 400 || out != nil || calls != before {
			t.Errorf("非法请求未被拦截: %s 状态=%d", b, s)
		}
	}
}

// 上游响应必须给出与文本数一致的 float 行、等宽有限值和真实计费输入数。
func TestCohereInvalidResponse(t *testing.T) {
	for _, tc := range []struct {
		label string
		reply string
	}{
		{"缺少 meta", `{"embeddings":{"float":[[0.5,0.5]]}}`},
		{"缺少 billed_units", `{"embeddings":{"float":[[0.5,0.5]]},"meta":{}}`},
		{"缺少 input_tokens", `{"embeddings":{"float":[[0.5,0.5]]},"meta":{"billed_units":{}}}`},
		{"input_tokens 为零", `{"embeddings":{"float":[[0.5,0.5]]},"meta":{"billed_units":{"input_tokens":0}}}`},
		{"input_tokens 为负", `{"embeddings":{"float":[[0.5,0.5]]},"meta":{"billed_units":{"input_tokens":-3}}}`},
		{"input_tokens 非整数", `{"embeddings":{"float":[[0.5,0.5]]},"meta":{"billed_units":{"input_tokens":4.5}}}`},
		{"行数不足", cohereReply(1, 2, cohereMetaOK)},
		{"空行", `{"embeddings":{"float":[[]]},"meta":{"billed_units":{"input_tokens":7}}}`},
		{"行宽不一致", `{"embeddings":{"float":[[0.5,0.5],[0.5]]},"meta":{"billed_units":{"input_tokens":7}}}`},
		{"null 元素", `{"embeddings":{"float":[[0.5,null]]},"meta":{"billed_units":{"input_tokens":7}}}`},
		{"超出 float32", `{"embeddings":{"float":[[1e100,0.5]]},"meta":{"billed_units":{"input_tokens":7}}}`},
		{"只有 base64", `{"embeddings":{"base64":"AAAA"},"meta":{"billed_units":{"input_tokens":7}}}`},
		{"float 为 null", `{"embeddings":{"float":null},"meta":{"billed_units":{"input_tokens":7}}}`},
		{"尾随数据", cohereReply(1, 2, cohereMetaOK) + `{}`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			calls := 0
			tr := cohereStub(t, 200, tc.reply, &calls, nil)
			body := []byte(`{"input":"hello"}`)
			if tc.label == "行数不足" || tc.label == "行宽不一致" {
				body = []byte(`{"input":["a","b"]}`)
			}
			if out, s, e := cohereRun("kb-cohere", body, cohereTestTarget(), tr); s != 502 || e == nil || out != nil {
				t.Fatalf("无效响应被接受: 状态=%d 错误=%v", s, e)
			}
		})
	}
	// 请求维度与响应宽度不一致：固定维度模型同样要拒绝。
	calls := 0
	tr := cohereStub(t, 200, cohereReply(1, 2, cohereMetaOK), &calls, nil)
	if _, s, e := cohereRun("kb-cohere", []byte(`{"input":"hello","dimensions":1024}`),
		compatTarget{Name: "embed-english-v3.0", Base: cohereTestBase, Key: "unit-cohere"}, tr); s != 502 || e == nil {
		t.Fatalf("维度不一致被接受: 状态=%d 错误=%v", s, e)
	}
}

// 禁止跟随重定向，上游错误正文和状态不能原样外泄。
func TestCohereRedirectAndUpstreamError(t *testing.T) {
	calls := 0
	tr := cohereStub(t, 302, "sensitive-upstream-body", &calls, nil)
	out, s, e := cohereRun("kb-cohere", []byte(`{"input":"x"}`), cohereTestTarget(), tr)
	if calls != 1 || s != 502 || e == nil || out != nil || strings.Contains(e.Error(), "sensitive") {
		t.Fatalf("重定向未被截断: 状态=%d 错误=%v 调用=%d", s, e, calls)
	}
	// 限流状态保留，其余异常统一为 502。
	calls = 0
	tr = cohereStub(t, 429, "rate limited", &calls, nil)
	if out, s, e := cohereRun("kb-cohere", []byte(`{"input":"x"}`), cohereTestTarget(), tr); s != 429 || e == nil || out != nil {
		t.Fatalf("限流状态被改写: 状态=%d 错误=%v", s, e)
	}
	calls = 0
	tr = cohereStub(t, 500, "internal", &calls, nil)
	if _, s, _ := cohereRun("kb-cohere", []byte(`{"input":"x"}`), cohereTestTarget(), tr); s != 500 {
		t.Fatalf("5xx 状态被改写: %d", s)
	}
}

// 输入规模、编码和目标地址在访问上游前就要拦截。
func TestCohereInputLimits(t *testing.T) {
	calls := 0
	tr := cohereStub(t, 200, cohereReply(1, 2, cohereMetaOK), &calls, nil)
	for _, tc := range []struct {
		label string
		body  string
		want  int
	}{
		{"空输入", `{"input":[]}`, 400},
		{"空字符串", `{"input":""}`, 400},
		{"null 输入", `{"input":null}`, 400},
		{"null 元素", `{"input":[null]}`, 400},
		{"非文本输入", `{"input":5}`, 400},
		{"超过 32 条", `{"input":[` + strings.TrimSuffix(strings.Repeat(`"a",`, 33), ",") + `]}`, 400},
		{"单条超限", `{"input":"` + strings.Repeat("x", embMaxInput+1) + `"}`, 413},
		{"请求体超限", `{"input":"` + strings.Repeat("x", 6*embMaxInput+4096) + `"}`, 413},
		{"非法 JSON", `{"input":`, 400},
		{"维度为零", `{"input":"a","dimensions":0}`, 400},
		{"维度为负", `{"input":"a","dimensions":-1}`, 400},
		{"维度为字符串", `{"input":"a","dimensions":"512"}`, 400},
		{"维度为 null", `{"input":"a","dimensions":null}`, 400},
	} {
		if out, s, e := cohereRun("kb-cohere", []byte(tc.body), cohereTestTarget(), tr); s != tc.want || e == nil || out != nil {
			t.Errorf("%s: 状态=%d want %d 错误=%v", tc.label, s, tc.want, e)
		}
	}
	if calls != 0 {
		t.Fatalf("非法输入被发往上游: %d", calls)
	}
	// 目标地址必须为 HTTPS 且不含凭据、查询串；缺失凭据同样不发起调用。
	for _, target := range []compatTarget{
		{Name: "embed-v4.0", Base: "http://api.cohere.com/v2", Key: "unit-cohere"},
		{Name: "embed-v4.0", Base: "https://api.cohere.com/v2?x=1", Key: "unit-cohere"},
		{Name: "embed-v4.0", Base: "https://user:pass@api.cohere.com/v2", Key: "unit-cohere"},
		{Name: "embed-v4.0", Base: "", Key: "unit-cohere"},
		{Name: "", Base: cohereTestBase, Key: "unit-cohere"},
		{Name: "embed-v4.0", Base: cohereTestBase, Key: ""},
	} {
		if out, s, e := cohereRun("kb-cohere", []byte(`{"input":"a"}`), target, tr); s != 502 || e == nil || out != nil {
			t.Errorf("非法目标未被拦截: %+v 状态=%d", target, s)
		}
	}
	if calls != 0 {
		t.Fatal("非法目标发起了上游调用")
	}
}
