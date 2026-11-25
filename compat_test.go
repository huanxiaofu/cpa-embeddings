package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const compatTestModel = "BAAI/bge-m3"

// 生产形状的专用组：组名以 -embeddings 结尾，base 尾部多一个斜杠也要归一化。
const compatFixture = "api-keys:\n  openai-compatibility:\n" +
	"    - name: siliconflow-embeddings\n" +
	"      base-url: https://api.siliconflow.cn/v1/\n" +
	"      models: [{name: BAAI/bge-m3, alias: BAAI/bge-m3}]\n" +
	"      keys: [{api-key: unit-silicon, weight: 1}]\n"

func compatGroupYAML(name, base, extra, models, keys string) string {
	return fmt.Sprintf("    - name: %s\n      base-url: %s\n%s      models: %s\n      keys: %s\n", name, base, extra, models, keys)
}

func compatCfg(groups ...string) string {
	return "api-keys:\n  openai-compatibility:\n" + strings.Join(groups, "")
}

func compatConfig(t *testing.T, text string) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "fixture.yaml")
	if e := os.WriteFile(p, []byte(text), 0600); e != nil {
		t.Fatal(e)
	}
	t.Setenv("CPR_EMBED_CONFIG", p)
}

// 允许的绑定：任意 HTTPS 端点、alias 或 name 作客户端名、权重缺省按 1。
func TestCompatBindingAllowed(t *testing.T) {
	for _, tc := range []struct {
		label, group, base, extra, models, keys, model string
		wantBase, wantName, wantKey                    string
	}{
		{"既有 SiliconFlow 生产组", "siliconflow-embeddings", "https://api.siliconflow.cn/v1/", "",
			"[{name: BAAI/bge-m3, alias: BAAI/bge-m3}]", "[{api-key: unit-silicon, weight: 1}]", "BAAI/bge-m3",
			"https://api.siliconflow.cn/v1", "BAAI/bge-m3", "unit-silicon"},
		{"DashScope 兼容端点与别名切换", "dashscope-embeddings", "https://dashscope.aliyuncs.com/compatible-mode/v1", "",
			"[{name: text-embedding-v4, alias: qwen-embed}]", "[{api-key: unit-dashscope}]", "qwen-embed",
			"https://dashscope.aliyuncs.com/compatible-mode/v1", "text-embedding-v4", "unit-dashscope"},
		{"Mistral 只写真实 name", "mistral-embeddings", "https://api.mistral.ai/v1/", "",
			"[{name: mistral-embed}]", "[{api-key: unit-mistral, weight: 2}]", "mistral-embed",
			"https://api.mistral.ai/v1", "mistral-embed", "unit-mistral"},
		{"OpenAI 兼容端点", "openai-embeddings", "https://api.openai.com/v1", "",
			"[{name: text-embedding-3-small, alias: kb-embed}]", "[{api-key: unit-openai, weight: 1}]", "kb-embed",
			"https://api.openai.com/v1", "text-embedding-3-small", "unit-openai"},
		{"私有兼容部署带路径", "private-embeddings", "https://embeddings.internal.test/compat/v1/", "",
			"[{name: bge-m3-private, alias: private-embed}]", "[{api-key: unit-private, weight: 1}]", "private-embed",
			"https://embeddings.internal.test/compat/v1", "bge-m3-private", "unit-private"},
		{"组级单一凭据", "groupkey-embeddings", "https://embeddings.internal.test/v1", "      api-key: unit-group\n",
			"[{name: group-embed}]", "[]", "group-embed",
			"https://embeddings.internal.test/v1", "group-embed", "unit-group"},
	} {
		compatConfig(t, compatCfg(compatGroupYAML(tc.group, tc.base, tc.extra, tc.models, tc.keys)))
		target, err := compatBinding(tc.model)
		if err != nil || target.Base != tc.wantBase || target.Name != tc.wantName || target.Key != tc.wantKey {
			t.Errorf("%s: %v %+v", tc.label, err, target)
		}
		if !compatAllowed(tc.model) {
			t.Errorf("%s: 模型未获准", tc.label)
		}
	}
}

// 拒绝不支持的凭据策略、非 HTTPS 地址以及任何无法唯一确定渠道的同名绑定。
func TestCompatBindingRejected(t *testing.T) {
	const base = "https://api.siliconflow.cn/v1"
	const models = "[{name: BAAI/bge-m3, alias: BAAI/bge-m3}]"
	const keys = "[{api-key: unit-silicon, weight: 1}]"
	for _, tc := range []struct{ label, cfg string }{
		{"组名缺少 -embeddings 后缀", compatCfg(compatGroupYAML("siliconflow", base, "", models, keys))},
		{"组名只有后缀", compatCfg(compatGroupYAML("-embeddings", base, "", models, keys))},
		{"模型只在普通聊天组", compatCfg(compatGroupYAML("chat", base, "", models, keys), compatGroupYAML("chat-embeddings", base, "", "[{name: other-embed}]", keys))},
		{"别名不一致", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", "[{name: BAAI/bge-m3, alias: other-alias}]", keys))},
		{"组停用", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      disabled: true\n", models, keys))},
		{"组权重为 0", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      weight: 0\n", models, keys))},
		{"组权重为负", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      weight: -1\n", models, keys))},
		{"组前缀策略", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      prefix: private\n", models, keys))},
		{"组附加头", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      headers: {X-Test: '1'}\n", models, keys))},
		{"组代理", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      proxy-url: http://127.0.0.1:9\n", models, keys))},
		{"组排除规则", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "      excluded-models: ['*']\n", models, keys))},
		{"凭据停用", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-silicon, disabled: true}]"))},
		{"凭据权重为 0", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-silicon, weight: 0}]"))},
		{"凭据前缀", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-silicon, weight: 1, prefix: private}]"))},
		{"凭据附加头", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-silicon, weight: 1, headers: {X-Test: '1'}}]"))},
		{"凭据代理", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-silicon, weight: 1, proxy-url: http://127.0.0.1:9}]"))},
		{"凭据排除规则", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-silicon, weight: 1, excluded-models: ['*']}]"))},
		{"无凭据", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[]"))},
		{"空 api-key", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: '', weight: 1}]"))},
		{"两个可用凭据", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, "[{api-key: unit-a, weight: 1}, {api-key: unit-b, weight: 1}]"))},
		{"http base", compatCfg(compatGroupYAML("siliconflow-embeddings", "http://api.siliconflow.cn/v1", "", models, keys))},
		{"base 带查询", compatCfg(compatGroupYAML("siliconflow-embeddings", "https://api.siliconflow.cn/v1?x=1", "", models, keys))},
		{"base 带片段", compatCfg(compatGroupYAML("siliconflow-embeddings", "https://api.siliconflow.cn/v1#frag", "", models, keys))},
		{"base 带用户信息", compatCfg(compatGroupYAML("siliconflow-embeddings", "https://user:pass@api.siliconflow.cn/v1", "", models, keys))},
		{"base 为空", compatCfg(compatGroupYAML("siliconflow-embeddings", "''", "", models, keys))},
		{"同名跨组冲突", compatCfg(
			compatGroupYAML("siliconflow-embeddings", base, "", models, keys),
			compatGroupYAML("other-embeddings", "https://api.example.test/v1", "", models, "[{api-key: unit-other, weight: 1}]"),
		)},
		{"组内重复模型", compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", "[{name: BAAI/bge-m3, alias: BAAI/bge-m3}, {name: other-embed, alias: BAAI/bge-m3}]", keys))},
	} {
		compatConfig(t, tc.cfg)
		if compatAllowed(compatTestModel) {
			t.Errorf("%s: 被错误接受", tc.label)
		}
	}
	// 全局代理策略与多文档配置同样拒绝，不退回直连。
	compatConfig(t, "proxy-url: http://127.0.0.1:9\n"+compatCfg(compatGroupYAML("siliconflow-embeddings", base, "", models, keys)))
	if compatAllowed(compatTestModel) {
		t.Error("全局代理策略被忽略")
	}
	compatConfig(t, compatFixture+"---\n"+strings.TrimPrefix(compatFixture, "api-keys:\n  openai-compatibility:\n"))
	if compatAllowed(compatTestModel) {
		t.Error("多文档配置被接受")
	}
}

// 停用或未登记的组不能路由，且不产生任何上游调用。
func TestCompatUnavailable(t *testing.T) {
	for _, cfg := range []string{
		compatCfg(compatGroupYAML("siliconflow-embeddings", "https://api.siliconflow.cn/v1", "      disabled: true\n", "[{name: BAAI/bge-m3, alias: BAAI/bge-m3}]", "[{api-key: unit-silicon, weight: 1}]")),
		compatCfg(compatGroupYAML("siliconflow-embeddings", "https://api.siliconflow.cn/v1", "", "[{name: BAAI/bge-m3, alias: BAAI/bge-m3}]", "[{api-key: unit-silicon, disabled: true}]")),
		compatCfg(compatGroupYAML("chat-upstream", "https://api.siliconflow.cn/v1", "", "[{name: BAAI/bge-m3, alias: BAAI/bge-m3}]", "[{api-key: unit-silicon, weight: 1}]")),
	} {
		compatConfig(t, cfg)
		if compatAllowed(compatTestModel) || compatModel(compatTestModel) {
			t.Fatal("不可用组被路由")
		}
		if _, s, e := runCompatEmbedding(compatTestModel, []byte(`{"input":"text"}`)); s != 502 || e == nil {
			t.Fatalf("不可用组未返回 502: %d %v", s, e)
		}
	}
}

type compatTransport func(*http.Request) (*http.Response, error)

func (f compatTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// 请求侧只重建白名单字段，客户端 URL/认证不向上游透传；redirect 和错误正文不外泄。
func TestCompatExchange(t *testing.T) {
	compatConfig(t, compatFixture)
	const good = `{"data":[{"index":0,"embedding":[1,-2]}],"usage":{"prompt_tokens":3,"total_tokens":4}}`
	reply := good
	status := 200
	calls := 0
	var wire map[string]any
	tr := compatTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		if r.Method != http.MethodPost || r.URL.String() != "https://api.siliconflow.cn/v1/embeddings" || r.Header.Get("Authorization") != "Bearer unit-silicon" {
			t.Fatalf("非预期上游请求: %s %s", r.Method, r.URL)
		}
		wire = nil
		if json.NewDecoder(r.Body).Decode(&wire) != nil {
			t.Fatal("上游请求体无法解码")
		}
		return &http.Response{StatusCode: status, Header: http.Header{"Location": {"https://example.test"}}, Body: io.NopCloser(strings.NewReader(reply))}, nil
	})
	run := func(body string) (string, int, error) {
		b, s, e := compatRun(compatTestModel, []byte(body), tr)
		return string(b), s, e
	}
	out, s, e := run(`{"input":["text"],"encoding_format":"base64","url":"https://evil.test","api-key":"leak"}`)
	if e != nil || s != 200 || !strings.Contains(out, `"AACAPwAAAMA="`) || !strings.Contains(out, `"total_tokens":4`) || !strings.Contains(out, `"model":"BAAI/bge-m3"`) {
		t.Fatal(out, s, e)
	}
	if len(wire) != 2 || wire["encoding_format"] != nil || wire["model"] != compatTestModel || wire["url"] != nil {
		t.Fatalf("白名单未生效: %v", wire)
	}
	// 客户端未指定编码格式时，不向上游强制发送其可能不支持的 encoding_format。
	if out, s, e := run(`{"input":"text"}`); s != 200 || e != nil {
		t.Fatal(out, s, e)
	}
	if _, ok := wire["encoding_format"]; ok || len(wire) != 2 {
		t.Fatalf("encoding_format 被强制发送: %v", wire)
	}
	for _, pair := range [][2]string{{"1,-2", "null"}, {"1,-2", "1e100"}, {"1,-2", ""}, {"\"index\":0", "\"index\":1"}, {"\"index\":0,", ""}, {"prompt_tokens", "unused"}, {"total_tokens\":4", "total_tokens\":0"}} {
		reply = strings.Replace(good, pair[0], pair[1], 1)
		if _, s, e := run(`{"input":"text"}`); s != 502 || e == nil {
			t.Errorf("无效响应被接受: %s", reply)
		}
	}
	reply = good
	if _, s, _ := run(`{"input":["a","b"]}`); s != 502 {
		t.Error("短批量被接受")
	}
	status = 302
	reply = "sensitive-upstream-body"
	before := calls
	if out, s, e := run(`{"input":"x"}`); calls != before+1 || s != 502 || e == nil || strings.Contains(e.Error(), reply) || out != "" {
		t.Fatal("重定向未被截断")
	}
	status, reply = 200, good
	before = calls
	for _, b := range []string{`{"input":null}`, `{"input":[null]}`, `{"input":"a","stream":true}`, `{"input":"a","encoding_format":null}`, `{"input":[` + strings.Repeat(`"a",`, 32) + `"a"]}`, `{"input":"` + strings.Repeat("x", embMaxInput+1) + `"}`} {
		if _, s, _ := run(b); s != 400 && s != 413 {
			t.Errorf("无效输入被接受: %s", b)
		}
	}
	if before != calls {
		t.Fatal("无效输入被发往上游")
	}
}

// 无序批量按 index 复原，上游 base64 向量可解码，usage 必须原样透传真实计数。
func TestCompatBatchOrderAndUsage(t *testing.T) {
	compatConfig(t, compatCfg(compatGroupYAML("dashscope-embeddings", "https://dashscope.aliyuncs.com/compatible-mode/v1", "",
		"[{name: text-embedding-v4, alias: qwen-embed}]", "[{api-key: unit-dashscope}]")))
	const reply = `{"data":[{"index":1,"embedding":[-1,0]},{"index":0,"embedding":"AADAPwAAAAA="}],"usage":{"prompt_tokens":7,"total_tokens":9}}`
	tr := compatTransport(func(r *http.Request) (*http.Response, error) {
		var q struct {
			Model      string
			Encoding   string `json:"encoding_format"`
			Input      []string
			Dimensions int
		}
		if r.URL.String() != "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings" || r.Header.Get("Authorization") != "Bearer unit-dashscope" || json.NewDecoder(r.Body).Decode(&q) != nil {
			t.Fatalf("非预期上游请求: %s", r.URL)
		}
		if q.Model != "text-embedding-v4" || q.Encoding != "" || len(q.Input) != 2 || q.Input[0] != "first" || q.Input[1] != "second" || q.Dimensions != 2 {
			t.Fatalf("客户端字段被改写: %+v", q)
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(reply))}, nil
	})
	b, s, e := compatRun("qwen-embed", []byte(`{"dimensions":2,"input":["first","second"]}`), tr)
	out := string(b)
	if s != 200 || e != nil || !strings.Contains(out, `"model":"qwen-embed"`) ||
		!strings.Contains(out, `"embedding":[1.5,0]`) || !strings.Contains(out, `"embedding":[-1,0]`) ||
		!strings.Contains(out, `"prompt_tokens":7`) || !strings.Contains(out, `"total_tokens":9`) {
		t.Fatal(out, s, e)
	}
	for _, mut := range []string{
		strings.Replace(reply, `"prompt_tokens":7`, `"prompt_tokens":0`, 1),
		strings.Replace(reply, `"total_tokens":9`, `"total_tokens":3`, 1),
		strings.Replace(reply, `"index":1`, `"index":0`, 1),
		strings.Replace(reply, `"index":1`, `"index":2`, 1),
		strings.Replace(reply, `[-1,0]`, `[-1,0,3]`, 1),
		strings.Replace(reply, `"AADAPwAAAAA="`, `"not base64!"`, 1),
	} {
		bad := compatTransport(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(mut))}, nil
		})
		if _, s, e := compatRun("qwen-embed", []byte(`{"input":["first","second"]}`), bad); s != 502 || e == nil {
			t.Errorf("无效批量响应被接受: %s", mut)
		}
	}
}

// 同一模型名跨两个端点时必须显式切换：各自只绑定自己的上游，响应 model 用客户端名。
func TestCompatAliasSwitchAcrossEndpoints(t *testing.T) {
	compatConfig(t, compatCfg(
		compatGroupYAML("siliconflow-embeddings", "https://api.siliconflow.cn/v1", "",
			"[{name: BAAI/bge-m3, alias: BAAI/bge-m3}]", "[{api-key: unit-silicon, weight: 1}]"),
		compatGroupYAML("qwen-embeddings", "https://dashscope.aliyuncs.com/compatible-mode/v1", "",
			"[{name: text-embedding-v4, alias: qwen-embed}]", "[{api-key: unit-dashscope}]"),
	))
	for _, tc := range []struct{ client, real, base, key string }{
		{"BAAI/bge-m3", "BAAI/bge-m3", "https://api.siliconflow.cn/v1", "unit-silicon"},
		{"qwen-embed", "text-embedding-v4", "https://dashscope.aliyuncs.com/compatible-mode/v1", "unit-dashscope"},
	} {
		tr := compatTransport(func(r *http.Request) (*http.Response, error) {
			var q struct{ Model string }
			if r.URL.String() != tc.base+"/embeddings" || r.Header.Get("Authorization") != "Bearer "+tc.key || json.NewDecoder(r.Body).Decode(&q) != nil || q.Model != tc.real {
				t.Fatalf("%s: 绑定错误 %s %+v", tc.client, r.URL, q)
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"data":[{"index":0,"embedding":[0.5]}],"usage":{"prompt_tokens":2,"total_tokens":2}}`))}, nil
		})
		b, s, e := compatRun(tc.client, []byte(`{"model":"`+tc.client+`","input":"text"}`), tr)
		if s != 200 || e != nil || !strings.Contains(string(b), `"model":"`+tc.client+`"`) || !strings.Contains(string(b), `"embedding":[0.5]`) {
			t.Fatalf("%s: %s %d %v", tc.client, b, s, e)
		}
	}
}
