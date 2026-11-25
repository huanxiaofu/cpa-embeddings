// SPDX-License-Identifier: MIT
// C ABI 改自 CLIProxyAPI 8.1.0 examples/plugin/claude-web-search-router/go（见 LICENSE）。
package main

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	void* ptr;
	size_t len;
} cliproxy_buffer;

typedef int (*cliproxy_host_call_fn)(void*, const char*, const uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_host_free_fn)(void*, size_t);

typedef struct {
	uint32_t abi_version;
	void* host_ctx;
	cliproxy_host_call_fn call;
	cliproxy_host_free_fn free_buffer;
} cliproxy_host_api;

typedef int (*cliproxy_plugin_call_fn)(char*, uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_plugin_free_fn)(void*, size_t);
typedef void (*cliproxy_plugin_shutdown_fn)(void);

typedef struct {
	uint32_t abi_version;
	cliproxy_plugin_call_fn call;
	cliproxy_plugin_free_fn free_buffer;
	cliproxy_plugin_shutdown_fn shutdown;
} cliproxy_plugin_api;

extern int cliproxyPluginCall(char*, uint8_t*, size_t, cliproxy_buffer*);
extern void cliproxyPluginFree(void*, size_t);
extern void cliproxyPluginShutdown(void);

*/
import "C"

import (
	"encoding/json"
	"strings"
	"sync/atomic"

	"gopkg.in/yaml.v3"
	"unsafe"
)

const abiPluginID = "cpr-gemini-embeddings"
const abiFormat = "openai-response"

var abiDisabled atomic.Bool

func main() { edgeCLI() }

//export cliproxy_plugin_init
func cliproxy_plugin_init(host *C.cliproxy_host_api, plugin *C.cliproxy_plugin_api) C.int {
	if plugin == nil || host == nil || host.abi_version != 1 {
		return 1
	}
	plugin.abi_version = 1
	plugin.call = C.cliproxy_plugin_call_fn(C.cliproxyPluginCall)
	plugin.free_buffer = C.cliproxy_plugin_free_fn(C.cliproxyPluginFree)
	plugin.shutdown = C.cliproxy_plugin_shutdown_fn(C.cliproxyPluginShutdown)
	return 0
}

//export cliproxyPluginFree
func cliproxyPluginFree(p unsafe.Pointer, _ C.size_t) { C.free(p) }

//export cliproxyPluginShutdown
func cliproxyPluginShutdown() {}

//export cliproxyPluginCall
func cliproxyPluginCall(method *C.char, request *C.uint8_t, size C.size_t, response *C.cliproxy_buffer) C.int {
	if response == nil {
		return 1
	}
	response.ptr = nil
	response.len = 0
	if method == nil || (request == nil && size != 0) || size > C.size_t(^uint(0)>>1) {
		return 1
	}
	// 借用宿主缓冲区，仅在本次同步调用中读取；返回值由宿主调用 free_buffer 释放。
	result := abiDispatch(C.GoString(method), unsafe.Slice((*byte)(unsafe.Pointer(request)), int(size)), dispatchEmbedding)
	b, err := json.Marshal(result)
	if err != nil {
		return 1
	}
	response.ptr = C.CBytes(b)
	response.len = C.size_t(len(b))
	return 0
}

type abiEnvelope struct {
	OK     bool      `json:"ok"`
	Result any       `json:"result,omitempty"`
	Error  *abiError `json:"error,omitempty"`
}
type abiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  int    `json:"http_status"`
}

func abiOK(v any) abiEnvelope { return abiEnvelope{OK: true, Result: v} }
func abiFail(code, msg string, status int) abiEnvelope {
	return abiEnvelope{Error: &abiError{code, msg, status}}
}

type abiIngress struct {
	SourceFormat string
	Stream       bool
	Headers      map[string][]string
}

func (r abiIngress) accepts(model string) bool {
	if abiDisabled.Load() || r.Stream || r.SourceFormat != abiFormat {
		return false
	}
	// 标记由精确入口设置。旧边缘清单中的型号即使已撤销，也由执行器拒绝，不能回落聊天。
	if model == "" {
		return false
	}
	// 标记必须唯一；拒绝重复字段，不能因 map 遍历顺序改变路由。
	found := false
	for key, values := range r.Headers {
		if strings.EqualFold(key, "X-Cpr-Operation") {
			if found || len(values) != 1 || values[0] != "embeddings" {
				return false
			}
			found = true
		}
	}
	return found
}

type abiRoute struct {
	Handled    bool
	TargetKind string `json:",omitempty"`
}
type abiPayload struct{ Payload []byte }

func abiDispatch(method string, raw []byte, run func(string, []byte) ([]byte, int, error)) abiEnvelope {
	invalid := func() abiEnvelope { return abiFail("invalid_request", "invalid plugin request", 400) }
	switch method {
	case "plugin.register", "plugin.reconfigure":
		// 先解开 config_yaml 的 base64 字节串，再读取插件实例的 YAML；配置无效时保留旧状态。
		var cfg struct {
			ConfigYAML []byte `json:"config_yaml"`
		}
		if json.Unmarshal(raw, &cfg) != nil {
			return invalid()
		}
		local := struct {
			Enabled bool `yaml:"enabled"`
		}{Enabled: true}
		if yaml.Unmarshal(cfg.ConfigYAML, &local) != nil {
			return abiFail("invalid_config", "invalid plugin configuration", 400)
		}
		abiDisabled.Store(!local.Enabled)
		return abiOK(map[string]any{
			"schema_version": 10,
			"metadata":       map[string]any{"Name": abiPluginID, "Version": "0.2.0", "Author": "huanxiaofu", "GitHubRepository": "https://github.com/huanxiaofu/cpa-embeddings", "Logo": "https://github.com/huanxiaofu.png", "ConfigFields": []any{}},
			"capabilities": map[string]any{
				"model_router": true, "executor": true, "executor_model_scope": "static",
				"executor_input_formats": []string{abiFormat}, "executor_output_formats": []string{abiFormat},
			},
		})
	case "plugin.quiesce", "plugin.shutdown":
		return abiOK(struct{}{})
	case "executor.identifier":
		return abiOK(map[string]string{"identifier": abiPluginID})
	case "model.route":
		var r struct {
			abiIngress
			RequestedModel string
		}
		if json.Unmarshal(raw, &r) != nil {
			return invalid()
		}
		if r.accepts(r.RequestedModel) {
			return abiOK(abiRoute{true, "self"})
		}
		return abiOK(abiRoute{})
	case "executor.execute":
		// 不解析 OriginalRequest：原始输入未经宿主拦截器过滤。
		var r struct {
			abiIngress
			Model   string
			Payload []byte
		}
		if json.Unmarshal(raw, &r) != nil || !r.accepts(r.Model) {
			return invalid()
		}
		body, status, err := run(r.Model, r.Payload)
		if err != nil || status < 200 || status >= 300 {
			if status < 400 || status > 599 {
				status = 502
			}
			return abiFail("executor_error", "embedding request failed", status)
		}
		return abiOK(abiPayload{body})
	case "executor.execute_stream", "executor.count_tokens", "executor.http_request":
		return abiFail("not_implemented", "unsupported executor operation", 501)
	default:
		return abiFail("unknown_method", "unknown plugin method", 400)
	}
}
