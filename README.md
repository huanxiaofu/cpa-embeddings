# CPR Embeddings

让知识库和其他 OpenAI 兼容客户端用原有 CPA 客户端 Key 调用 `POST /v1/embeddings`。支持 Gemini 原生向量接口和 SiliconFlow OpenAI 兼容接口。用量由 **CLIProxyAPI 的执行链**登记；插件不保管下游 Key，也不直接改面板数据库。

## 客户端

```python
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ['CPR_BASE_URL'], api_key=os.environ['CPR_API_KEY'])
r = client.embeddings.create(model='gemini-embedding-2', input=['入库单', '出库单'])
print(len(r.data), len(r.data[0].embedding), r.usage.prompt_tokens)
```

`CPR_BASE_URL` 是网关的 `https://域名/v1`，不是 CPA 管理地址。`input` 支持一个字符串或最多 32 条文本，总大小不超过 256 KiB；响应支持 `float` 和 `base64`。不处理图片或 token-ID 数组。`dimensions` 是否可用取决于选用的上游模型。不同模型的向量不能在同一个索引里混用，切换时应重新嵌入数据。

## 工作方式与计量

1. OpenResty 的**精确匹配** `/v1/embeddings` 校验文本请求和允许的模型，再将其交给 CPA 已有的、需要普通客户端 Key 的 `/v1/responses` 入口，并附加内部操作标记。其他路径不变；在普通 `/v1/` 规则清除客户端自带的 `X-Cpr-Operation`。客户端不接触管理密钥。
2. `.so` 通过 CPA 8.1.0 的 ModelRouter/Executor ABI 处理带标记的非流式请求。Gemini 使用 `embedContent` 或 `batchEmbedContents`；SiliconFlow 使用其固定 HTTPS `/v1/embeddings`。两者均只从当前 CPA 配置读取显式启用、模型匹配的上游凭据，当前版本要求单个可用凭据，不实现多 Key 调度。
3. Gemini 优先读取 Google 的 `usageMetadata.promptTokenCount`，缺失时请求同模型的 `countTokens`。SiliconFlow 读取响应里的 `usage`。无法获得有效计数则不返回成功，不使用字符数估算。原生宿主从成功响应解析输入和总 Token，再按鉴权后的客户端身份记账。由于内部沿用了 Responses，后台的 endpoint 可能显示 `/v1/responses`；用量按 Key 和实际模型分组。

统计 Token 不等于定价或扣费。金额取决于面板的价格设置；本插件不设置单价，也不另建额度或计费系统。

## 构建和部署

目标：Linux，CLIProxyAPI **8.1.0**，Go 1.24+、C 编译器及支持 Lua 的 OpenResty。

```sh
go test ./...
go vet ./...
mkdir -p dist
CGO_ENABLED=1 go build -trimpath -ldflags='-s -w' -buildmode=c-shared \
  -o dist/cpr-gemini-embeddings.so .
sha256sum dist/cpr-gemini-embeddings.so
```

- 部署前备份 CPA 配置与站点配置。将 `.so` 放入 CPA 对应 OS/架构插件目录，在**已有** `plugins.configs` 下添加 `cpr-gemini-embeddings: {enabled: true}`。不要复制一个新的顶层 `plugins` 段。
- 通过 CPA 自己的渠道管理登记实际要开放的模型，并保留真实凭据在服务端。Gemini 的 `base-url` 为 `https://generativelanguage.googleapis.com`；SiliconFlow 的为 `https://api.siliconflow.cn/v1`。别名需与真实模型名一致；不把其他聊天或重排模型加入向量入口。插件默认从服务工作目录读取 `config.yaml`，可用 `CPR_EMBED_CONFIG` 指定路径。
- 若 Gemini 渠道配置的 Docker 代理主机名在 CPA 宿主不可解析，可在配置文件同目录创建 `cpr-embeddings.local.yaml`，用 `proxy_aliases` 映射到**同一出口**的宿主监听地址；不要让请求静默改走直连。
- 在现有 HTTPS 站点添加 [`deploy/openresty.conf.in`](deploy/openresty.conf.in)，将 `CPA_UPSTREAM` 替换为 CPA 服务的内网 `host:port`，把实际验证的 SiliconFlow 型号加入模板内 `models` 白名单。对普通 `/v1/` 移除同名内部请求头，执行 `nginx -t` 后平滑重载。不需要暴露 `/v0/management`。
- 验收：错误 Key 应返回 401；有效 Key 对两类上游均返回数值向量和正的 Token 数。在 CPA/oh-my-cpa 后台核对该 Key、模型、请求次数、输入和总 Token；再确认普通聊天仍正常。回退时先移除公开向量入口，再停用插件，恢复配置与原有路由。

代码不包含实际 API Key、服务器地址、管理凭据或生产配置。部署清单与私有配置勿提交至 Git。
