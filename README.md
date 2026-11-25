# CPA Embeddings

让知识库和其他 OpenAI 兼容客户端用原有 CPA 客户端 Key 调用 `POST /v1/embeddings`。上游按管理员配置选择：支持 OpenAI 兼容 Embeddings、Google Gemini 原生和 Cohere v2 Embed。添加兼容供应商或模型不需要重新编译。用量由 **CLIProxyAPI 的执行链**登记；插件不保管下游 Key，也不直接改面板数据库。

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
2. 插件处理带内部标记的非流式请求，未登记型号不回落到聊天。客户端只能传模型名，不能指定上游地址、认证或自定义请求头。
3. **原生用量**：OpenAI 兼容响应必须提供 `usage.prompt_tokens` 和 `usage.total_tokens`；Gemini 使用 `usageMetadata.promptTokenCount`，缺失时请求同模型 `countTokens`；Cohere 使用 `meta.billed_units.input_tokens`。缺少有效计数不返回成功，不按字数估算。CPA 记录的是客户端 Key 与请求模型（配置了别名时即别名）。内部 endpoint 可能显示 `/v1/responses`。

## 接入新上游

| 上游协议 | CPA 配置位置 | 示例 | 是否实测真实上游 |
|---|---|---|---|
| OpenAI 兼容 `POST {base-url}/embeddings` | `api-keys.openai-compatibility`，组名以 `-embeddings` 结尾 | OpenAI、硅基流动、阿里云百炼兼容模式、Mistral、自建 HTTPS 网关 | 硅基流动已实测；其他按协议和合成测试验证 |
| Gemini 原生 `embedContent` / `batchEmbedContents` | `api-keys.gemini`，官方 base，显式填写原生 `gemini-embedding-*` / `text-embedding-*` 型号 | Gemini Embedding | 已实测 |
| [Cohere v2 Embed](https://docs.cohere.com/reference/embed) | 同一兼容渠道配置区，组名为 `cohere-embeddings` 或以 `-cohere-embeddings` 结尾，base 为 `https://api.cohere.com/v2` | `embed-v4.0` | 原生请求与回执已做合成测试，未用真实账号调用 |

示例：在**已有** `api-keys.openai-compatibility` 列表追加一项；凭据只在 CPA 服务端注入，不要提交真实配置。普通聊天组不要改名。

```yaml
- name: openai-embeddings
  base-url: https://api.openai.com/v1
  keys:
    - api-key: "<服务端上游凭据>"
  models:
    - name: text-embedding-3-small   # 发给上游的真实名称
      alias: kb-openai              # 客户端及面板使用的名称，可省略
```

已有渠道不需要为后缀改名：可在 `plugins.configs.cpr-gemini-embeddings.upstream-protocols` 中登记 `已有组名: openai` 或 `已有组名: cohere`。该组应只列出要开放的向量模型，聊天、重排另放普通组。

例如换为百炼时可另建 `dashscope-embeddings`，填写服务地区对应的 `.../compatible-mode/v1` 地址及真实模型名。每个客户可见的模型名须唯一；同组可登记多个向量模型，不同组可以并存。每个模型目前要求唯一可用凭据，不实现多 Key 轮询。兼容组仅支持 HTTPS Bearer，无重定向；配置了尚不支持的代理、前缀、额外请求头或排除规则时不静默直连。协议不同（如 AWS SigV4）或不提供真实用量的服务不等于已支持。Cohere 可用 `input_type: search_document`（默认）建库，查询时用 `search_query`。

CPA 的 `observability.usage.usage-statistics-enabled` 必须为 `true`，并确认运行实例已经加载，oh-my-cpa 也在消费该实例的用量队列。开关关闭时，客户端仍可能收到 `usage`，但后台不入账；已经丢弃的历史请求不会自动补回。

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
- 按上文配置上游，确保 CPA 已加载配置；插件读取运行目录 `config.yaml`，可通过 `CPR_EMBED_CONFIG` 指定。
- 若 Gemini 渠道配置的 Docker 代理主机名在 CPA 宿主不可解析，可在配置文件同目录创建 `cpr-embeddings.local.yaml`，用 `proxy_aliases` 映射到**同一出口**的宿主监听地址；不要让请求静默改走直连。
- 使用配套 `cpa-embeddings-config` 工具（或源码 `go run .`）生成现有 HTTPS 站点使用的 location。**每次增删公开模型**时重新生成该片段，diff 确认后替换旧片段，`nginx -t` 并平滑 reload；不用手抄型号、改 Go 代码或重新发布插件。普通 `/v1/` 保留 `proxy_set_header X-Cpr-Operation "";`。不要把未渲染的模板直接装到生产。

  ```sh
  cpa-embeddings-config render-edge /path/to/config.yaml cpa:8317 > embeddings.location.new
  ```

  工具只输出公开模型别名和网关 location，不输出上游密钥。原生插件 ID 与旧版本一致，升级不用改客户端或重建用量分组；热升级可使用 `cpr-gemini-embeddings-v0.2.0.so` 作为 CPA 版本化文件名。
- 验收：错误 Key 应返回 401；有效 Key 对两类上游均返回数值向量和正的 Token 数。在 CPA/oh-my-cpa 后台核对该 Key、模型、请求次数、输入和总 Token；再确认普通聊天仍正常。回退时先移除公开向量入口，再停用插件，恢复配置与原有路由。

代码不包含实际 API Key、服务器地址、管理凭据或生产配置。部署清单与私有配置勿提交至 Git。
