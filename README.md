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

## 自动识别和管理面板

**一个 CPA 渠道可以同时登记聊天、向量与重排模型。** 不需要改渠道名、拆渠道、为 Embeddings 再填一遍 Key，也不需要增删型号时重新生成 OpenResty 配置。

OpenAI 兼容渠道先看模型的 `output-modalities`；没有明确声明时，按真实 `name` 识别常见向量模型族，例如 `text-embedding-3-small`、`text-embedding-3-large`、`text-embedding-ada-002`、Qwen Embedding、BGE、Mistral Embed、Jina Embeddings 等。`alias` 只用于客户端调用，不能把聊天模型伪装为向量模型。模型名不是标准能力标记，不认识的型号默认不开。自动识别不代表上游账号有额度。

需要覆盖自动识别时：

1. 管理员打开网关的 **`/embedding-models/`**，用 **CPA 管理密钥**连接。普通应用仍然只使用客户端 Key。面板只列兼容渠道的模型、别名和选择结果，不返回上游地址或密钥。
2. 选渠道或搜索真实名/别名，将需要的**单个模型**设为「开放」「关闭」或「自动」，再点该行的「保存」。默认「自动」，不用逐条选择。修改不会影响原有聊天路由；不能因人工开放而跳过凭据、唯一绑定、真实用量等校验。
3. 保存后下一次向量请求即按新规则处理。人工设置以**渠道名 + 真实模型名**存放在配置同目录 `cpa-embeddings-state/overrides.json`，备份时一并保存。「自动」恢复自动识别（也会覆盖同项旧 YAML 预设）。

协议：`api-keys.openai-compatibility` 默认走 HTTPS Bearer 的 `POST {base-url}/embeddings`，可接 OpenAI、硅基流动、百炼兼容模式、Mistral 等；同模型要求唯一可用绑定，不会在不支持的代理或凭据策略下静默直连。Cohere v2 原生用 `https://api.cohere.com/v2`，任意组名可通过 `plugins.configs.cpr-gemini-embeddings.upstream-protocols` 指定 `渠道名: cohere`，该字段只决定协议、不会开放整组。原生 Gemini 仍从 `api-keys.gemini` 读取显式配置的 Google 向量模型。与协议不兼容或不返回真实用量的服务不等同于已支持。

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
- 在 CPA 配置目录内创建私有 `cpa-embeddings-state/` 目录（`0700`，归服务运行账户所有）。不要把包含上游 Key 的配置文件暴露给网页或 OpenResty。
- 首次安装/从 v0.2 升级时，使用匹配版本的工具生成站点片段，替换旧的向量 location，执行 `nginx -t` 后平滑 reload。此后添加渠道、增删模型、在面板保存都**不需要**再渲染或重载站点。

  ```sh
  cpa-embeddings-config render-edge /path/to/config.yaml cpa:8317 > embeddings.location.new
  ```

  仅部署在 HTTPS 管理员可信来源。模板只代理一个有 CPA 管理鉴权的模型选择 API 和不含数据的页面，不要另行暴露通用管理路径。普通 `/v1/` 规则须清除 `X-Cpr-Operation`。精确 `/v1/embeddings` 用内部子请求查询正在运行的插件，未就绪时拒绝请求而不回落到聊天。原生插件 ID 保持不变；版本化库文件名为 `cpr-gemini-embeddings-v0.3.0.so`。
- 验收：错误 Key 应返回 401；有效 Key 对两类上游均返回数值向量和正的 Token 数。在 CPA/oh-my-cpa 后台核对该 Key、模型、请求次数、输入和总 Token；再确认普通聊天仍正常。回退时先移除公开向量入口，再停用插件，恢复配置与原有路由。

代码不包含实际 API Key、服务器地址、管理凭据或生产配置。部署清单与私有配置勿提交至 Git。
