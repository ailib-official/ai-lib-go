# ai-lib-go

**[AI-Protocol](https://github.com/ailib-official/ai-protocol) 协议运行时** — Go 参考实现（**v1.2.0**，Go 1.21+）。

[English](README.md)

`ai-lib-go` 将执行层与策略层拆到两个公共包：

| 包 | 层级 | 职责 |
|----|------|------|
| `github.com/ailib-official/ai-lib-go/pkg/ailib` | 执行层 (E) | `Client`、清单驱动的 HTTP 对话、能力端点 |
| `github.com/ailib-official/ai-lib-go/pkg/contact` | 策略层 (P) | `FallbackClient`、熔断策略、`HealthSnapshot` |

> **钉版本：** 优先模块标签 **`v1.2.0`**。CI 钉住 `ai-protocol` **v1.2.0**。见 [CHANGELOG](CHANGELOG.md)。

## 工作原理

**默认对话路径（`pkg/ailib`）：** 加载清单（可选）→ 构造 JSON → 从清单解析 endpoint/鉴权 → `net/http` 发送 → 微重试（`internal/resilience`）→ 解析响应 → 填充 `ExecutionMetadata`。

**无清单时：** `WithBaseURL` + `WithAPIKey` 使用 OpenAI 兼容默认（`/chat/completions`、`openai_sse` 解码器）。这是逃生舱，不是主集成模式。

**流式：** `ChatStream` → 清单指定的 SSE 解码器（默认 `openai_sse`）→ `Stream.Next()` / `Event()`。成功流结束后先 `Close()`，再取 `ExecutionMetadata()`。

没有公开的算子 `Pipeline` API；对话是直接 HTTP + 解码器。

能力端点（embeddings、batch、STT/TTS、rerank、MCP、computer use、reasoning、video）通过 `protocol.EndpointFor` 解析，仅回退到**路径**（如 `/embeddings`、`/rerank`），不会静默写入厂商主机。

## 快速开始（协议优先）

```bash
go get github.com/ailib-official/ai-lib-go@v1.2.0
export OPENAI_API_KEY="your-key"
```

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
)

func main() {
	manifestYAML := `id: openai
protocol_version: "2.0"
endpoint:
  base_url: "https://api.openai.com/v1"
`
	client, err := ailib.NewClientBuilder().
		WithProtocolData([]byte(manifestYAML)).
		WithAPIKey(os.Getenv("OPENAI_API_KEY")).
		Build()
	if err != nil {
		panic(err)
	}
	defer client.Close()

	resp, err := client.Chat(context.Background(), []ailib.Message{
		{Role: ailib.RoleUser, Content: "你好！"},
	}, &ailib.ChatOptions{Model: "gpt-4o"})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Println("provider:", resp.ExecutionMetadata.ProviderID)
}
```

示例来源：`pkg/ailib/client_execution_metadata_test.go`。

### 仅 BaseURL 模式（OpenAI 兼容）

```go
client, err := ailib.NewClientBuilder().
	WithBaseURL("https://api.openai.com/v1").
	WithAPIKey(os.Getenv("OPENAI_API_KEY")).
	Build()
```

适用于本地没有清单仓库时；行为使用内置 OpenAI 风格路径。

### 流式调用

```go
stream, err := client.ChatStream(ctx, messages, opts)
if err != nil {
	return err
}
defer stream.Close()

for stream.Next() {
	if ev := stream.Event(); ev.Delta != "" {
		fmt.Print(ev.Delta)
	}
}
if err := stream.Err(); err != nil {
	return err
}
if meta, ok := stream.ExecutionMetadata(); ok {
	_ = meta // provider_id、model_id、usage、micro_retry_count
}
return nil
```

### 策略层：回退客户端

熔断与多提供商回退在 **`pkg/contact`**，不在 `pkg/ailib`：

```go
import (
	"github.com/ailib-official/ai-lib-go/pkg/ailib"
	"github.com/ailib-official/ai-lib-go/pkg/contact"
)

primary, _ := ailib.NewClientBuilder(). /* ... */ Build()
secondary, _ := ailib.NewClientBuilder(). /* ... */ Build()

fb := contact.NewFallbackClient(primary, secondary)
resp, err := fb.Chat(ctx, messages, opts)
_ = fb.HealthSnapshot() // 各提供商熔断状态
```

`NewFallbackClientWithPolicy` 可传入自定义 `FallbackPolicy`（`FailureThreshold`、`CircuitOpenFor`）。

## 公共 API（`pkg/ailib`）

**构建器：** `NewClientBuilder` → `WithProtocolPath` / `WithProtocolData` / `WithBaseURL` / `WithAPIKey` / `WithHeader` / `WithTimeout` / `WithMaxRetries` / `WithHTTPClient` → `Build()`。协议数据、协议路径、BaseURL 三者必选其一。

**`Client` 接口：**

- `Chat`、`ChatStream`
- 能力门禁 HTTP 路由：`Embeddings`、`BatchCreate`/`BatchGet`/`BatchCancel`、`STTTranscribe`、`TTSSpeak`、`Rerank`、`MCPListTools`、`MCPCallTool`、`ComputerUse`、`Reason`、`VideoGenerate`、`VideoGet`
- `Close`

未声明能力返回 **E1005**（`ErrUnsupported`；`StandardErrorName` 映射为 `request_too_large` — 同时用作能力门禁）。

**类型：** `Message`、`ChatOptions`、`ChatResponse`、`StreamingEvent`、`Stream`、`ExecutionMetadata`、`ExecutionUsage`、`ExecutionResult`、能力请求/响应结构体、标准错误常量 **E1001–E9999**、`IsRetryableCode` / `IsFallbackableCode`。

**`pkg/streaming`：** 面向合规的 SSE / 事件映射辅助（合规测试使用；普通 `Client.ChatStream` 不必依赖）。

## 能力边界（如实说明）

| 领域 | 模块内有 | 不包含 |
|------|----------|--------|
| **MCP** | 清单 HTTP 路由 + 能力门禁 | MCP 线协议客户端 |
| **Computer Use** | HTTP 路由 + 请求/响应类型 | 动作执行沙箱 |
| **多模态** | `Message.Content` JSON 透传 | 高层多模态构造器 |
| **重试** | 非流式对话的 E 层微重试 | 熔断（仅 P 层） |
| **熔断 / 回退** | `pkg/contact.FallbackClient` | 不在裸 `pkg/ailib.Client` 上 |
| **提供商身份** | `internal/protocol.Loader.LoadProvider`（main / Unreleased） | 尚无公开的 `ClientBuilder.WithProviderID` |

## 协议清单（加载器）

`Client` 通过 `WithProtocolPath` / `WithProtocolData` 加载清单。按 id 查找时，`internal/protocol.Loader` 解析顺序为：

1. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH`，或 `Loader.Root`，或邻近 `ai-protocol/` 检出
2. 每个 id：`dist/v2/providers/<id>.json` → `dist/v1/...` → 源码树 `v2`/`v1` YAML|JSON
3. **身份 / 别名（GO-ID-001）：** 精确 id 缺失时，经 `dist/provider-identity.json`（多家族 map）将别名解析为规范 id，例如 `google` → `gemini`、`kimi` → `moonshot`。解析/校验错误不会被别名回退掩盖。

## API 密钥（BYOK 链）

1. `WithAPIKey` 覆盖
2. 清单 `endpoint.auth` / 顶层 `auth` 声明的环境变量
3. 约定式 `<PROVIDER_ID>_API_KEY`

## 代理

Go **不会**自动读取 `AI_PROXY_URL` / `HTTP_PROXY`。请通过 `WithHTTPClient` 传入自定义 `*http.Client`（例如 `Transport: &http.Transport{Proxy: http.ProxyFromEnvironment}`）。

跨运行时说明：[CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md)。

## 标准错误码（V2）

| 错误码 | 名称 | 可重试 | 可回退 |
|--------|------|--------|--------|
| E1001 | `invalid_request` | 否 | 否 |
| E1002 | `authentication` | 否 | 是 |
| E1003 | `permission_denied` | 否 | 否 |
| E1004 | `not_found` | 否 | 否 |
| E1005 | `request_too_large` | 否 | 否 |
| E2001 | `rate_limited` | 是 | 是 |
| E2002 | `quota_exhausted` | 否 | 是 |
| E3001 | `server_error` | 是 | 是 |
| E3002 | `overloaded` | 是 | 是 |
| E3003 | `timeout` | 是 | 是 |
| E4001 | `conflict` | 是 | 否 |
| E4002 | `cancelled` | 否 | 否 |
| E9999 | `unknown` | 否 | 否 |

分类优先级：清单 `error_classification` → 提供商 code/type → HTTP 状态码 → `E9999`。

## 测试

```bash
go test ./...
```

合规（共享 YAML）：

```bash
COMPLIANCE_DIR=../ai-protocol/tests/compliance go test ./tests/compliance/ -v
```

Mock 服务（[ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)）：

```bash
client, _ := ailib.NewClientBuilder().
	WithBaseURL("http://localhost:4010").
	WithAPIKey("test-key").
	Build()
```

## 相关链接

- [AI-Protocol](https://github.com/ailib-official/ai-protocol)
- [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust)
- [ai-lib-python](https://github.com/ailib-official/ai-lib-python)
- [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts)

## 许可证

采用 [Apache-2.0](LICENSE-APACHE) 或 [MIT](LICENSE-MIT) 双许可。
