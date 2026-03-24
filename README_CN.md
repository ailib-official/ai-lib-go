# ai-lib-go

**AI-Protocol 官方 Go 运行时** - 统一 AI 模型交互的高性能、惯用 Go 实现。

[![Go 1.21+](https://img.shields.io/badge/go-1.21+-00ADD8.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

## 🎯 设计理念

`ai-lib-go` 是 [AI-Protocol](https://github.com/ailib-official/ai-protocol) 规范的**官方 Go 运行时**实现。它体现了核心设计原则：

> **一切逻辑皆算子，一切配置皆协议** (All logic is operators, all configuration is protocol)

与传统的适配器库硬编码特定提供商逻辑不同，`ai-lib-go` 是一个**协议驱动的运行时**，执行 AI-Protocol 规范。这意味着：

- **零硬编码提供商逻辑**：所有行为都由协议清单（YAML/JSON 配置）驱动
- **基于算子的架构**：流式处理使用 manifest 驱动的解码器（openai_sse、anthropic_sse），支持可组合流水线
- **统一接口**：开发者与单一、一致的 API 交互，无论底层提供商是什么
- **跨运行时一致**：通过共享合规测试与 ai-lib-rust、ai-lib-python、ai-lib-ts 对齐

## 🚀 快速入门

### 安装

```bash
go get github.com/ailib-official/ai-lib-go
```

### 基础用法

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
)

func main() {
	client, err := ailib.NewClientBuilder().
		WithProtocolData([]byte(manifestYAML)).
		WithAPIKey(os.Getenv("OPENAI_API_KEY")).
		Build()
	if err != nil {
		panic(err)
	}
	defer client.Close()

	resp, err := client.Chat(context.Background(), []ailib.Message{
		{Role: ailib.RoleUser, Content: "你好！2+2 等于多少？"},
	}, &ailib.ChatOptions{Model: "gpt-4o"})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
}
```

### 流式调用

```go
stream, err := client.ChatStream(ctx, messages, opts)
if err != nil {
	return err
}
defer stream.Close()

for stream.Next() {
	ev := stream.Event()
	if ev.Delta != "" {
		fmt.Print(ev.Delta)
	}
}
if err := stream.Err(); err != nil {
	return err
}
```

## ✨ 特性

- **协议驱动**：所有行为由 YAML/JSON 协议文件驱动
- **统一接口**：单一 API 支持所有 AI 提供商（OpenAI、Anthropic、Gemini、DeepSeek 等）
- **流式优先**：原生流式接口，支持 manifest 驱动的解码器（openai_sse、anthropic_sse）
- **类型安全**：完整的 Go 类型定义
- **生产就绪**：内置重试、熔断和回退执行器
- **易于扩展**：通过协议配置轻松添加新提供商
- **多模态支持**：支持文本、图像、音频、视频
- **能力门禁**：未声明高级能力快速失败（E1005）
- **向量嵌入**：Embeddings API
- **批处理**：BatchCreate、BatchGet、BatchCancel
- **STT/TTS**：语音转文字与文字转语音
- **重排序**：文档重排序
- **MCP**：MCP 工具桥接（list/call）
- **回退客户端**：健康快照与熔断策略

## 🔄 V2 协议对齐

`ai-lib-go` 与 **AI-Protocol V2** 规范对齐。V0.5.0 包含 V1/V2 manifest 解析、manifest 驱动的流式解码器、标准错误码及合规测试覆盖。

### 标准错误码（V2，ARCH-003）

所有 provider 错误被分类为 13 个标准错误码，具有统一的重试/回退语义：

| 错误码 | 名称             | 可重试 | 可回退 |
|--------|------------------|--------|--------|
| E1001  | invalid_request  | 否     | 否     |
| E1002  | authentication   | 否     | 是     |
| E1003  | permission_denied| 否     | 否     |
| E1004  | not_found        | 否     | 否     |
| E1005  | request_too_large| 否     | 否     |
| E2001  | rate_limited     | 是     | 是     |
| E2002  | quota_exhausted  | 是     | 是     |
| E3001  | server_error     | 是     | 是     |
| E3002  | overloaded       | 是     | 是     |
| E3003  | timeout          | 是     | 是     |
| E4001  | conflict         | 是     | 否     |
| E4002  | cancelled        | 否     | 否     |
| E9999  | unknown          | 否     | 否     |

分类优先级：manifest `error_classification` → provider code/type → HTTP 状态码 → `E9999`。

### 合规测试

跨运行时行为一致性通过 `ai-protocol` 仓库中的共享 YAML 测试套件验证：

```bash
# 运行单元测试
go test ./pkg/ailib/...

# 运行合规测试（需在工作区包含 ai-protocol 或设置 COMPLIANCE_DIR）
COMPLIANCE_DIR=../ai-protocol/tests/compliance go test ./tests/compliance/...
```

详见 [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md)。

### 使用 ai-protocol-mock 测试

无需真实 API 调用的集成测试，可使用 [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)：

```bash
# 启动 mock 服务（在 ai-protocol-mock 仓库中）
docker-compose up -d

# 使用 mock base URL 运行测试
client, _ := ailib.NewClientBuilder().
	WithBaseURL("http://localhost:4010").
	WithAPIKey("test-key").
	Build()
```

## 📁 目录结构

- `internal/protocol` — manifest 契约、加载、能力元信息、流式解码格式
- `internal/stream` — SSE 解码（openai_sse、anthropic_sse）
- `internal/resilience` — 重试与退避策略
- `pkg/ailib` — 对外 SDK API（`Client`、`ClientBuilder`、能力模型、`FallbackClient`）
- `tests/compliance` — 基于共享夹具的合规执行器

## 🗺️ 生态定位

| 项目            | 用途                     |
|-----------------|--------------------------|
| ai-protocol     | 规范、Schema、Manifest   |
| ai-lib-rust     | Rust 运行时             |
| ai-lib-python   | Python 运行时           |
| ai-lib-ts       | TypeScript 运行时       |
| **ai-lib-go**   | **Go 运行时**           |
| ai-protocol-mock| 测试用 Mock 服务        |
| spiderswitch    | MCP 模型切换            |
| ailib.info      | 文档站点                |

## 📜 发布约定

- 采用 semver，并按 ai-lib 矩阵发布节奏推进
- 运行时发布需与 `ai-protocol` 兼容窗口对齐
- `ailib.info` 站点对齐在运行时发布收敛后执行

## 📄 许可证

MIT OR Apache-2.0
