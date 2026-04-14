package protocol

// ResponsePathsConfig maps logical fields to dotted JSON paths (non-streaming extraction).
type ResponsePathsConfig struct {
	Content          string `yaml:"content" json:"content"`
	Usage            string `yaml:"usage" json:"usage"`
	FinishReason     string `yaml:"finish_reason" json:"finish_reason"`
	ToolCalls        string `yaml:"tool_calls" json:"tool_calls"`
	ReasoningContent string `yaml:"reasoning_content" json:"reasoning_content"`
	Reasoning        string `yaml:"reasoning" json:"reasoning"`
}

// ResponsePathsFor returns manifest response_paths, if any.
func ResponsePathsFor(m any) *ResponsePathsConfig {
	switch v := m.(type) {
	case *V1Manifest:
		return v.ResponsePaths
	case *V2Manifest:
		return v.ResponsePaths
	default:
		return nil
	}
}
