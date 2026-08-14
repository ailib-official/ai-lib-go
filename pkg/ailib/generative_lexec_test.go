package ailib

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIAndDashscopeBodiesDiffer(t *testing.T) {
	req := ImageGenerationRequest{Model: "m", Prompt: "a cat"}
	oai := OpenAIImageBody(req)
	ds := DashscopeImageBody(req)
	if oai["prompt"] != "a cat" {
		t.Fatalf("openai prompt: %v", oai["prompt"])
	}
	if _, ok := oai["input"]; ok {
		t.Fatal("openai body must not have input")
	}
	if _, ok := ds["prompt"]; ok {
		t.Fatal("dashscope body must not have top-level prompt")
	}
	input := ds["input"].(map[string]any)
	msgs := input["messages"].([]map[string]any)
	content := msgs[0]["content"].([]map[string]any)
	if content[0]["text"] != "a cat" {
		t.Fatalf("dashscope text: %v", content[0]["text"])
	}
}

func TestParseDashscopeImageShapes(t *testing.T) {
	viaChoices := ParseDashscopeImage("m", map[string]any{
		"output": map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"content": []any{map[string]any{"image": "https://img/a.png"}},
					},
				},
			},
		},
	})
	if len(viaChoices.Images) != 1 || viaChoices.Images[0].URL == nil || *viaChoices.Images[0].URL != "https://img/a.png" {
		t.Fatalf("choices parse: %+v", viaChoices)
	}
	viaResults := ParseDashscopeImage("m", map[string]any{
		"output": map[string]any{
			"results": []any{map[string]any{"url": "https://img/b.png"}},
		},
	})
	if len(viaResults.Images) != 1 || *viaResults.Images[0].URL != "https://img/b.png" {
		t.Fatalf("results parse: %+v", viaResults)
	}
}

func TestGenerateImageLExecHTTP(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"url": "https://cdn/x.png"}},
		})
	}))
	defer srv.Close()

	yaml := `
id: openai
protocol_version: "2.0"
endpoint:
  base_url: "` + srv.URL + `"
  auth:
    type: bearer
    token_env: OPENAI_API_KEY
endpoints:
  image_generation:
    path: "/images/generations"
    method: POST
    adapter: openai
metadata:
  models:
    gpt-image-1:
      model_capabilities:
        image_generation: true
`
	t.Setenv("OPENAI_API_KEY", "test-key")
	c, err := NewClientBuilder().WithProtocolData([]byte(yaml)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	res, err := c.GenerateImage(context.Background(), ImageGenerationRequest{
		Model:  "gpt-image-1",
		Prompt: "a cat",
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if gotPath != "/images/generations" {
		t.Fatalf("path: %s", gotPath)
	}
	if gotBody["prompt"] != "a cat" {
		t.Fatalf("body: %+v", gotBody)
	}
	if len(res.Images) != 1 || res.Images[0].URL == nil {
		t.Fatalf("result: %+v", res)
	}
}

func TestGenerateImageOmitFailClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("must not call HTTP when capability omitted")
	}))
	defer srv.Close()
	yaml := `
id: openai
protocol_version: "2.0"
endpoint:
  base_url: "` + srv.URL + `"
  auth:
    type: bearer
    token_env: OPENAI_API_KEY
endpoints:
  image_generation:
    path: "/images/generations"
metadata:
  models:
    gpt-4o:
      context_window: 128000
`
	t.Setenv("OPENAI_API_KEY", "test-key")
	c, err := NewClientBuilder().WithProtocolData([]byte(yaml)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()
	_, err = c.GenerateImage(context.Background(), ImageGenerationRequest{Model: "gpt-4o", Prompt: "x"})
	if err == nil || !strings.Contains(err.Error(), "omit") {
		t.Fatalf("expected omit error, got %v", err)
	}
}

func TestTranscribeSpeechMultipart(t *testing.T) {
	var gotCT string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotCT = r.Header.Get("Content-Type")
		_ = json.NewEncoder(w).Encode(map[string]any{"text": "hello"})
	}))
	defer srv.Close()
	yaml := `
id: openai
protocol_version: "2.0"
endpoint:
  base_url: "` + srv.URL + `"
  auth:
    type: bearer
    token_env: OPENAI_API_KEY
endpoints:
  speech_to_text:
    path: "/audio/transcriptions"
    adapter: openai
metadata:
  models:
    whisper-1:
      model_capabilities:
        speech_to_text: true
`
	t.Setenv("OPENAI_API_KEY", "test-key")
	c, err := NewClientBuilder().WithProtocolData([]byte(yaml)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()
	res, err := c.TranscribeSpeech(context.Background(), SpeechToTextRequest{
		Model: "whisper-1",
		Audio: []byte("fake-wav"),
	})
	if err != nil {
		t.Fatalf("TranscribeSpeech: %v", err)
	}
	if res.Text != "hello" {
		t.Fatalf("text: %s", res.Text)
	}
	if !strings.HasPrefix(gotCT, "multipart/form-data") {
		t.Fatalf("content-type: %s", gotCT)
	}
}
