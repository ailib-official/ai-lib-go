package protocol

import "testing"

func TestBuildAuthMetadataNormalizesBearerPrefixSeparator(t *testing.T) {
	metadata := BuildAuthMetadata(&V2Manifest{
		ID: "prefixauth",
		Endpoint: EndpointConfig{
			Auth: &V2Auth{
				Type:   "bearer",
				Prefix: "Bearer",
			},
		},
	}, ResolvedCredential{
		Value:      "secret",
		SourceKind: CredentialSourceExplicit,
		SourceName: "explicit",
	}, false)

	if got := metadata.Headers["Authorization"]; got != "Bearer secret" {
		t.Fatalf("Authorization header expected %q got %q", "Bearer secret", got)
	}
}

func TestShadowedAuthReportsDivergentV2CoreAuth(t *testing.T) {
	shadowed, ok := ShadowedAuth(&V2Manifest{
		ID: "dualauth",
		Endpoint: EndpointConfig{
			Auth: &V2Auth{
				Type:     "bearer",
				TokenEnv: "DUALAUTH_API_TOKEN",
			},
		},
		Core: &V2CoreLegacy{
			Auth: V2Auth{
				Type:   "api_key",
				KeyEnv: "DUALAUTH_LEGACY_KEY",
			},
		},
	})
	if !ok {
		t.Fatal("expected V2 core auth to be reported as shadowed")
	}
	if shadowed.KeyEnv != "DUALAUTH_LEGACY_KEY" {
		t.Fatalf("shadowed KeyEnv expected %q got %q", "DUALAUTH_LEGACY_KEY", shadowed.KeyEnv)
	}
}
