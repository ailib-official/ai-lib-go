package protocol

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type CredentialSourceKind string

const (
	CredentialSourceExplicit        CredentialSourceKind = "explicit"
	CredentialSourceManifestEnv     CredentialSourceKind = "manifest_env"
	CredentialSourceConventionalEnv CredentialSourceKind = "conventional_env"
	CredentialSourceKeyring         CredentialSourceKind = "keyring"
	CredentialSourceNone            CredentialSourceKind = "none"
)

type ResolvedCredential struct {
	Value               string
	SourceKind          CredentialSourceKind
	SourceName          string
	RequiredEnvVars     []string
	ConventionalEnvVars []string
}

type AuthMetadata struct {
	Headers     map[string]string
	QueryParams map[string]string
	SourceKind  CredentialSourceKind
	SourceName  string
}

const RedactedCredential = "<redacted>"

type authConfig struct {
	Type      string
	Header    string
	Prefix    string
	TokenEnv  string
	KeyEnv    string
	EnvVar    string
	ParamName string
}

func ResolveCredential(m any, explicit string) ResolvedCredential {
	required := RequiredEnvVars(m)
	conventional := ConventionalEnvVars(m)
	if explicit != "" {
		return ResolvedCredential{
			Value:               explicit,
			SourceKind:          CredentialSourceExplicit,
			SourceName:          "explicit",
			RequiredEnvVars:     required,
			ConventionalEnvVars: conventional,
		}
	}
	for _, name := range required {
		if v := os.Getenv(name); v != "" {
			return ResolvedCredential{
				Value:               v,
				SourceKind:          CredentialSourceManifestEnv,
				SourceName:          name,
				RequiredEnvVars:     required,
				ConventionalEnvVars: conventional,
			}
		}
	}
	for _, name := range conventional {
		if v := os.Getenv(name); v != "" {
			return ResolvedCredential{
				Value:               v,
				SourceKind:          CredentialSourceConventionalEnv,
				SourceName:          name,
				RequiredEnvVars:     required,
				ConventionalEnvVars: conventional,
			}
		}
	}
	return ResolvedCredential{
		SourceKind:          CredentialSourceNone,
		RequiredEnvVars:     required,
		ConventionalEnvVars: conventional,
	}
}

func BuildAuthMetadata(m any, credential ResolvedCredential, redacted bool) AuthMetadata {
	headers := map[string]string{}
	query := map[string]string{}
	auth, ok := PrimaryAuth(m)
	if !ok || credential.Value == "" {
		return AuthMetadata{Headers: headers, QueryParams: query, SourceKind: credential.SourceKind, SourceName: credential.SourceName}
	}
	value := credential.Value
	if redacted {
		value = RedactedCredential
	}
	switch auth.Type {
	case "query_param":
		name := firstNonEmpty(auth.ParamName, "api_key")
		query[name] = value
	case "api_key", "custom_header", "header":
		name := firstNonEmpty(auth.Header, "x-api-key")
		headers[name] = value
	case "bearer":
		fallthrough
	default:
		name := firstNonEmpty(auth.Header, "Authorization")
		prefix := auth.Prefix
		if prefix == "" {
			prefix = "Bearer "
		}
		headers[name] = prefix + value
	}
	return AuthMetadata{Headers: headers, QueryParams: query, SourceKind: credential.SourceKind, SourceName: credential.SourceName}
}

func PrimaryAuth(m any) (authConfig, bool) {
	switch v := m.(type) {
	case *V1Manifest:
		if v.Endpoint.Auth != nil {
			return authFromV2(*v.Endpoint.Auth), true
		}
		if v.Auth != nil {
			return authFromV1(*v.Auth), true
		}
	case *V2Manifest:
		if v.Endpoint.Auth != nil {
			return authFromV2(*v.Endpoint.Auth), true
		}
		if v.Core != nil {
			return authFromV2(v.Core.Auth), true
		}
	}
	return authConfig{}, false
}

func ShadowedAuth(m any) (authConfig, bool) {
	switch v := m.(type) {
	case *V1Manifest:
		if v.Endpoint.Auth == nil || v.Auth == nil {
			return authConfig{}, false
		}
		endpoint := authFromV2(*v.Endpoint.Auth)
		topLevel := authFromV1(*v.Auth)
		if authSignature(endpoint) == authSignature(topLevel) {
			return authConfig{}, false
		}
		return topLevel, true
	}
	return authConfig{}, false
}

func RequiredEnvVars(m any) []string {
	auth, ok := PrimaryAuth(m)
	if !ok {
		return nil
	}
	return uniqueNonEmpty(auth.TokenEnv, auth.KeyEnv, auth.EnvVar)
}

func ConventionalEnvVars(m any) []string {
	id := ProviderID(m)
	if id == "" {
		return nil
	}
	return []string{strings.ToUpper(strings.ReplaceAll(id, "-", "_")) + "_API_KEY"}
}

func ProviderID(m any) string {
	switch v := m.(type) {
	case *V1Manifest:
		return v.ID
	case *V2Manifest:
		return v.ID
	default:
		return ""
	}
}

func CredentialDiagnostic(m any, credential ResolvedCredential) string {
	parts := []string{fmt.Sprintf("Credential missing for provider %s", ProviderID(m))}
	if len(credential.RequiredEnvVars) > 0 {
		parts = append(parts, "required env: "+strings.Join(credential.RequiredEnvVars, ", "))
	}
	if len(credential.ConventionalEnvVars) > 0 {
		parts = append(parts, "conventional fallback: "+strings.Join(credential.ConventionalEnvVars, ", "))
	}
	return strings.Join(parts, "; ")
}

func authFromV1(a V1Auth) authConfig {
	return authConfig{
		Type:      a.Type,
		Header:    firstNonEmpty(a.Header, a.HeaderName),
		Prefix:    a.Prefix,
		TokenEnv:  a.TokenEnv,
		KeyEnv:    a.KeyEnv,
		EnvVar:    a.EnvVar,
		ParamName: a.ParamName,
	}
}

func authFromV2(a V2Auth) authConfig {
	return authConfig{
		Type:      a.Type,
		Header:    firstNonEmpty(a.Header, a.HeaderName, a.Key),
		Prefix:    a.Prefix,
		TokenEnv:  a.TokenEnv,
		KeyEnv:    a.KeyEnv,
		EnvVar:    a.EnvVar,
		ParamName: a.ParamName,
	}
}

func authSignature(auth authConfig) string {
	b, _ := json.Marshal(auth)
	return string(b)
}

func uniqueNonEmpty(values ...string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
