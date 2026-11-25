package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAdminDecisions(t *testing.T) {
	compatConfig(t, compatFixture)
	before, _ := readConfigBytes()
	if e := os.Mkdir(filepath.Dir(rulesPath()), 0700); e != nil {
		t.Fatal(e)
	}
	get := func() string {
		s, b := panelModels("GET", nil)
		var r struct{ Revision string }
		if s != 200 || json.Unmarshal(b, &r) != nil || bytes.Contains(b, []byte("unit-silicon")) || bytes.Contains(b, []byte("https://")) {
			t.Fatal("invalid snapshot")
		}
		return r.Revision
	}
	post := func(rev, mode string) int {
		b, _ := json.Marshal(map[string]string{"revision": rev, "name": compatTestModel, "channel": "siliconflow-embeddings", "mode": mode})
		s, _ := panelModels("POST", b)
		return s
	}
	r := get()
	if post(r, "deny") != 200 || post(r, "allow") != 409 {
		t.Fatal("stale write")
	}
	cfg, _, e := activeCatalog()
	if _, _, ok := resolveCatalog(cfg, compatTestModel); e != nil || ok {
		t.Fatal("deny not active")
	}
	if post(get(), "auto") != 200 {
		t.Fatal("auto")
	}
	cfg, _, _ = activeCatalog()
	if _, _, ok := resolveCatalog(cfg, compatTestModel); !ok {
		t.Fatal("not restored")
	}
	st, e := os.Stat(rulesPath())
	after, _ := readConfigBytes()
	if e != nil || st.Mode().Perm() != 0600 || !bytes.Equal(before, after) {
		t.Fatal("unsafe persistence")
	}
}
