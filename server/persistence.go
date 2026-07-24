package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const manifestVersion = 1

type manifest struct {
	Version      int    `json:"version"`
	ID           string `json:"id"`
	Emotion      string `json:"emotion"`
	Seed         uint64 `json:"seed"`
	CreatedAt    int64  `json:"createdAt"`
	StartedAt    int64  `json:"startedAt"`
	FinishedAt   int64  `json:"finishedAt"`
	DurationMS   int64  `json:"durationMs"`
	ClipSamples  int64  `json:"clipSamples"`
	TotalFrames  int    `json:"totalFrames"`
	PortraitType string `json:"portraitType"`
	AudioType    string `json:"audioType"`
	AudioName    string `json:"audioName,omitempty"`
}

func writeManifest(dir string, j *job) error {
	j.mu.Lock()
	m := manifest{
		Version: manifestVersion, ID: j.ID, Emotion: j.Emotion, Seed: j.Seed,
		CreatedAt: j.CreatedAt, StartedAt: j.StartedAt, FinishedAt: j.FinishedAt,
		DurationMS: j.DurationMS, ClipSamples: j.ClipSamples, TotalFrames: j.TotalFrames,
		PortraitType: j.PortraitType, AudioType: j.AudioType, AudioName: j.AudioName,
	}
	j.mu.Unlock()
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("encode manifest: %w", err)
	}
	data = append(data, '\n')
	path := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

func (s *server) restore() (int, error) {
	entries, err := os.ReadDir(s.store)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		dir := filepath.Join(s.store, entry.Name())
		j, err := loadManifest(dir)
		if err != nil {
			log.Printf("ignoring saved job %s: %v", entry.Name(), err)
			continue
		}
		s.jobs[j.ID] = j
		count++
	}
	return count, nil
}

func loadManifest(dir string) (*job, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	if m.Version != manifestVersion {
		return nil, fmt.Errorf("unsupported manifest version %d", m.Version)
	}
	if m.ID == "" || m.ID != filepath.Base(dir) || filepath.Base(m.ID) != m.ID {
		return nil, fmt.Errorf("invalid job id %q", m.ID)
	}
	for _, name := range []string{"portrait.bin", "audio-original.bin", "audio.wav", "audio-normalized.wav", "result.mp4"} {
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil || !info.Mode().IsRegular() {
			if err == nil {
				err = fmt.Errorf("not a regular file")
			}
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	return &job{
		ID: m.ID, State: "done", Phase: "complete", Emotion: m.Emotion,
		CreatedAt: m.CreatedAt, StartedAt: m.StartedAt, FinishedAt: m.FinishedAt,
		DurationMS: m.DurationMS, ClipSamples: m.ClipSamples,
		Frame: m.TotalFrames, TotalFrames: m.TotalFrames, Seed: m.Seed,
		PortraitType: m.PortraitType, AudioType: m.AudioType, AudioName: m.AudioName,
		Dir: dir,
	}, nil
}
