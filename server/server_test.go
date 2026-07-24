package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFloatWAVRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.wav")
	samples := []float32{-1, -.5, 0, .5, 1}
	if err := writeFloatWAV(path, samples); err != nil {
		t.Fatal(err)
	}
	n, err := wavSampleCount(path)
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(len(samples)) {
		t.Fatalf("sample count = %d, want %d", n, len(samples))
	}
}

func TestEmbeddedPictureFrameGLB(t *testing.T) {
	f, err := embeddedWeb.Open("web/picture-frame.glb")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	var header [12]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		t.Fatal(err)
	}
	if string(header[:4]) != "glTF" || binary.LittleEndian.Uint32(header[4:8]) != 2 {
		t.Fatal("embedded picture frame is not a GLB 2.0 asset")
	}
	if got := int64(binary.LittleEndian.Uint32(header[8:12])); got != info.Size() {
		t.Fatalf("GLB header length = %d, embedded size = %d", got, info.Size())
	}
}

func TestEmbeddedWallpaperPNG(t *testing.T) {
	f, err := embeddedWeb.Open("web/wallpaper.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var header [24]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		t.Fatal(err)
	}
	if string(header[:8]) != "\x89PNG\r\n\x1a\n" {
		t.Fatal("embedded wallpaper is not a PNG")
	}
	width := binary.BigEndian.Uint32(header[16:20])
	height := binary.BigEndian.Uint32(header[20:24])
	if width != 1024 || height != 1024 {
		t.Fatalf("wallpaper dimensions = %dx%d, want 1024x1024", width, height)
	}
}

func TestWAVSampleCountExtensibleFloat(t *testing.T) {
	var wav bytes.Buffer
	dataSize := uint32(2 * 4)
	for _, value := range []any{
		[4]byte{'R', 'I', 'F', 'F'}, uint32(60) + dataSize,
		[4]byte{'W', 'A', 'V', 'E'}, [4]byte{'f', 'm', 't', ' '},
		uint32(40), uint16(0xfffe), uint16(1), uint32(sampleRate),
		uint32(sampleRate * 4), uint16(4), uint16(32), uint16(22),
		uint16(32), uint32(4),
		[16]byte{3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71},
		[4]byte{'d', 'a', 't', 'a'}, dataSize,
		[2]float32{-0.5, 0.5},
	} {
		if err := binary.Write(&wav, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "extensible.wav")
	if err := os.WriteFile(path, wav.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	n, err := wavSampleCount(path)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("sample count = %d, want 2", n)
	}
}

func TestSampleByteClamps(t *testing.T) {
	tests := []struct {
		in   float32
		want byte
	}{{-2, 0}, {-1, 0}, {0, 127}, {1, 255}, {2, 255}}
	for _, test := range tests {
		if got := sampleByte(test.in); got != test.want {
			t.Errorf("sampleByte(%v) = %d, want %d", test.in, got, test.want)
		}
	}
}

func TestManifestRoundTripAndRestore(t *testing.T) {
	store := t.TempDir()
	id := "0123456789abcdef"
	dir := filepath.Join(store, id)
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"portrait.bin", "audio-original.bin", "audio.wav", "audio-normalized.wav", "result.mp4",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("asset"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	j := &job{
		ID: id, State: "done", Emotion: "happy", Seed: 42, CreatedAt: 10,
		StartedAt: 11, FinishedAt: 12, DurationMS: 2, ClipSamples: 32000,
		Frame: 50, TotalFrames: 50, PortraitType: "image/png",
		AudioType: "audio/webm", AudioName: "voice.webm",
	}
	if err := writeManifest(dir, j); err != nil {
		t.Fatal(err)
	}
	got, err := loadManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != id || got.Emotion != "happy" || got.TotalFrames != 50 || got.Dir != dir {
		t.Fatalf("unexpected restored job: %+v", got)
	}

	s := newServer(nil, store, "cuda", "f16")
	n, err := s.restore()
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || s.jobs[id] == nil {
		t.Fatalf("restore count=%d jobs=%v", n, s.jobs)
	}
}

func TestRestoreIgnoresTemporaryAndCorruptDirectories(t *testing.T) {
	store := t.TempDir()
	if err := os.Mkdir(filepath.Join(store, ".job.tmp-123"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(store, "bad"), 0o755); err != nil {
		t.Fatal(err)
	}
	s := newServer(nil, store, "cuda", "f16")
	n, err := s.restore()
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("restored %d invalid jobs", n)
	}
}

func TestInfoAndBusyHandlers(t *testing.T) {
	s := newServer(nil, t.TempDir(), "cuda", "f16")
	info := httptest.NewRecorder()
	s.handleInfo(info, httptest.NewRequest(http.MethodGet, "/api/info", nil))
	if info.Code != http.StatusOK {
		t.Fatalf("info status %d", info.Code)
	}
	var body map[string]any
	if err := json.NewDecoder(info.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["backend"] != "cuda" || body["maxClipSeconds"] != float64(30) {
		t.Fatalf("unexpected info: %v", body)
	}

	s.active = true
	busy := httptest.NewRecorder()
	s.handleGenerate(busy, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader("")))
	if busy.Code != http.StatusConflict {
		t.Fatalf("busy status %d, want 409", busy.Code)
	}
}

func TestReadUploadLimitsAndMetadata(t *testing.T) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, err := mw.CreateFormFile("audio", "clip.webm")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write([]byte("sample")); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if err := req.ParseMultipartForm(1024); err != nil {
		t.Fatal(err)
	}
	data, _, name, err := readUpload(req, "audio", 6)
	if err != nil || string(data) != "sample" || name != "clip.webm" {
		t.Fatalf("data=%q name=%q err=%v", data, name, err)
	}
	if _, _, _, err := readUpload(req, "audio", 5); err == nil {
		t.Fatal("expected upload limit error")
	}
}

func TestGetJobRejectsNestedPaths(t *testing.T) {
	s := newServer(nil, t.TempDir(), "cuda", "f16")
	s.jobs["good"] = &job{ID: "good"}
	if s.getJob("/api/job/good", "/api/job/") == nil {
		t.Fatal("valid job not found")
	}
	for _, path := range []string{"/api/job/", "/api/job/../good", "/other/good"} {
		if s.getJob(path, "/api/job/") != nil {
			t.Fatalf("accepted invalid path %q", path)
		}
	}
}

func TestFFmpegMediaPipeline(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg is not installed")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "source.wav")
	const nSamples = 4000
	if err := writeFloatWAV(source, make([]float32, nSamples)); err != nil {
		t.Fatal(err)
	}
	canonical := filepath.Join(dir, "canonical.wav")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := transcodeAudio(ctx, source, canonical); err != nil {
		t.Fatal(err)
	}
	n, err := wavSampleCount(canonical)
	if err != nil || n != nSamples {
		t.Fatalf("canonical samples=%d err=%v", n, err)
	}
	output := filepath.Join(dir, "result.mp4")
	encoder, err := startVideoEncoder(ctx, canonical, output)
	if err != nil {
		t.Fatal(err)
	}
	frame := make([]float32, 3*framePixels)
	for i := 0; i < 7; i++ {
		if err := encoder.writeFrame(frame); err != nil {
			encoder.abort()
			t.Fatal(err)
		}
	}
	if err := encoder.finish(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(output)
	if err != nil || info.Size() == 0 {
		t.Fatalf("output info=%v err=%v", info, err)
	}

	tooLong := filepath.Join(dir, "too-long.wav")
	if err := writeFloatWAV(tooLong, make([]float32, sampleRate*maxClipSeconds+800)); err != nil {
		t.Fatal(err)
	}
	err = transcodeAudio(ctx, tooLong, filepath.Join(dir, "too-long-canonical.wav"))
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("over-limit audio error = %v", err)
	}
}
