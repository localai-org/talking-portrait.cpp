// talking-portrait demo server: portrait + speech -> a persistent talking MP4.
//
//	cmake -B build-cuda-shared -G Ninja -DTP_SHARED=ON -DTP_CUDA=ON
//	cmake --build build-cuda-shared -j
//	cd server && go build .
//	./talking-portrait-server -lib ../build-cuda-shared/libtp.so \
//	    -ggufs /path/to/ggufs
package main

import (
	"crypto/rand"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed web
var embeddedWeb embed.FS

const (
	maxPortrait = 32 << 20
	maxAudio    = 32 << 20
	maxRequest  = maxPortrait + maxAudio + (2 << 20)
)

var emotions = []string{"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}

type job struct {
	mu sync.Mutex

	ID          string `json:"id"`
	State       string `json:"state"` // processing | done | error
	Phase       string `json:"phase,omitempty"`
	Emotion     string `json:"emotion"`
	CreatedAt   int64  `json:"createdAt"`
	StartedAt   int64  `json:"startedAt,omitempty"`
	FinishedAt  int64  `json:"finishedAt,omitempty"`
	DurationMS  int64  `json:"durationMs,omitempty"`
	ClipSamples int64  `json:"clipSamples,omitempty"`
	Frame       int    `json:"frame"`
	TotalFrames int    `json:"totalFrames"`
	Error       string `json:"error,omitempty"`

	Seed         uint64 `json:"-"`
	PortraitType string `json:"-"`
	AudioType    string `json:"-"`
	AudioName    string `json:"-"`
	Dir          string `json:"-"`
}

type server struct {
	engine *engine
	store  string
	device string
	ftype  string

	mu     sync.Mutex
	active bool
	jobs   map[string]*job
}

func newServer(e *engine, store, device, ftype string) *server {
	return &server{engine: e, store: store, device: device, ftype: ftype, jobs: map[string]*job{}}
}

func (s *server) claim() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active {
		return false
	}
	s.active = true
	return true
}

func (s *server) release() {
	s.mu.Lock()
	s.active = false
	s.mu.Unlock()
}

func (s *server) isBusy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}

func (s *server) handleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, map[string]any{
		"backend": s.device, "ftype": s.ftype, "busy": s.isBusy(),
		"maxClipSeconds": maxClipSeconds, "emotions": emotions,
	})
}

func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if !s.claim() {
		http.Error(w, "the GPU is busy; wait for the current portrait to finish", http.StatusConflict)
		return
	}
	claimed := true
	defer func() {
		if claimed {
			s.release()
		}
	}()

	r.Body = http.MaxBytesReader(w, r.Body, maxRequest)
	if err := r.ParseMultipartForm(maxRequest); err != nil {
		http.Error(w, "invalid or oversized upload", http.StatusBadRequest)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	portrait, portraitType, _, err := readUpload(r, "portrait", maxPortrait)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	audio, audioType, audioName, err := readUpload(r, "audio", maxAudio)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	emotion := r.FormValue("emotion")
	emotionIndex := indexOf(emotions, emotion)
	if emotionIndex < 0 {
		http.Error(w, "unknown emotion", http.StatusBadRequest)
		return
	}

	id, err := randomID()
	if err != nil {
		http.Error(w, "could not allocate a job", http.StatusInternalServerError)
		return
	}
	seed, err := randomSeed()
	if err != nil {
		http.Error(w, "could not allocate a seed", http.StatusInternalServerError)
		return
	}
	now := time.Now().UnixMilli()
	j := &job{
		ID: id, State: "processing", Phase: "validating media", Emotion: emotion,
		CreatedAt: now, StartedAt: now, Seed: seed,
		PortraitType: portraitType, AudioType: audioType, AudioName: audioName,
	}
	s.mu.Lock()
	s.jobs[id] = j
	s.mu.Unlock()

	claimed = false
	go s.process(j, portrait, audio, emotionIndex)
	writeJSONStatus(w, http.StatusAccepted, map[string]string{"job": id})
}

func (s *server) process(j *job, portrait, audio []byte, emotion int) {
	defer s.release()
	start := time.Now()

	tmp, err := os.MkdirTemp(s.store, "."+j.ID+".tmp-")
	if err != nil {
		s.fail(j, fmt.Errorf("create job storage: %w", err))
		return
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(tmp)
		}
	}()

	if err := os.WriteFile(filepath.Join(tmp, "portrait.bin"), portrait, 0o644); err != nil {
		s.fail(j, fmt.Errorf("save portrait: %w", err))
		return
	}
	if err := os.WriteFile(filepath.Join(tmp, "audio-original.bin"), audio, 0o644); err != nil {
		s.fail(j, fmt.Errorf("save original audio: %w", err))
		return
	}

	j.setPhase("decoding audio")
	ctx, cancel := mediaTimeout()
	defer cancel()
	originalPath := filepath.Join(tmp, "audio-original.bin")
	wavPath := filepath.Join(tmp, "audio.wav")
	if err := transcodeAudio(ctx, originalPath, wavPath); err != nil {
		s.fail(j, err)
		return
	}
	wav, err := os.ReadFile(wavPath)
	if err != nil {
		s.fail(j, fmt.Errorf("read canonical audio: %w", err))
		return
	}

	j.setPhase("preparing portrait")
	in, err := s.engine.prepare(portrait, wav)
	if err != nil {
		s.fail(j, err)
		return
	}
	defer s.engine.releaseInput(in)
	if err := writeFloatWAV(filepath.Join(tmp, "audio-normalized.wav"), s.engine.normalizedSamples(in)); err != nil {
		s.fail(j, err)
		return
	}
	total := int(s.engine.nFrames(in.samples))
	j.mu.Lock()
	j.ClipSamples = in.samples
	j.TotalFrames = total
	j.Phase = "generating motion"
	j.mu.Unlock()

	videoPath := filepath.Join(tmp, "result.mp4")
	encoder, err := startVideoEncoder(ctx, wavPath, videoPath)
	if err != nil {
		s.fail(j, err)
		return
	}
	var callbackErr error
	n, generateErr := s.engine.generate(in, emotion, j.Seed, func(index int64, frame []float32) int32 {
		if err := encoder.writeFrame(frame); err != nil {
			callbackErr = err
			return 1
		}
		j.mu.Lock()
		j.Phase = "encoding frames"
		j.Frame = int(index) + 1
		j.mu.Unlock()
		return 0
	})
	if generateErr != nil || callbackErr != nil {
		encoder.abort()
		if callbackErr != nil {
			s.fail(j, callbackErr)
		} else {
			s.fail(j, generateErr)
		}
		return
	}
	if n != total {
		encoder.abort()
		s.fail(j, fmt.Errorf("inference returned %d frames; expected %d", n, total))
		return
	}

	j.setPhase("finalizing video")
	if err := encoder.finish(); err != nil {
		s.fail(j, err)
		return
	}
	j.mu.Lock()
	j.FinishedAt = time.Now().UnixMilli()
	j.DurationMS = time.Since(start).Milliseconds()
	j.Frame = total
	j.TotalFrames = total
	j.Phase = "saving result"
	j.mu.Unlock()
	if err := writeManifest(tmp, j); err != nil {
		s.fail(j, err)
		return
	}

	final := filepath.Join(s.store, j.ID)
	if err := os.Rename(tmp, final); err != nil {
		s.fail(j, fmt.Errorf("commit result: %w", err))
		return
	}
	committed = true
	j.mu.Lock()
	j.Dir = final
	j.State = "done"
	j.Phase = "complete"
	j.mu.Unlock()
}

func (s *server) fail(j *job, err error) {
	log.Printf("job %s: %v", j.ID, err)
	j.mu.Lock()
	j.State = "error"
	j.Phase = "failed"
	j.Error = err.Error()
	j.FinishedAt = time.Now().UnixMilli()
	j.mu.Unlock()
}

func (j *job) setPhase(phase string) {
	j.mu.Lock()
	j.Phase = phase
	j.mu.Unlock()
}

func (s *server) handleJob(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r.URL.Path, "/api/job/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	if r.Method == http.MethodDelete {
		j.mu.Lock()
		state, dir := j.State, j.Dir
		j.mu.Unlock()
		if state == "processing" {
			http.Error(w, "job is still active", http.StatusConflict)
			return
		}
		if dir != "" {
			if err := os.RemoveAll(dir); err != nil {
				http.Error(w, "delete job: "+err.Error(), http.StatusInternalServerError)
				return
			}
		}
		s.mu.Lock()
		delete(s.jobs, j.ID)
		s.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "GET or DELETE only", http.StatusMethodNotAllowed)
		return
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	writeJSON(w, j)
}

type jobSummary struct {
	ID          string `json:"id"`
	Emotion     string `json:"emotion"`
	CreatedAt   int64  `json:"createdAt"`
	DurationMS  int64  `json:"durationMs"`
	ClipSamples int64  `json:"clipSamples"`
}

func (s *server) handleJobs(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	all := make([]*job, 0, len(s.jobs))
	for _, j := range s.jobs {
		all = append(all, j)
	}
	s.mu.Unlock()

	if r.Method == http.MethodDelete {
		deleted := 0
		for _, j := range all {
			j.mu.Lock()
			state, dir := j.State, j.Dir
			j.mu.Unlock()
			if state == "processing" {
				continue
			}
			if dir != "" {
				if err := os.RemoveAll(dir); err != nil {
					http.Error(w, "delete history: "+err.Error(), http.StatusInternalServerError)
					return
				}
			}
			s.mu.Lock()
			delete(s.jobs, j.ID)
			s.mu.Unlock()
			deleted++
		}
		writeJSON(w, map[string]int{"deleted": deleted})
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "GET or DELETE only", http.StatusMethodNotAllowed)
		return
	}
	out := make([]jobSummary, 0, len(all))
	for _, j := range all {
		j.mu.Lock()
		if j.State == "done" {
			out = append(out, jobSummary{
				ID: j.ID, Emotion: j.Emotion, CreatedAt: j.CreatedAt,
				DurationMS: j.DurationMS, ClipSamples: j.ClipSamples,
			})
		}
		j.mu.Unlock()
	}
	sort.Slice(out, func(i, k int) bool { return out[i].CreatedAt > out[k].CreatedAt })
	writeJSON(w, out)
}

func (s *server) handleMedia(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", http.StatusMethodNotAllowed)
			return
		}
		j := s.getJob(r.URL.Path, "/api/"+kind+"/")
		if j == nil {
			http.Error(w, "no such job", http.StatusNotFound)
			return
		}
		j.mu.Lock()
		state, dir := j.State, j.Dir
		portraitType, audioType, audioName := j.PortraitType, j.AudioType, j.AudioName
		j.mu.Unlock()
		if state != "done" || dir == "" {
			http.Error(w, "media is not ready", http.StatusConflict)
			return
		}
		name, contentType := "", ""
		switch kind {
		case "video":
			name, contentType = "result.mp4", "video/mp4"
			w.Header().Set("Accept-Ranges", "bytes")
			if r.URL.Query().Get("download") == "1" {
				w.Header().Set("Content-Disposition", `attachment; filename="talking-portrait.mp4"`)
			}
		case "portrait":
			name, contentType = "portrait.bin", portraitType
		case "audio":
			if r.URL.Query().Get("kind") == "normalized" {
				name, contentType = "audio-normalized.wav", "audio/wav"
			} else if r.URL.Query().Get("kind") == "wav" {
				name, contentType = "audio.wav", "audio/wav"
			} else {
				name, contentType = "audio-original.bin", audioType
				if audioName != "" {
					w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=%q", filepath.Base(audioName)))
				}
			}
		}
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
		http.ServeFile(w, r, filepath.Join(dir, name))
	}
}

func (s *server) getJob(path, prefix string) *job {
	id := strings.TrimPrefix(path, prefix)
	if id == path || id == "" || strings.Contains(id, "/") {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

func readUpload(r *http.Request, field string, limit int64) ([]byte, string, string, error) {
	f, hdr, err := r.FormFile(field)
	if err != nil {
		return nil, "", "", fmt.Errorf("%s is required", field)
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, "", "", fmt.Errorf("read %s: %w", field, err)
	}
	if len(data) == 0 {
		return nil, "", "", fmt.Errorf("%s is empty", field)
	}
	if int64(len(data)) > limit {
		return nil, "", "", fmt.Errorf("%s exceeds %d MiB", field, limit>>20)
	}
	contentType := http.DetectContentType(data)
	if hdr.Header.Get("Content-Type") != "" {
		if mt, _, err := mime.ParseMediaType(hdr.Header.Get("Content-Type")); err == nil {
			contentType = mt
		}
	}
	return data, contentType, hdr.Filename, nil
}

func indexOf(values []string, target string) int {
	for i, value := range values {
		if value == target {
			return i
		}
	}
	return -1
}

func randomID() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

func randomSeed() (uint64, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func writeJSON(w http.ResponseWriter, value any) {
	writeJSONStatus(w, http.StatusOK, value)
}

func writeJSONStatus(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func main() {
	libPath := flag.String("lib", "../build-cuda-shared/libtp.so", "path to libtp.so")
	ggufDir := flag.String("ggufs", "../ggufs", "directory containing talking-portrait GGUFs")
	device := flag.String("device", "cuda", "libtp device: cpu, gpu, or cuda")
	ftype := flag.String("ftype", "f16", "model precision: f16 or f32")
	threads := flag.Int("threads", 0, "CPU thread count (0 = automatic)")
	store := flag.String("store", "../generations", "persistent generation directory")
	addr := flag.String("addr", ":4001", "HTTP listen address")
	flag.Parse()

	if err := os.MkdirAll(*store, 0o755); err != nil {
		log.Fatal(err)
	}
	e, err := openEngine(*libPath, *ggufDir, *device, *ftype, *threads)
	if err != nil {
		log.Fatal(err)
	}
	defer e.close()

	s := newServer(e, *store, *device, *ftype)
	if n, err := s.restore(); err != nil {
		log.Printf("restore history: %v", err)
	} else if n > 0 {
		log.Printf("restored %d completed portrait(s)", n)
	}

	webRoot, err := fs.Sub(embeddedWeb, "web")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	staticFiles := http.FileServer(http.FS(webRoot))
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/picture-frame.glb" {
			// The generated PBR frame is large and shared by every portrait.
			// Let browsers reuse it across page loads while keeping iteration
			// reasonably quick when the checked-in model is replaced.
			w.Header().Set("Cache-Control", "public, max-age=3600")
			w.Header().Set("Content-Type", "model/gltf-binary")
		} else if r.URL.Path == "/wallpaper.png" {
			w.Header().Set("Cache-Control", "public, max-age=3600")
		}
		staticFiles.ServeHTTP(w, r)
	}))
	mux.HandleFunc("/api/info", s.handleInfo)
	mux.HandleFunc("/api/generate", s.handleGenerate)
	mux.HandleFunc("/api/jobs", s.handleJobs)
	mux.HandleFunc("/api/job/", s.handleJob)
	mux.HandleFunc("/api/video/", s.handleMedia("video"))
	mux.HandleFunc("/api/portrait/", s.handleMedia("portrait"))
	mux.HandleFunc("/api/audio/", s.handleMedia("audio"))

	log.Printf("talking-portrait demo listening on http://%s (%s/%s)", *addr, *device, *ftype)
	if err := http.ListenAndServe(*addr, mux); !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
