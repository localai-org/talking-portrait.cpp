package main

// In-process binding to libtp.so. The frame callback runs synchronously on the
// inference thread, so a single active generation and a single callback sink
// are deliberate invariants.

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const tpABIVersion = 1
const framePixels = 512 * 512

type engine struct {
	mu sync.Mutex

	ctx uintptr

	free           func(uintptr)
	lastError      func(uintptr) string
	loadWAV        func(unsafe.Pointer, uintptr, *int64, unsafe.Pointer, uintptr) unsafe.Pointer
	loadImage      func(unsafe.Pointer, uintptr, unsafe.Pointer, uintptr) unsafe.Pointer
	normalizeAudio func(unsafe.Pointer, int64)
	bufferFree     func(unsafe.Pointer)
	nFrames        func(int64) int64
	generateCB     func(uintptr, unsafe.Pointer, unsafe.Pointer, int64, int32, unsafe.Pointer, uint64, uintptr, unsafe.Pointer) int32
}

var (
	frameSinkMu sync.Mutex
	frameSink   func(int64, []float32) int32
	frameCB     uintptr
)

func openEngine(libPath, ggufDir, device, ftype string, threads int) (*engine, error) {
	lib, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", libPath, err)
	}

	var abi func() int32
	var load func(string, string, int32, string) uintptr
	e := &engine{}
	purego.RegisterLibFunc(&abi, lib, "tp_abi_version")
	if got := abi(); got != tpABIVersion {
		return nil, fmt.Errorf("ABI mismatch: library reports %d, server expects %d", got, tpABIVersion)
	}
	purego.RegisterLibFunc(&load, lib, "tp_load")
	purego.RegisterLibFunc(&e.free, lib, "tp_free")
	purego.RegisterLibFunc(&e.lastError, lib, "tp_last_error")
	purego.RegisterLibFunc(&e.loadWAV, lib, "tp_load_wav")
	purego.RegisterLibFunc(&e.loadImage, lib, "tp_load_image")
	purego.RegisterLibFunc(&e.normalizeAudio, lib, "tp_normalize_audio")
	purego.RegisterLibFunc(&e.bufferFree, lib, "tp_buf_free")
	purego.RegisterLibFunc(&e.nFrames, lib, "tp_n_frames")
	purego.RegisterLibFunc(&e.generateCB, lib, "tp_generate_cb")

	e.ctx = load(ggufDir, device, int32(threads), ftype)
	if e.ctx == 0 {
		return nil, fmt.Errorf("tp_load failed; see server stderr for model/backend details")
	}

	if frameCB == 0 {
		frameCB = purego.NewCallback(func(index int64, chw unsafe.Pointer, _ unsafe.Pointer) int32 {
			frameSinkMu.Lock()
			sink := frameSink
			frameSinkMu.Unlock()
			if sink == nil || chw == nil {
				return 1
			}
			frame := unsafe.Slice((*float32)(chw), 3*framePixels)
			return sink(index, frame)
		})
	}
	return e, nil
}

func (e *engine) close() {
	if e != nil && e.ctx != 0 {
		e.free(e.ctx)
		e.ctx = 0
	}
}

type preparedInput struct {
	image   unsafe.Pointer
	audio   unsafe.Pointer
	samples int64
}

func (e *engine) prepare(portrait, wav []byte) (*preparedInput, error) {
	var errbuf [512]byte
	image := e.loadImage(unsafe.Pointer(&portrait[0]), uintptr(len(portrait)),
		unsafe.Pointer(&errbuf[0]), uintptr(len(errbuf)))
	if image == nil {
		return nil, fmt.Errorf("portrait: %s", cString(errbuf[:]))
	}
	var samples int64
	clear(errbuf[:])
	audio := e.loadWAV(unsafe.Pointer(&wav[0]), uintptr(len(wav)), &samples,
		unsafe.Pointer(&errbuf[0]), uintptr(len(errbuf)))
	if audio == nil {
		e.bufferFree(image)
		return nil, fmt.Errorf("audio: %s", cString(errbuf[:]))
	}
	e.normalizeAudio(audio, samples)
	return &preparedInput{image: image, audio: audio, samples: samples}, nil
}

func (e *engine) releaseInput(in *preparedInput) {
	if in == nil {
		return
	}
	e.bufferFree(in.image)
	e.bufferFree(in.audio)
}

func (e *engine) normalizedSamples(in *preparedInput) []float32 {
	if in == nil || in.audio == nil || in.samples <= 0 {
		return nil
	}
	src := unsafe.Slice((*float32)(in.audio), int(in.samples))
	return append([]float32(nil), src...)
}

func (e *engine) generate(in *preparedInput, emotion int, seed uint64,
	sink func(int64, []float32) int32) (int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	frameSinkMu.Lock()
	frameSink = sink
	frameSinkMu.Unlock()
	defer func() {
		frameSinkMu.Lock()
		frameSink = nil
		frameSinkMu.Unlock()
	}()

	n := e.generateCB(e.ctx, in.image, in.audio, in.samples, int32(emotion),
		nil, seed, frameCB, nil)
	if n < 0 {
		return 0, fmt.Errorf("%s", e.lastError(e.ctx))
	}
	return int(n), nil
}

func cString(buf []byte) string {
	for i, b := range buf {
		if b == 0 {
			return string(buf[:i])
		}
	}
	return string(buf)
}
