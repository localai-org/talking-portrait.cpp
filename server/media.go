package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"
)

const (
	sampleRate     = 16000
	maxClipSeconds = 30
)

var waveExtensibleGUIDTail = []byte{
	0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
}

// transcodeAudio converts any ffmpeg-supported input to the exact WAV shape
// accepted by libtp. Decoding 40 ms beyond the public limit makes over-limit
// compressed inputs distinguishable from a valid clip ending at 30 seconds.
func transcodeAudio(ctx context.Context, source, wav string) error {
	cmd := exec.CommandContext(ctx, "ffmpeg", "-nostdin", "-v", "error", "-y",
		"-i", source, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
		"-c:a", "pcm_f32le", "-t", "30.04", wav)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return fmt.Errorf("audio decoding timed out")
		}
		return fmt.Errorf("could not decode audio: %s", compactError(stderr.String(), err))
	}
	n, err := wavSampleCount(wav)
	if err != nil {
		return err
	}
	if n <= 0 {
		return fmt.Errorf("audio contains no samples")
	}
	if n > sampleRate*maxClipSeconds {
		return fmt.Errorf("audio exceeds the %d second limit", maxClipSeconds)
	}
	return nil
}

func wavSampleCount(path string) (int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read converted audio: %w", err)
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return 0, fmt.Errorf("ffmpeg produced an invalid WAV")
	}
	var channels, bits uint16
	var format uint16
	var dataBytes uint32
	for off := 12; off+8 <= len(data); {
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		start := off + 8
		if start+size > len(data) {
			return 0, fmt.Errorf("truncated converted WAV")
		}
		switch string(data[off : off+4]) {
		case "fmt ":
			if size >= 16 {
				format = binary.LittleEndian.Uint16(data[start : start+2])
				channels = binary.LittleEndian.Uint16(data[start+2 : start+4])
				bits = binary.LittleEndian.Uint16(data[start+14 : start+16])
				// ffmpeg uses WAVE_FORMAT_EXTENSIBLE for pcm_f32le. Its
				// sub-format GUID carries the legacy PCM (1) or IEEE float
				// (3) tag in the first four bytes.
				if format == 0xfffe && size >= 40 {
					subFormat := data[start+24 : start+40]
					if subFormat[2] == 0 && subFormat[3] == 0 &&
						bytes.Equal(subFormat[4:], waveExtensibleGUIDTail) {
						format = binary.LittleEndian.Uint16(subFormat[:2])
					}
				}
			}
		case "data":
			dataBytes = uint32(size)
		}
		off = start + size + size%2
	}
	if channels == 0 || bits == 0 || dataBytes == 0 || (format != 1 && format != 3) {
		return 0, fmt.Errorf("converted WAV has unsupported metadata")
	}
	return int64(dataBytes) / int64(channels) / int64(bits/8), nil
}

func writeFloatWAV(path string, samples []float32) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("create normalized WAV: %w", err)
	}
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	dataSize := uint32(len(samples) * 4)
	if _, err := f.Write([]byte("RIFF")); err != nil {
		return err
	}
	for _, value := range []any{
		uint32(36) + dataSize, [4]byte{'W', 'A', 'V', 'E'}, [4]byte{'f', 'm', 't', ' '},
		uint32(16), uint16(3), uint16(1), uint32(sampleRate), uint32(sampleRate * 4),
		uint16(4), uint16(32), [4]byte{'d', 'a', 't', 'a'}, dataSize,
	} {
		if err := binary.Write(f, binary.LittleEndian, value); err != nil {
			return fmt.Errorf("write normalized WAV header: %w", err)
		}
	}
	if err := binary.Write(f, binary.LittleEndian, samples); err != nil {
		return fmt.Errorf("write normalized WAV samples: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync normalized WAV: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close normalized WAV: %w", err)
	}
	ok = true
	return nil
}

type videoEncoder struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stderr bytes.Buffer
	rgb    []byte
}

func startVideoEncoder(ctx context.Context, wav, output string) (*videoEncoder, error) {
	v := &videoEncoder{rgb: make([]byte, 3*framePixels)}
	v.cmd = exec.CommandContext(ctx, "ffmpeg", "-nostdin", "-v", "error", "-y",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", "512x512", "-r", "25",
		"-i", "pipe:0", "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
		"-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
		"-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
		"-movflags", "+faststart", "-shortest", output)
	v.cmd.Stderr = &v.stderr
	var err error
	v.stdin, err = v.cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("open video encoder: %w", err)
	}
	if err := v.cmd.Start(); err != nil {
		return nil, fmt.Errorf("start video encoder: %w", err)
	}
	return v, nil
}

func (v *videoEncoder) writeFrame(chw []float32) error {
	for i := 0; i < framePixels; i++ {
		v.rgb[3*i] = sampleByte(chw[i])
		v.rgb[3*i+1] = sampleByte(chw[framePixels+i])
		v.rgb[3*i+2] = sampleByte(chw[2*framePixels+i])
	}
	if _, err := v.stdin.Write(v.rgb); err != nil {
		return fmt.Errorf("encode video frame: %w", err)
	}
	return nil
}

func (v *videoEncoder) finish() error {
	closeErr := v.stdin.Close()
	waitErr := v.cmd.Wait()
	if waitErr != nil {
		return fmt.Errorf("video encoding failed: %s", compactError(v.stderr.String(), waitErr))
	}
	if closeErr != nil {
		return fmt.Errorf("finish video stream: %w", closeErr)
	}
	return nil
}

func (v *videoEncoder) abort() {
	_ = v.stdin.Close()
	if v.cmd.Process != nil {
		_ = v.cmd.Process.Kill()
	}
	_ = v.cmd.Wait()
}

func sampleByte(v float32) byte {
	if v < -1 {
		v = -1
	} else if v > 1 {
		v = 1
	}
	return byte((v + 1) * 127.5)
}

func compactError(stderr string, fallback error) string {
	if len(stderr) > 400 {
		stderr = stderr[:400]
	}
	if stderr != "" {
		return stderr
	}
	return fallback.Error()
}

func mediaTimeout() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 2*time.Minute)
}
