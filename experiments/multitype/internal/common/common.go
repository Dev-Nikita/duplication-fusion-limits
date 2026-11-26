// Package common: reproducible randomness, the episode clock and append-only JSONL output.
package common

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"os"
	"strconv"
	"sync"
	"time"
)

// Unit returns a reproducible number in [0,1) for a key (splitmix finaliser over FNV-1a).
func Unit(key string) float64 {
	h := fnv.New64a()
	h.Write([]byte(key))
	z := h.Sum64()
	z ^= z >> 30
	z *= 0xbf58476d1ce4e5b9
	z ^= z >> 27
	z *= 0x94d049bb133111eb
	z ^= z >> 31
	return float64(z>>11) / float64(1<<53)
}

// Clock: episode k occupies [T0 + k*Ep, T0 + (k+1)*Ep).
type Clock struct {
	T0 time.Time
	Ep time.Duration
	N  int
}

func ClockFromEnv() Clock {
	ms, err := strconv.ParseInt(Env("T0_UNIX_MS", ""), 10, 64)
	Must(err)
	ep, err := strconv.ParseFloat(Env("EP_SEC", "6"), 64)
	Must(err)
	n, err := strconv.Atoi(Env("EPISODES", ""))
	Must(err)
	return Clock{time.UnixMilli(ms), time.Duration(ep * float64(time.Second)), n}
}

func (c Clock) Start(k int) time.Time { return c.T0.Add(time.Duration(k) * c.Ep) }
func (c Clock) End(k int) time.Time   { return c.Start(k + 1) }

// Episode of time t, or -1 outside the campaign.
func (c Clock) Episode(t time.Time) int {
	if t.Before(c.T0) {
		return -1
	}
	k := int(t.Sub(c.T0) / c.Ep)
	if k >= c.N {
		return -1
	}
	return k
}

func Env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	if def == "" {
		log.Fatalf("missing environment variable %s", k)
	}
	return def
}

func Must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// Writer appends JSON lines to a file that must not exist yet (a recorded run is never overwritten).
type Writer struct {
	mu  sync.Mutex
	f   *os.File
	enc *json.Encoder
}

func NewWriter(path string) *Writer {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		log.Fatalf("%s: %v (a recorded run is never overwritten)", path, err)
	}
	return &Writer{f: f, enc: json.NewEncoder(f)}
}

func (w *Writer) Write(v any) {
	w.mu.Lock()
	defer w.mu.Unlock()
	Must(w.enc.Encode(v))
	Must(w.f.Sync())
}

func (w *Writer) Close() { Must(w.f.Close()) }

func Ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func Sprintf(f string, a ...any) string { return fmt.Sprintf(f, a...) }
