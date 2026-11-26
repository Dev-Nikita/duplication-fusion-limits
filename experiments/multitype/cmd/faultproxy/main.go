// Reverse proxy in front of one replica. The injector sets its fault mode through a control port that the probe does
// not know. Faults: added delay (with jitter), fast HTTP 503 at a rate, connection reset at a rate, stall (requests are
// held until the mode changes). The proxy never sees or logs a label.
package main

import (
	"encoding/json"
	"log"
	"math/rand"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"paefmt/internal/common"
)

type Mode struct {
	DelayMs   float64 `json:"delay_ms"`
	JitterMs  float64 `json:"jitter_ms"`
	ErrRate   float64 `json:"err_rate"`
	ResetRate float64 `json:"reset_rate"`
	Stall     bool    `json:"stall"`
}

var (
	mode    atomic.Pointer[Mode]
	changed = make(chan struct{})
	chMu    sync.Mutex
	rmu     sync.Mutex
	rnd     *rand.Rand
)

func r01() float64 { rmu.Lock(); defer rmu.Unlock(); return rnd.Float64() }

func main() {
	target, err := url.Parse(common.Env("TARGET", ""))
	common.Must(err)
	rnd = rand.New(rand.NewSource(int64(common.Unit(common.Env("SEED", "x")+"|"+common.Env("NAME", "p")) * (1 << 50))))
	mode.Store(&Mode{})
	px := httputil.NewSingleHostReverseProxy(target)
	data := http.NewServeMux()
	data.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		m := mode.Load()
		if m.Stall {
			chMu.Lock()
			c := changed
			chMu.Unlock()
			select {
			case <-c:
			case <-r.Context().Done():
				return
			}
			m = mode.Load()
		}
		if m.ResetRate > 0 && r01() < m.ResetRate {
			if hj, ok := w.(http.Hijacker); ok {
				if conn, _, err := hj.Hijack(); err == nil {
					conn.Close()
					return
				}
			}
		}
		if m.ErrRate > 0 && r01() < m.ErrRate {
			http.Error(w, "injected", http.StatusServiceUnavailable)
			return
		}
		if d := m.DelayMs + m.JitterMs*(2*r01()-1); d > 0 {
			select {
			case <-time.After(time.Duration(d * float64(time.Millisecond))):
			case <-r.Context().Done():
				return
			}
		}
		px.ServeHTTP(w, r)
	})
	ctl := http.NewServeMux()
	ctl.HandleFunc("/mode", func(w http.ResponseWriter, r *http.Request) {
		var m Mode
		if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		mode.Store(&m)
		chMu.Lock()
		close(changed)
		changed = make(chan struct{})
		chMu.Unlock()
		w.WriteHeader(204)
	})
	go func() { log.Fatal(http.ListenAndServe(common.Env("CONTROL", ":9090"), ctl)) }()
	log.Fatal(http.ListenAndServe(common.Env("LISTEN", ":8081"), data))
}
