// One probe = one source. Sends GET requests to its replica at a fixed rate and writes, for every episode, one raw
// record: number of requests, latency quantiles and error counts. Requests sent in the first GUARD_SEC of an episode
// are not used (the injector switches faults at episode boundaries). The probe knows no label and no fault.
package main

import (
	"errors"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	"paefmt/internal/common"
)

type obs struct {
	ep      int
	latMs   float64
	status  int // HTTP status, 0 transport error, -1 timeout
}

func main() {
	clk := common.ClockFromEnv()
	src := common.Env("SOURCE", "")
	target := common.Env("TARGET", "")
	rate, _ := strconv.ParseFloat(common.Env("RATE_HZ", "20"), 64)
	tout, _ := strconv.ParseFloat(common.Env("TIMEOUT_MS", "500"), 64)
	guard, _ := strconv.ParseFloat(common.Env("GUARD_SEC", "0.5"), 64)
	out := common.NewWriter(common.Env("OUT", "") + "/probe_" + src + ".jsonl")
	defer out.Close()
	cl := &http.Client{Timeout: time.Duration(tout * float64(time.Millisecond)), Transport: &http.Transport{MaxIdleConnsPerHost: 64, DisableKeepAlives: false}}
	ch := make(chan obs, 4096)
	var wg sync.WaitGroup
	time.Sleep(time.Until(clk.T0))
	tick := time.NewTicker(time.Duration(float64(time.Second) / rate))
	go func() {
		defer tick.Stop()
		for t := range tick.C {
			k := clk.Episode(t)
			if k < 0 {
				if t.After(clk.End(clk.N - 1)) {
					break
				}
				continue
			}
			if t.Sub(clk.Start(k)).Seconds() < guard {
				continue
			}
			wg.Add(1)
			go func(k int, t0 time.Time) {
				defer wg.Done()
				resp, err := cl.Get(target)
				o := obs{ep: k, latMs: common.Ms(time.Since(t0))}
				switch {
				case err == nil:
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
					o.status = resp.StatusCode
				case errors.Is(err, os.ErrDeadlineExceeded) || isTimeout(err):
					o.status, o.latMs = -1, tout
				default:
					o.status = 0
				}
				ch <- o
			}(k, t)
		}
		wg.Wait()
		close(ch)
	}()
	acc := map[int][]obs{}
	flushed := map[int]bool{}
	flush := func(k int) {
		v := acc[k]
		lat := make([]float64, len(v))
		n5, nr, nt := 0, 0, 0
		for i, o := range v {
			lat[i] = o.latMs
			switch {
			case o.status >= 500:
				n5++
			case o.status == 0:
				nr++
			case o.status == -1:
				nt++
			}
		}
		sort.Float64s(lat)
		q := func(p float64) float64 {
			if len(lat) == 0 {
				return -1
			}
			return lat[int(p*float64(len(lat)-1))]
		}
		out.Write(map[string]any{"episode": k, "source": src, "n_req": len(v), "p50_ms": q(0.5), "p90_ms": q(0.9),
			"n_5xx": n5, "n_reset": nr, "n_timeout": nt})
		delete(acc, k)
		flushed[k] = true
	}
	grace := time.Duration((tout + 200) * float64(time.Millisecond))
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case o, ok := <-ch:
			if !ok {
				for k := 0; k < clk.N; k++ {
					if !flushed[k] {
						flush(k)
					}
				}
				return
			}
			acc[o.ep] = append(acc[o.ep], o)
		case now := <-ticker.C:
			for k := 0; k < clk.N; k++ {
				if !flushed[k] && now.After(clk.End(k).Add(grace)) {
					flush(k)
				}
			}
		}
	}
}

func isTimeout(err error) bool {
	type t interface{ Timeout() bool }
	var te t
	return errors.As(err, &te) && te.Timeout()
}
