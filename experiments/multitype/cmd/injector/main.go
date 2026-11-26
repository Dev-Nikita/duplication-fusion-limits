// Fault injector. For every episode it draws the class y and a fault from a seeded schedule, writes the ground truth
// to its own file (mounted only here and for the analysis), and sets the fault mode of the affected proxies for the
// duration of the episode. Probes, broker and fuser never see this file.
//
// Classes:
//   y = 0  latency fault: added delay on k replicas; long delays also cause time-outs (latency and error together);
//   y = 1  error fault: fast HTTP 503 (or resets) at a rate on k replicas; with probability p_co also a mild delay.
// Structural condition, drawn per episode with fixed weights and recorded in the ground truth (Proposition 3(i)):
//   nested    no background: the patterns of an episode are nested (10/11 or 01/11), the order of the counts survives
//             whole-report copying;
//   mixed     background at the base rates p_bg (small delay -> 10) and p_bg_err (fast errors -> 01);
//   opposing  background at the high rates: 10 and 01 occur together, the order of the counts is no longer invariant.
package main

import (
	"bytes"
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"paefmt/internal/common"
)

type Config struct {
	DelayMs   [2]float64 `json:"delay_ms"`    // y=0: delay uniform on this range
	ErrRate   [2]float64 `json:"err_rate"`    // y=1: error rate uniform on this range
	ResetFrac float64    `json:"reset_frac"`  // y=1: share of the error rate realised as connection resets
	PCo       float64    `json:"p_co"`        // y=1: probability of an added mild delay on an affected replica
	CoDelayMs [2]float64 `json:"co_delay_ms"` // range of that mild delay
	BgDelayMs [2]float64 `json:"bg_delay_ms"`
	BgErr     [2]float64 `json:"bg_err_rate"` // background fast-error rate range
	Conds     []Cond     `json:"conditions"`
	KMin      int        `json:"k_min"` // number of affected replicas uniform on [k_min, k_max]
	KMax      int        `json:"k_max"`
	PY1       float64    `json:"p_y1"`
}

type Cond struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
	PBg    float64 `json:"p_bg"`     // background small delay on a non-affected replica
	PBgErr float64 `json:"p_bg_err"` // background fast errors on a non-affected replica
}

type Mode struct {
	DelayMs   float64 `json:"delay_ms"`
	JitterMs  float64 `json:"jitter_ms"`
	ErrRate   float64 `json:"err_rate"`
	ResetRate float64 `json:"reset_rate"`
	Stall     bool    `json:"stall"`
}

func main() {
	clk := common.ClockFromEnv()
	seed := common.Env("SEED", "")
	var cfg Config
	b, err := os.ReadFile(common.Env("CONFIG", ""))
	common.Must(err)
	common.Must(json.Unmarshal(b, &cfg))
	proxies := map[string]string{} // source -> control URL
	var names []string
	for _, kv := range strings.Split(common.Env("PROXIES", ""), ",") {
		p := strings.SplitN(kv, "=", 2)
		proxies[p[0]] = p[1]
		names = append(names, p[0])
	}
	sort.Strings(names)
	M := len(names)
	truth := common.NewWriter(common.Env("TRUTH", "") + "/truth.jsonl")
	defer truth.Close()
	cl := &http.Client{Timeout: 2 * time.Second}
	set := func(src string, m Mode) {
		body, _ := json.Marshal(m)
		for try := 0; try < 3; try++ {
			resp, err := cl.Post(proxies[src]+"/mode", "application/json", bytes.NewReader(body))
			if err == nil && resp.StatusCode == 204 {
				resp.Body.Close()
				return
			}
			time.Sleep(50 * time.Millisecond)
		}
		log.Printf("could not set mode on %s", src)
	}
	u := func(k int, what string) float64 { return common.Unit(common.Sprintf("%s|%d|%s", seed, k, what)) }
	lerp := func(r [2]float64, x float64) float64 { return r[0] + (r[1]-r[0])*x }
	for k := 0; k < clk.N; k++ {
		y := 0
		if u(k, "y") < cfg.PY1 {
			y = 1
		}
		tot := 0.0
		for _, c := range cfg.Conds {
			tot += c.Weight
		}
		cond, acc, x := cfg.Conds[len(cfg.Conds)-1], 0.0, u(k, "cond")*tot
		for _, c := range cfg.Conds {
			acc += c.Weight
			if x < acc {
				cond = c
				break
			}
		}
		kk := cfg.KMin + int(math.Floor(u(k, "k")*float64(cfg.KMax-cfg.KMin+1)))
		order := append([]string(nil), names...)
		sort.Slice(order, func(i, j int) bool { return u(k, "perm|"+order[i]) < u(k, "perm|"+order[j]) })
		modes := map[string]Mode{}
		for i, s := range order {
			var m Mode
			if i < kk {
				if y == 0 {
					d := lerp(cfg.DelayMs, u(k, "d|"+s))
					m = Mode{DelayMs: d, JitterMs: 0.2 * d}
				} else {
					e := lerp(cfg.ErrRate, u(k, "e|"+s))
					m = Mode{ErrRate: e * (1 - cfg.ResetFrac), ResetRate: e * cfg.ResetFrac}
					if u(k, "co|"+s) < cfg.PCo {
						d := lerp(cfg.CoDelayMs, u(k, "cod|"+s))
						m.DelayMs, m.JitterMs = d, 0.2*d
					}
				}
			} else if r := u(k, "bg|"+s); r < cond.PBg {
				d := lerp(cfg.BgDelayMs, u(k, "bgd|"+s))
				m = Mode{DelayMs: d, JitterMs: 0.2 * d}
			} else if r < cond.PBg+cond.PBgErr {
				m = Mode{ErrRate: lerp(cfg.BgErr, u(k, "bge|"+s))}
			}
			modes[s] = m
		}
		truth.Write(map[string]any{"episode": k, "y": y, "condition": cond.Name, "k": kk, "M": M, "modes": modes})
		time.Sleep(time.Until(clk.Start(k)))
		for _, s := range names {
			set(s, modes[s])
		}
	}
	time.Sleep(time.Until(clk.End(clk.N - 1)))
	for _, s := range names {
		set(s, Mode{})
	}
}
