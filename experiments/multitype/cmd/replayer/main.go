// Phase B. Replays frozen original whole reports (pattern of two types) through NATS JetStream and logs every
// delivery. Copies are produced by the broker only: a report whose processing takes longer than AckWait is redelivered
// until it is acknowledged or MaxDeliver is reached. The program never duplicates a report itself.
// Originals are published episode by episode at PUB_RATE reports/s; publish times are logged so that the analysis
// can apply the episode window (late deliveries).
// Env: NATS_URL, REPORTS (jsonl with episode_id, source_id, root_event_id, pattern), OUT, REGIMES, SEED, ACK_WAIT_MS,
// PUB_RATE, LOAD_REF.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"

	"paefmt/internal/common"
)

type report struct {
	EpisodeID   string `json:"episode_id"`
	SourceID    string `json:"source_id"`
	RootEventID string `json:"root_event_id"`
	Pattern     [2]int `json:"pattern"`
}

type regime struct {
	maxDeliver int
	pSlow      float64 // probability that processing exceeds AckWait
	pBurst     float64 // the same inside a burst episode (0: no bursts)
	slowMax    float64 // slow processing time uniform on [1.1, slowMax] x AckWait
	content    bool    // processing time set by the pattern
	load       bool    // probability of slow processing grows with the number of unacknowledged messages
}

// D = 2 for every regime except S6 (stress above the cap, outside the primary analysis).
var regimes = map[string]regime{
	"S0": {3, 0, 0, 0, false, false},
	"S1": {3, 0.30, 0, 2.5, false, false},
	"S2": {3, 0.60, 0, 2.5, false, false},
	"S3": {3, 0.05, 0.90, 2.5, false, false},
	"S4": {3, 0, 0, 2.5, true, false},
	"S5": {3, 0, 0, 2.5, false, true},
	"S6": {5, 0.70, 0, 4.5, false, false},
}

func procUnits(name string, g regime, r report, seed string, inflight int64, loadRef float64) float64 {
	if g.content { // 11: two redeliveries, 10: one, 01: none
		switch r.Pattern {
		case [2]int{1, 1}:
			return 2.2
		case [2]int{1, 0}:
			return 1.3
		}
		return 0
	}
	p := g.pSlow
	if g.load {
		p = math.Min(0.9, float64(inflight)/loadRef)
	}
	if g.pBurst > 0 && common.Unit(seed+"|burst|"+name+"|"+r.EpisodeID) < 0.2 {
		p = g.pBurst
	}
	if common.Unit(seed+"|slow|"+name+"|"+r.RootEventID) >= p {
		return 0
	}
	return 1.1 + (g.slowMax-1.1)*common.Unit(seed+"|len|"+name+"|"+r.RootEventID)
}

func main() {
	seed := common.Env("SEED", "")
	ackMs, _ := strconv.Atoi(common.Env("ACK_WAIT_MS", "250"))
	pubRate, _ := strconv.ParseFloat(common.Env("PUB_RATE", "100"), 64)
	loadRef, _ := strconv.ParseFloat(common.Env("LOAD_REF", "200"), 64)
	ackWait := time.Duration(ackMs) * time.Millisecond
	var reports []report
	var raw [][]byte
	f, err := os.Open(common.Env("REPORTS", ""))
	common.Must(err)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r report
		common.Must(json.Unmarshal(sc.Bytes(), &r))
		if r.Pattern == [2]int{0, 0} {
			log.Fatal("zero pattern in the input: only positive whole reports are replayed")
		}
		reports = append(reports, r)
		raw = append(raw, append([]byte(nil), sc.Bytes()...))
	}
	f.Close()
	nc, err := nats.Connect(common.Env("NATS_URL", ""), nats.MaxReconnects(-1), nats.RetryOnFailedConnect(true))
	common.Must(err)
	for i := 0; i < 150 && !nc.IsConnected(); i++ {
		time.Sleep(200 * time.Millisecond)
	}
	if !nc.IsConnected() {
		log.Fatal("no connection to the broker")
	}
	defer nc.Drain()
	js, err := jetstream.New(nc)
	common.Must(err)
	ctx := context.Background()
	out := common.Env("OUT", "")
	common.Must(os.MkdirAll(out, 0o755))
	meta := map[string]any{"seed": seed, "ack_wait_ms": ackMs, "pub_rate": pubRate, "reports": len(reports),
		"server": nc.ConnectedServerVersion(), "started": time.Now().UTC().Format(time.RFC3339)}
	for _, name := range strings.Split(common.Env("REGIMES", ""), ",") {
		g, ok := regimes[name]
		if !ok {
			log.Fatalf("unknown regime %q", name)
		}
		dw := common.NewWriter(fmt.Sprintf("%s/deliveries_%s.jsonl", out, name))
		pw := common.NewWriter(fmt.Sprintf("%s/published_%s.jsonl", out, name))
		stream := "MT_" + name
		_ = js.DeleteStream(ctx, stream) // reset between regimes: a fresh stream and consumer
		_, err := js.CreateStream(ctx, jetstream.StreamConfig{Name: stream, Subjects: []string{"mt." + name}, Storage: jetstream.MemoryStorage})
		common.Must(err)
		cons, err := js.CreateOrUpdateConsumer(ctx, stream, jetstream.ConsumerConfig{
			Durable: "fuser", AckPolicy: jetstream.AckExplicitPolicy, AckWait: ackWait, MaxDeliver: g.maxDeliver, MaxAckPending: 100000})
		common.Must(err)
		byRoot := map[string]report{}
		for _, r := range reports {
			byRoot[r.RootEventID] = r
		}
		var acked, inflight int64
		done := make(chan struct{})
		t0 := time.Now()
		go func() { // consumer
			for atomic.LoadInt64(&acked) < int64(len(reports)) {
				batch, err := cons.Fetch(500, jetstream.FetchMaxWait(100*time.Millisecond))
				if err != nil {
					continue
				}
				for m := range batch.Messages() {
					md, err := m.Metadata()
					common.Must(err)
					var r report
					common.Must(json.Unmarshal(m.Data(), &r))
					dw.Write(map[string]any{"regime": name, "root_event_id": r.RootEventID, "episode_id": r.EpisodeID,
						"source_id": r.SourceID, "pattern": r.Pattern, "stream_sequence": md.Sequence.Stream,
						"num_delivered": md.NumDelivered, "bytes": len(m.Data()), "received_at": time.Now().UTC().Format(time.RFC3339Nano)})
					if md.NumDelivered == 1 {
						d := time.Duration(procUnits(name, g, byRoot[r.RootEventID], seed, atomic.LoadInt64(&inflight), loadRef) * float64(ackWait))
						atomic.AddInt64(&inflight, 1)
						go func(m jetstream.Msg, d time.Duration) {
							time.Sleep(d)
							if err := m.Ack(); err != nil {
								log.Printf("ack: %v", err)
							}
							atomic.AddInt64(&inflight, -1)
							atomic.AddInt64(&acked, 1)
						}(m, d)
					}
				}
			}
			close(done)
		}()
		gap := time.Duration(float64(time.Second) / pubRate)
		next := time.Now()
		for i, r := range reports {
			time.Sleep(time.Until(next))
			next = next.Add(gap)
			m := nats.NewMsg("mt." + name)
			m.Data = raw[i]
			m.Header.Set(nats.MsgIdHdr, name+"|"+r.RootEventID) // a publish retry cannot create a second original
			_, err := js.PublishMsg(ctx, m)
			common.Must(err)
			pw.Write(map[string]any{"root_event_id": r.RootEventID, "episode_id": r.EpisodeID, "published_at": time.Now().UTC().Format(time.RFC3339Nano)})
		}
		select {
		case <-done:
		case <-time.After(time.Duration(len(reports))*gap + 10*time.Minute):
			log.Printf("%s: not every original acknowledged (MaxDeliver exhausted?)", name)
		}
		time.Sleep(2 * ackWait)
		if b, err := cons.Fetch(500, jetstream.FetchMaxWait(200*time.Millisecond)); err == nil {
			n := 0
			for range b.Messages() {
				n++
			}
			meta["late_"+name] = n
		}
		dw.Close()
		pw.Close()
		info, err := cons.Info(ctx)
		common.Must(err)
		meta["regime_"+name] = map[string]any{"max_deliver": g.maxDeliver, "seconds": time.Since(t0).Seconds(),
			"ack_pending_at_end": info.NumAckPending, "redelivered_at_end": info.NumRedelivered}
		common.Must(js.DeleteStream(ctx, stream))
		log.Printf("%s done in %.1fs", name, time.Since(t0).Seconds())
	}
	meta["finished"] = time.Now().UTC().Format(time.RFC3339)
	b, _ := json.MarshalIndent(meta, "", " ")
	common.Must(os.WriteFile(out+"/run_meta.json", b, 0o644))
}
