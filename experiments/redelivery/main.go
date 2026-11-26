// Replays positive reports through NATS JetStream and logs every delivery.
// Copies are produced by the broker only: a report whose processing takes longer than AckWait is redelivered by
// JetStream until it is acknowledged or MaxDeliver is reached. The program never duplicates a report itself.
// Env: NATS_URL, REPORTS (jsonl), OUT (dir), REGIMES (comma list), SEED, ACK_WAIT_MS.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

type report struct {
	EpisodeID   string  `json:"episode_id"`
	SourceID    string  `json:"source_id"`
	RootEventID string  `json:"root_event_id"`
	Severity    float64 `json:"severity"`
}

type regime struct {
	maxDeliver int
	pSlow      float64 // probability that processing exceeds AckWait
	pBurst     float64 // the same inside a burst episode (0: no bursts)
	slowMax    float64 // slow processing time is uniform on [1.1, slowMax] x AckWait
	content    bool    // processing time grows with the reported severity instead
}

var regimes = map[string]regime{
	"S0":  {3, 0, 0, 0, false},
	"S1":  {3, 0.30, 0, 2.5, false},
	"S2":  {3, 0.05, 0.90, 2.5, false},
	"S3a": {3, 0.70, 0, 2.5, false},
	"S3b": {5, 0.70, 0, 4.5, false},
	"S4":  {3, 0, 0, 0, true},
}

// unit returns a reproducible number in [0,1) for a key.
func unit(key string) float64 {
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

// processing time of a report, in units of AckWait
func procUnits(name string, g regime, r report, seed string) float64 {
	if g.content {
		return math.Max(0, math.Min(2.5, 2*(r.Severity-1)))
	}
	p := g.pSlow
	if g.pBurst > 0 && unit(seed+"|burst|"+name+"|"+r.EpisodeID) < 0.2 {
		p = g.pBurst
	}
	if unit(seed+"|slow|"+name+"|"+r.RootEventID) >= p {
		return 0
	}
	return 1.1 + (g.slowMax-1.1)*unit(seed+"|len|"+name+"|"+r.RootEventID)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	seed := os.Getenv("SEED")
	ackMs, err := strconv.Atoi(os.Getenv("ACK_WAIT_MS"))
	if err != nil {
		ackMs = 250
	}
	ackWait := time.Duration(ackMs) * time.Millisecond
	var reports []report
	var raw [][]byte
	f, err := os.Open(os.Getenv("REPORTS"))
	must(err)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r report
		must(json.Unmarshal(sc.Bytes(), &r))
		reports = append(reports, r)
		raw = append(raw, append([]byte(nil), sc.Bytes()...))
	}
	f.Close()
	nc, err := nats.Connect(os.Getenv("NATS_URL"), nats.MaxReconnects(-1), nats.RetryOnFailedConnect(true))
	must(err)
	for i := 0; i < 150 && !nc.IsConnected(); i++ { // the broker container may still be starting
		time.Sleep(200 * time.Millisecond)
	}
	if !nc.IsConnected() {
		log.Fatal("no connection to the broker")
	}
	defer nc.Drain()
	js, err := jetstream.New(nc)
	must(err)
	ctx := context.Background()
	out := os.Getenv("OUT")
	must(os.MkdirAll(out, 0o755))
	meta := map[string]any{"seed": seed, "ack_wait_ms": ackMs, "reports": len(reports), "server": nc.ConnectedServerVersion(), "started": time.Now().UTC().Format(time.RFC3339)}
	for _, name := range strings.Split(os.Getenv("REGIMES"), ",") {
		g, ok := regimes[name]
		if !ok {
			log.Fatalf("unknown regime %q", name)
		}
		path := fmt.Sprintf("%s/deliveries_%s.jsonl", out, name)
		if _, err := os.Stat(path); err == nil {
			log.Fatalf("%s exists: a recorded run is never overwritten", path)
		}
		stream := "REP_" + name
		_ = js.DeleteStream(ctx, stream)
		_, err := js.CreateStream(ctx, jetstream.StreamConfig{Name: stream, Subjects: []string{"rep." + name}, Storage: jetstream.MemoryStorage})
		must(err)
		cons, err := js.CreateOrUpdateConsumer(ctx, stream, jetstream.ConsumerConfig{
			Durable: "fuser", AckPolicy: jetstream.AckExplicitPolicy, AckWait: ackWait, MaxDeliver: g.maxDeliver, MaxAckPending: 100000})
		must(err)
		byRoot := map[string]report{}
		for i, r := range reports {
			byRoot[r.RootEventID] = r
			m := nats.NewMsg("rep." + name)
			m.Data = raw[i]
			m.Header.Set(nats.MsgIdHdr, name+"|"+r.RootEventID) // a publish retry cannot create a second original
			_, err := js.PublishMsg(ctx, m)
			must(err)
		}
		w, err := os.Create(path)
		must(err)
		enc := json.NewEncoder(w)
		var mu sync.Mutex
		var acked int64
		t0 := time.Now()
		for atomic.LoadInt64(&acked) < int64(len(reports)) {
			batch, err := cons.Fetch(500, jetstream.FetchMaxWait(100*time.Millisecond))
			if err != nil {
				continue
			}
			for m := range batch.Messages() {
				md, err := m.Metadata()
				must(err)
				var r report
				must(json.Unmarshal(m.Data(), &r))
				mu.Lock()
				must(enc.Encode(map[string]any{"regime": name, "root_event_id": r.RootEventID, "episode_id": r.EpisodeID, "source_id": r.SourceID,
					"stream_sequence": md.Sequence.Stream, "num_delivered": md.NumDelivered, "bytes": len(m.Data()),
					"received_at": time.Now().UTC().Format(time.RFC3339Nano)}))
				mu.Unlock()
				if md.NumDelivered == 1 { // the work on a report starts at its first delivery and is acknowledged when it ends
					d := time.Duration(procUnits(name, g, byRoot[r.RootEventID], seed) * float64(ackWait))
					go func(m jetstream.Msg, d time.Duration) {
						time.Sleep(d)
						if err := m.Ack(); err != nil {
							log.Printf("ack: %v", err)
						}
						atomic.AddInt64(&acked, 1)
					}(m, d)
				}
			}
		}
		time.Sleep(2 * ackWait) // nothing may arrive after the last acknowledgement; wait to verify
		if b, err := cons.Fetch(500, jetstream.FetchMaxWait(200*time.Millisecond)); err == nil {
			n := 0
			for range b.Messages() {
				n++
			}
			meta["late_"+name] = n
		}
		w.Close()
		info, err := cons.Info(ctx)
		must(err)
		meta["regime_"+name] = map[string]any{"max_deliver": g.maxDeliver, "seconds": time.Since(t0).Seconds(), "ack_pending_at_end": info.NumAckPending, "redelivered_at_end": info.NumRedelivered}
		must(js.DeleteStream(ctx, stream))
		log.Printf("%s done in %.1fs", name, time.Since(t0).Seconds())
	}
	meta["finished"] = time.Now().UTC().Format(time.RFC3339)
	b, _ := json.MarshalIndent(meta, "", " ")
	must(os.WriteFile(out+"/run_meta.json", b, 0o644))
}
