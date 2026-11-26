// A small HTTP service: GET /work does a fixed amount of CPU work and returns 200.
package main

import (
	"crypto/sha256"
	"fmt"
	"log"
	"net/http"

	"paefmt/internal/common"
)

func main() {
	name := common.Env("NAME", "replica")
	http.HandleFunc("/work", func(w http.ResponseWriter, r *http.Request) {
		b := []byte(name)
		for i := 0; i < 2000; i++ {
			s := sha256.Sum256(b)
			b = s[:]
		}
		fmt.Fprintf(w, "{\"ok\":true,\"d\":\"%x\"}", b[:4])
	})
	log.Fatal(http.ListenAndServe(common.Env("LISTEN", ":8080"), nil))
}
