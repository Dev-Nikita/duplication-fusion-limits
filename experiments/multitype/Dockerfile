FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY internal ./internal
COPY cmd ./cmd
RUN CGO_ENABLED=0 go build -o /bin/ ./cmd/...

FROM alpine:3.20
COPY --from=build /bin/replica /bin/faultproxy /bin/probe /bin/injector /bin/replayer /usr/local/bin/
