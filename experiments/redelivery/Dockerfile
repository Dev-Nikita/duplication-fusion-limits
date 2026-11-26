FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum main.go ./
RUN go build -o /redelivery .

FROM alpine:3.20
COPY --from=build /redelivery /redelivery
ENTRYPOINT ["/redelivery"]
