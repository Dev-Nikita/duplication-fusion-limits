# Open5GS (v2.7.6, built from source) + UERANSIM (v3.2.7) in one image; multi-arch (arm64 on Apple silicon).
ARG BASE=ubuntu:22.04
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-setuptools python3-wheel python3-yaml ninja-build build-essential flex bison \
      git cmake meson libsctp-dev lksctp-tools libgnutls28-dev libgcrypt-dev libssl-dev libmongoc-dev libbson-dev \
      libyaml-dev libnghttp2-dev libmicrohttpd-dev libcurl4-gnutls-dev libtins-dev libtalloc-dev libidn-dev \
      iproute2 iptables tcpdump tshark curl procps ca-certificates g++ make \
    && rm -rf /var/lib/apt/lists/*
ARG OPEN5GS_TAG=v2.7.6
RUN git clone --depth 1 --branch ${OPEN5GS_TAG} https://github.com/open5gs/open5gs.git /src/open5gs \
 && cd /src/open5gs && meson setup build --prefix=/opt/open5gs && ninja -C build install \
 && rm -rf /src/open5gs/build
ARG UERANSIM_TAG=v3.2.7
RUN git clone --depth 1 --branch ${UERANSIM_TAG} https://github.com/aligungr/UERANSIM.git /opt/ueransim \
 && cd /opt/ueransim && make -j"$(nproc)" && rm -rf /opt/ueransim/cmake-build-release/CMakeFiles
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh
ENTRYPOINT ["/scripts/core_entrypoint.sh"]
