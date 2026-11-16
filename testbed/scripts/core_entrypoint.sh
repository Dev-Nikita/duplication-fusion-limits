#!/usr/bin/env bash
# Configure and start the 5G SA core (all NFs on loopback addresses, as in the Open5GS default
# configuration) and keep the container alive. UERANSIM processes are started by the campaign driver.
set -u
CFG=/opt/open5gs/etc/open5gs
mkdir -p /out/logs /opt/open5gs/var/log/open5gs
# patch configuration: database in the mongo container, more UE contexts
sed -i "s#mongodb://localhost/open5gs#${DB_URI:-mongodb://mongo/open5gs}#" $CFG/*.yaml
sed -i "s/^    ue: 1024 .*/    ue: 8192/" $CFG/*.yaml
# user-plane TUN for the UPF (registration does not need it, PDU sessions do)
ip tuntap add name ogstun mode tun 2>/dev/null || true
ip addr add 10.45.0.1/16 dev ogstun 2>/dev/null || true
ip link set ogstun up 2>/dev/null || true
BIN=/opt/open5gs/bin
for nf in nrf scp amf smf upf ausf udm pcf nssf bsf udr; do
  $BIN/open5gs-${nf}d > /out/logs/${nf}.stdout 2>&1 &
  echo "$!" > /tmp/${nf}.pid
  sleep 0.5
done
echo "core started: $(date -u +%FT%TZ)" > /out/logs/core_started.txt
# gNB (needs SCTP to the AMF at 127.0.0.5:38412)
sleep 3
/opt/ueransim/build/nr-gnb -c /opt/ueransim/config/open5gs-gnb.yaml > /out/logs/gnb.stdout 2>&1 &
echo "$!" > /tmp/gnb.pid
tail -f /dev/null
