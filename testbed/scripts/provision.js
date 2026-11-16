// Bulk-provision subscribers with the same document as open5gs-dbctl "add imsi key opc" (v2.7.6).
// Usage (inside the mongo container): mongosh mongodb://localhost/open5gs --quiet --eval "var N=4000" /scripts/provision.js
const K = "465B5CE8B199B49FAA5F0A2EE238A6BC", OPC = "E8ED289DEBA952E4283B54E88E6183CA";
const base = 999700000000001;  // provisioned range; 999700000900001.. is left unprovisioned (reject fault)
const n = (typeof N === "undefined") ? 4000 : N;
db.subscribers.deleteMany({});
const docs = [];
for (let i = 0; i < n; i++) {
  docs.push({
    schema_version: NumberInt(1), imsi: String(base + i), msisdn: [], imeisv: [], mme_host: [], mm_realm: [], purge_flag: [],
    slice: [{ sst: NumberInt(1), default_indicator: true, session: [{
      name: "internet", type: NumberInt(3),
      qos: { index: NumberInt(9), arp: { priority_level: NumberInt(8), pre_emption_capability: NumberInt(1), pre_emption_vulnerability: NumberInt(2) } },
      ambr: { downlink: { value: NumberInt(1000000000), unit: NumberInt(0) }, uplink: { value: NumberInt(1000000000), unit: NumberInt(0) } },
      pcc_rule: [], _id: new ObjectId() }], _id: new ObjectId() }],
    security: { k: K, op: null, opc: OPC, amf: "8000" },
    ambr: { downlink: { value: NumberInt(1000000000), unit: NumberInt(0) }, uplink: { value: NumberInt(1000000000), unit: NumberInt(0) } },
    access_restriction_data: 32, network_access_mode: 0, subscriber_status: 0, operator_determined_barring: 0,
    subscribed_rau_tau_timer: 12, __v: 0 });
}
db.subscribers.insertMany(docs);
print("provisioned " + db.subscribers.countDocuments({}) + " subscribers");
