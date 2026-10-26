# Synthetic E01 fixtures — not METR-LA observations

`speeds.csv` is a hand-specified deterministic 40×5 software fixture, with IDs
SYN_01–SYN_05. The clock begins Sunday 2012-01-01 23:00, with 41 five-minute
points minus point 10. Original synthetic formula: `40 + 3*sensor_index + row%6`;
SYN_04 is `60 + .001*(row%5)` and SYN_05 is constant 55. Deliberate overrides:
(row 0, sensor 0)=NaN, (1,1)=0, (2,2)=-1, (3,3)=+Inf, (4,4)=-Inf.
Units are labeled mph solely to exercise the interface. No scientific or geographic
claim follows from these numbers.

`adjacency.csv` contains five invented nodes and directed affinity edges
SYN_01→SYN_01=1, SYN_01→SYN_02=.8, SYN_02→SYN_01=.4,
SYN_02→SYN_03=.2, SYN_03→SYN_04=.5; SYN_05 is isolated.
The graph is unrelated to the authentic METR-LA sensor network.

Run the two E01 CLIs with `config/metr_la_synthetic.yaml` to reproduce all committed
`e01_synthetic_*` outputs. Its source checksums pin these fixture files.
