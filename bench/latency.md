Latency Comparison Numbers (~2012 by Jeff Dean)

----------------------------------

| operation                         | latency   |  comment                      |
| --------                          | -----:    | :----:                        |
| L1 cache reference                | 0.5 ns    |                               |
| Branch mispredict                 | 5 ns      |                               |
| L2 cache reference                | 7 ns      | 14x L1 cache                  |
| Mutex lock/unlock                 | 25 ns     |                               |
| Main memory reference             | 100 ns    | 20x L2 cache, 200x L1 cache   |
| Compress 1K bytes with Zippy      | 3 us      |                               |
| Send 1K bytes over 1 Gbps network | 10 us     |                               |
| Read 4K randomly from SSD         | 150 us    | 1GB/sec SSD                   |
| Read 1 MB sequentially from memory| 250 us    |                               |
| Round trip within same data-center| 500 us    |                               |
| Read 1 MB sequentially from SSD   | 1 ms      | 1GB/sec SSD, 4X memory        |
| Disk seek                         | 10 ms     | 20x data-center round trip    |
| Read 1 MB sequentially from disk  | 20 ms     | 80x memory, 20X SSD           |
| Send packet CA->Netherlands->CA   | 150 ms    | CA, Certificate Authority     |


