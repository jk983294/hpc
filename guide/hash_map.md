## recommendation
* good simple, default choice, google's flat_hash_map (SIMD hash table), un-success lookup is very very good, success lookup hurts, also good for insertion
* bytell_hash_map, both success/un-success lookup are good, memory consumption is less
* ska::flat_hash_map, good for both lookup, and consume more memory than bytell_hash_map
