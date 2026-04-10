okay, so here are the notes

- crit-bit tries
- qp tries
- basic tries
- fixie trie: qp trie but with fixed-length keys only

source

https://github.com/sdleffler/qp-trie-rs/blob/master/src/trie.rs

moving parts for performance
- root node type (Internal/Trie)
- popcount with alloc or const-sized array
- crit-bit
- fanout (the SIZE in [InternalNode<K, V>; SIZE])
- hashing or no hashing (hash type) (useful for ipfs and such)
- value stored at every branching or not (variable size values??? forbid, with leaves only, or allow prefix storage)
- prefetching
- concurrent or not concurrent
- immutable or mutable
- adaptive
