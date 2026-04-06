# fuzzer for bit-vec

Based on fuzzing in `smallvec`.

# fuzzing

```sh
cargo afl build --release --bin trie_ops --features afl && cargo afl fuzz -i in -o out target/release/trie_ops
```
