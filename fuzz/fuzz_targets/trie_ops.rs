//! Simple fuzzer testing all available `SmallVec` operations
use trie::Map;
use trie::Chunk;
use std::borrow::Borrow;

// There's no point growing too much, so try not to grow
// over this size.
const CAP_GROWTH: usize = 256;

macro_rules! next_usize {
    ($b:ident) => {
        $b.next().unwrap_or(0) as usize
    };
}

macro_rules! next_u8 {
    ($b:ident) => {
        $b.next().unwrap_or(0)
    };
}

fn black_box_trie<K: Chunk + Borrow<[u8]>>(s: &Map<K, u8>) {
    // print to work as a black_box
    print!("{}", s.iter().fold(0u8, |acc, (k, v)| acc + v + k.borrow().iter().fold(0u8, |acc2, &elem| acc2 + elem)));
}

fn do_test(data: &[u8]) -> Map<Vec<u8>, u8> {
    let mut v = Map::<Vec<u8>, u8>::new();

    let mut bytes = data.iter().copied();

    while let Some(op) = bytes.next() {
        match op % 5 {
            0 => {
                v = Map::new();
            }
            1 => {
                let mut v2 = vec![];
                for _ in 0 .. next_u8!(bytes) {
                    v2.push(next_u8!(bytes));
                }
                v.insert(v2, next_u8!(bytes));
            }
            2 => {
                if v.is_empty() {
                    v.remove(&vec![]);
                } else {
                    let len = v.len();
                    let key = v.iter().nth(next_usize!(bytes) % len).unwrap().0.clone();
                    v.remove(&key);
                }
            }
            3 => {
                black_box_trie(&v);
            }
            4 => {
                let len = next_usize!(bytes) % 10;
                let mut v3 = vec![];
                for _ in 0 .. len {
                    let mut v2 = vec![];
                    for _ in 0 .. next_u8!(bytes) {
                        v2.push(next_u8!(bytes));
                    }
                    v3.push(v2);
                }
                v.extend(v3.into_iter().map(|vv| (vv, next_u8!(bytes))));
            }
            _ => panic!("booo"),
        }
    }
    v
}

#[cfg(feature = "afl")]
fn main() {
    afl::fuzz!(|data| {
        // Remove the panic hook so we can actually catch panic
        // See https://github.com/rust-fuzz/afl.rs/issues/150
        std::panic::set_hook(Box::new(|_| {}));
        do_test(data);
    });
}

#[cfg(feature = "honggfuzz")]
fn main() {
    loop {
        honggfuzz::fuzz!(|data| {
            // Remove the panic hook so we can actually catch panic
            // See https://github.com/rust-fuzz/afl.rs/issues/150
            std::panic::set_hook(Box::new(|_| {}));
            do_test_all(data);
        });
    }
}

#[cfg(test)]
mod tests {
    fn extend_vec_from_hex(hex: &str, out: &mut Vec<u8>) {
        let mut b = 0;
        for (idx, c) in hex.as_bytes().iter().enumerate() {
            b <<= 4;
            match *c {
                b'A'..=b'F' => b |= c - b'A' + 10,
                b'a'..=b'f' => b |= c - b'a' + 10,
                b'0'..=b'9' => b |= c - b'0',
                b'\n' => {}
                b' ' => {}
                _ => panic!("Bad hex"),
            }
            if (idx & 1) == 1 {
                out.push(b);
                b = 0;
            }
        }
    }

    #[test]
    fn duplicate_crash() {
        let mut a = Vec::new();
        // paste the output of `xxd -p <crash_dump>` here and run `cargo test`
        extend_vec_from_hex(
            r#"
            782a
            "#,
            &mut a,
        );
        super::do_test(&a);
    }
}
