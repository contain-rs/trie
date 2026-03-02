// Copyright 2012-2026 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(stable_features)]
#![feature(test)]
#![feature(hint_assert_unchecked)]

extern crate test;

use rand::{RngExt, SeedableRng, rngs, seq::SliceRandom};
use test::black_box;

use trie::{map::Map, map::Occupied, map::Vacant};

const MAP_SIZE: usize = 1000;

fn small_rng() -> rngs::SmallRng {
    rngs::SmallRng::seed_from_u64(42)
}

macro_rules! map_insert_rand_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use test::black_box;

            let n: usize = $n;
            let mut map = $map::new();
            // setup
            let mut rng = small_rng();

            for _ in 0..n {
                let i = rng.random::<u32>() as usize % n;
                map.insert(i, i);
            }

            // measure
            b.iter(|| {
                let k = rng.random::<u32>() as usize % n;
                map.insert(k, k);
                map.remove(&k);
            });
            black_box(map);
        }
    };
}

macro_rules! map_insert_seq_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use test::black_box;

            let mut map = $map::new();
            let n: usize = $n;
            // setup
            for i in 0..n {
                map.insert(i * 2, i * 2);
            }

            // measure
            let mut i = 1;
            b.iter(|| {
                map.insert(i, i);
                map.remove(&i);
                i = (i + 2) % n;
            });
            black_box(map);
        }
    };
}

macro_rules! map_find_rand_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use test::black_box;

            let mut map = $map::new();
            let n: usize = $n;

            // setup
            let mut rng = small_rng();
            let mut keys: Vec<_> = (0..n).map(|_| rng.random::<u32>() as usize % n).collect();

            for &k in &keys {
                map.insert(k, k);
            }

            keys.shuffle(&mut rng);

            // measure
            let mut i = 0;
            b.iter(|| {
                let t = map.get(&keys[i]);
                i = (i + 1) % n;
                black_box(t);
            })
        }
    };
}

macro_rules! map_find_seq_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use test::black_box;

            let mut map = $map::new();
            let n: usize = $n;

            // setup
            for i in 0..n {
                map.insert(i, i);
            }

            // measure
            let mut i = 0;
            b.iter(|| {
                let x = map.get(&i);
                i = (i + 1) % n;
                black_box(x);
            })
        }
    };
}

map_insert_rand_bench! {insert_rand_100,    100,    Map}
map_insert_rand_bench! {insert_rand_10_000, 10_000, Map}

map_insert_seq_bench! {insert_seq_100,    100,    Map}
map_insert_seq_bench! {insert_seq_10_000, 10_000, Map}

map_find_rand_bench! {find_rand_100,    100,    Map}
map_find_rand_bench! {find_rand_10_000, 10_000, Map}

map_find_seq_bench! {find_seq_100,    100,    Map}
map_find_seq_bench! {find_seq_10_000, 10_000, Map}

fn random_map(size: usize) -> Map<usize, usize> {
    let mut map = Map::<usize, usize>::new();
    let mut rng = small_rng();

    for _ in 0..size {
        map.insert(rng.random::<u32>() as usize, rng.random::<u32>() as usize);
    }
    map
}

fn bench_iter(b: &mut test::Bencher, size: usize) {
    let map = random_map(size);
    b.iter(|| {
        for entry in map.iter() {
            black_box(entry);
        }
    });
}

#[bench]
pub fn iter_20(b: &mut test::Bencher) {
    bench_iter(b, 20);
}

#[bench]
pub fn iter_1000(b: &mut test::Bencher) {
    bench_iter(b, 1000);
}

#[bench]
pub fn iter_100000(b: &mut test::Bencher) {
    bench_iter(b, 100000);
}

#[bench]
fn bench_lower_bound(b: &mut test::Bencher) {
    let mut m = Map::<usize>::new();
    let mut rng = small_rng();
    for _ in 0..MAP_SIZE {
        m.insert(rng.random::<u32>() as usize, rng.random::<u32>() as usize);
    }

    b.iter(|| {
        for _ in 0..10 {
            m.lower_bound(rng.random::<u32>() as usize);
        }
    });
}

#[bench]
fn bench_upper_bound(b: &mut test::Bencher) {
    let mut m = Map::<usize>::new();
    let mut rng = small_rng();
    for _ in 0..MAP_SIZE {
        m.insert(rng.random::<u32>() as usize, rng.random::<u32>() as usize);
    }

    b.iter(|| {
        for _ in 0..10 {
            m.upper_bound(rng.random::<u32>() as usize);
        }
    });
}

#[bench]
fn bench_insert_large(b: &mut test::Bencher) {
    let mut m = Map::<[usize; 10]>::new();
    let mut rng = small_rng();

    b.iter(|| {
        for _ in 0..MAP_SIZE {
            m.insert(rng.random::<u32>() as usize, [1; 10]);
        }
    });
}

#[bench]
fn bench_insert_large_entry(b: &mut test::Bencher) {
    let mut m = Map::<[usize; 10]>::new();
    let mut rng = small_rng();

    b.iter(|| {
        for _ in 0..MAP_SIZE {
            match m.entry(rng.random::<u32>() as usize) {
                Occupied(mut e) => {
                    e.insert([1; 10]);
                }
                Vacant(e) => {
                    e.insert([1; 10]);
                }
            }
        }
    });
}

#[bench]
fn bench_insert_large_low_bits(b: &mut test::Bencher) {
    let mut m = Map::<[usize; 10]>::new();
    let mut rng = small_rng();

    b.iter(|| {
        for _ in 0..MAP_SIZE {
            // only have the last few bits set.
            m.insert(rng.random::<u32>() as usize & 0xff_ff, [1; 10]);
        }
    });
}

#[bench]
fn bench_insert_small(b: &mut test::Bencher) {
    let mut m = Map::<()>::new();
    let mut rng = small_rng();

    b.iter(|| {
        for _ in 0..MAP_SIZE {
            m.insert(rng.random::<u32>() as usize, ());
        }
    });
}

#[bench]
fn bench_insert_small_low_bits(b: &mut test::Bencher) {
    let mut m = Map::<()>::new();
    let mut rng = small_rng();

    b.iter(|| {
        for _ in 0..MAP_SIZE {
            // only have the last few bits set.
            m.insert(rng.random::<u32>() as usize & 0xff_ff, ());
        }
    });
}

#[bench]
fn bench_get(b: &mut test::Bencher) {
    let map = random_map(MAP_SIZE);
    let keys: Vec<usize> = map.keys().collect();
    b.iter(|| {
        for key in keys.iter() {
            black_box(map.get(key));
        }
    });
}

#[bench]
fn bench_get_entry(b: &mut test::Bencher) {
    let mut map = random_map(MAP_SIZE);
    let keys: Vec<usize> = map.keys().collect();
    b.iter(|| {
        for key in keys.iter() {
            if let Occupied(e) = map.entry(*key) {
                black_box(e.get());
            }
        }
    });
}

#[bench]
fn bench_remove(b: &mut test::Bencher) {
    b.iter(|| {
        let mut map = random_map(MAP_SIZE);
        let keys: Vec<usize> = map.keys().collect();
        for key in keys.iter() {
            black_box(map.remove(key));
        }
    });
}

#[bench]
fn bench_remove_entry(b: &mut test::Bencher) {
    b.iter(|| {
        let mut map = random_map(MAP_SIZE);
        let keys: Vec<usize> = map.keys().collect();
        for key in keys.iter() {
            if let Occupied(e) = map.entry(*key) {
                black_box(e.remove());
            }
        }
    });
}
