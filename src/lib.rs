// Copyright 2014-2026 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An ordered map and set based on a trie.

#![recursion_limit = "1024"]

#![feature(generic_const_exprs)]
#![feature(associated_type_defaults)]
#![feature(generic_const_items)]

pub use chunk::Chunk;
pub type TrieMap<K, V> = map::Map<BasicTrieHint<K, V>>;
pub use map::Map;
#[cfg(feature = "extra")]
pub use set::Set;

use crate::map_trait::BasicTrieHint;

pub mod map_trait;
pub mod chunk;
pub mod map;
#[cfg(feature = "extra")]
pub mod set;
pub mod node;

// #[cfg(feature = "ordered_iter")]
// mod ordered_iter;
