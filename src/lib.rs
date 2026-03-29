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
#![feature(associated_type_defaults)]

pub use chunk::Chunk;
pub type TrieMap<K, V> = map::Map<BasicTrieHint<K, V>>;
pub use map::Map;
#[cfg(feature = "extra")]
pub use set::Set;

use crate::map_trait::BasicTrieHint;

pub mod chunk;
mod inner;
pub mod map;
pub mod map_trait;
pub mod node;
mod root_node;
#[cfg(feature = "extra")]
pub mod set;

// #[cfg(feature = "ordered_iter")]
// mod ordered_iter;
