use std::collections::{HashMap, VecDeque};



struct LRUCache {
    capacity: i32,
    cache: HashMap<i32, i32>,
    queue: VecDeque<i32>,
}

impl LRUCache {

    fn new(capacity: i32) -> Self {
        LRUCache {
            capacity,
            cache: HashMap::new(),
            queue: VecDeque::new(),
        }
    }
    
    fn get(&self, key: i32) -> i32 {
        match self.cache.get(&key) {
            Some(value) => *value,
            None => -1
        }
    }
    
    fn put(&mut self, key: i32, value: i32) {
        self.cache.insert(key, value);
        self.queue.push_back(key);
        if self.queue.len() > self.capacity as usize {
            let key = self.queue.pop_front().unwrap();
            self.cache.remove(&key);
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: add tests for LRUCache
}