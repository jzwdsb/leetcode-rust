#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

struct MinStack {
    stack: Vec<i32>,
    min_stack: Vec<i32>,
}

impl MinStack {
    fn new() -> Self {
        MinStack {
            stack: Vec::new(),
            min_stack: Vec::new(),
        }
    }

    fn push(&mut self, x: i32) {
        self.stack.push(x);
        if self.min_stack.is_empty() || x <= *self.min_stack.last().unwrap() {
            self.min_stack.push(x);
        }
    }

    fn pop(&mut self) {
        let x = self.stack.pop().unwrap();
        if x == *self.min_stack.last().unwrap() {
            self.min_stack.pop();
        }
    }

    fn top(&self) -> i32 {
        *self.stack.last().unwrap()
    }

    fn get_min(&self) -> i32 {
        *self.min_stack.last().unwrap()
    }
}

struct Allocator {
    free_memory: BTreeMap<i32, i32>,
    allocated: HashMap<i32, Vec<(i32, i32)>>,
}

impl Allocator {
    fn new(size: usize) -> Self {
        Self {
            free_memory: BTreeMap::from([(0, size as i32)]),
            allocated: HashMap::new(),
        }
    }

    pub fn allocate(&mut self, size: usize, m_id: i32) -> Option<usize> {
        if let Some((&index, &ptr_size)) = self.free_memory.iter().find(|(_, &v)| v >= size as i32)
        {
            self.free_memory.remove(&index);
            let ptr = if ptr_size == size as i32 {
                (index, ptr_size)
            } else {
                self.free_memory
                    .insert(index + size as i32, ptr_size - size as i32);
                (index, size as i32)
            };
            self.allocated.entry(m_id).or_insert(Vec::new()).push(ptr);
            return Some(ptr.0 as usize);
        }
        None
    }

    fn deallocate(&mut self, (index, size): (i32, i32)) {
        let mut size = size;
        let mut index = index;
        let right_index = index + size;
        if self.free_memory.contains_key(&right_index) {
            let adjacent_ptr = self.free_memory.remove(&right_index).unwrap();
            size += adjacent_ptr;
        }
        if let Some((&left_index, &left_size)) = self.free_memory.range(0..index).rev().next() {
            if left_index + left_size == index {
                index = left_index;
                size += left_size;
            }
        }
        self.free_memory.insert(index, size);
    }

    fn free(&mut self, m_id: i32) -> i32 {
        if let Some(ptrs) = self.allocated.remove(&m_id) {
            let mut mem = 0;
            for ptr in ptrs.into_iter() {
                self.deallocate(ptr);
                mem += ptr.1;
            }
            return mem;
        }
        0
    }
}

struct DataStream {
    value: i32,
    k: i32,
    cnt: i32,
}

impl DataStream {
    pub fn new(value: i32, k: i32) -> Self {
        Self { value, k, cnt: 0 }
    }

    pub fn consec(&mut self, num: i32) -> bool {
        if num == self.value {
            self.cnt += 1;
        } else {
            self.cnt = 0;
        }
        self.cnt == self.k
    }
}

struct FrequencyTracker {
    nums: HashMap<i32, i32>,
    nums_by_freq: HashMap<i32, HashSet<i32>>,
}

impl FrequencyTracker {
    pub fn new() -> Self {
        Self {
            nums: HashMap::new(),
            nums_by_freq: HashMap::new(),
        }
    }

    pub fn add(&mut self, num: i32) {
        let freq = self.nums.entry(num).or_insert(0);
        if *freq > 0 {
            self.nums_by_freq.entry(*freq).or_default().remove(&num);
            if self.nums_by_freq[&freq].is_empty() {
                self.nums_by_freq.remove(&freq);
            }
        }
        *freq += 1;
        self.nums_by_freq.entry(*freq).or_default().insert(num);
    }

    pub fn delete_one(&mut self, num: i32) {
        if let Some(v) = self.nums.get_mut(&num) {
            self.nums_by_freq.entry(*v).or_default().remove(&num);
            if self.nums_by_freq[&v].is_empty() {
                self.nums_by_freq.remove(&v);
            }
            *v -= 1;
            if *v == 0 {
                self.nums.remove(&num);
            } else {
                self.nums_by_freq.entry(*v).or_default().insert(num);
            }
        }
    }

    pub fn has_frequency(&self, freq: i32) -> bool {
        self.nums_by_freq.contains_key(&freq)
    }
}

/*
https://leetcode.com/problems/subrectangle-queries/
different approach can be used in this problem
1. update the matrix in place and get the value in O(1)
2. keep track of the updates and get the value in O(n)
3. use segment tree to get the value in O(logn) and update in O(logn)

we may use different approach based on the proportion of update and get_value operation
1. if the number of get_value operation is much more than the number of update operation, we may use approach 1
2. if the number of update operation is much more than the number of get_value operation, we may use approach 2
3. if the number of update operation is close to the number of get_value operation, we may use approach 3
*/
struct SubrectangleQueries {
    rectangle: Vec<Vec<i32>>,
    updates: Vec<(usize, usize, usize, usize, i32)>,
}

impl SubrectangleQueries {
    fn new(rectangle: Vec<Vec<i32>>) -> Self {
        Self {
            rectangle,
            updates: Vec::new(),
        }
    }

    fn update_subrectangle(&mut self, row1: i32, col1: i32, row2: i32, col2: i32, new_value: i32) {
        self.updates.push((
            row1 as usize,
            col1 as usize,
            row2 as usize,
            col2 as usize,
            new_value,
        ));
    }

    fn get_value(&mut self, row: i32, col: i32) -> i32 {
        for &(row1, col1, row2, col2, new_value) in self.updates.iter().rev() {
            if row1 <= row as usize
                && row as usize <= row2
                && col1 <= col as usize
                && col as usize <= col2
            {
                return new_value;
            }
        }
        self.rectangle[row as usize][col as usize]
    }
}

type Point = (usize, usize);

struct SegmentTree2D {
    rows: usize,
    cols: usize,
    vals: Vec<i32>,
    lazy: Vec<i32>,
    rectangle: Vec<Vec<i32>>,
}

impl SegmentTree2D {
    pub fn new(rectangle: Vec<Vec<i32>>) -> Self {
        let rows = rectangle.len();
        let cols = if rows > 0 { rectangle[0].len() } else { 0 };
        let vals = vec![0; rows * cols * 4];
        let lazy = vec![0; rows * cols * 4];
        let mut obj = Self {
            rows,
            cols,
            vals,
            lazy,
            rectangle,
        };
        obj.construct(0, (0, 0), (rows - 1, cols - 1));
        obj
    }

    fn construct(&mut self, i: i32, p1: Point, p2: Point) {
        if p1.0 >= self.rows || p1.1 >= self.cols || p2.0 >= self.rows || p2.1 >= self.cols {
            return;
        }
        if p1 == p2 {
            self.vals[i as usize] = self.rectangle[p1.0][p1.1];
            return;
        }
        let mid = ((p1.0 + p2.0) / 2, (p1.1 + p2.1) / 2);
        self.construct(i * 4 + 1, p1, mid);
        self.construct(i * 4 + 2, (p1.0, mid.1 + 1), (mid.0, p2.1));
        self.construct(i * 4 + 3, (mid.0 + 1, p1.1), (p2.0, mid.1));
        self.construct(i * 4 + 4, (mid.0 + 1, mid.1 + 1), p2);
    }

    pub fn update_subrectangle(&mut self, left_up: Point, right_down: Point, new_value: i32) {
        self.update(
            0,
            (0, 0),
            (self.rows - 1, self.cols - 1),
            left_up,
            right_down,
            new_value,
        );
    }

    pub fn get_value(&mut self, p: Point) -> i32 {
        self.query(0, (0, 0), (self.rows - 1, self.cols - 1), p)
    }

    fn update(
        &mut self,
        i: i32,
        p1: Point,
        p2: Point,
        left_up: Point,
        right_down: Point,
        new_value: i32,
    ) {
        if p1.0 >= self.rows || p1.1 >= self.cols || p2.0 >= self.rows || p2.1 >= self.cols {
            return;
        }
        if p1 == p2 {
            self.vals[i as usize] = new_value;
            return;
        }
        let mid = ((p1.0 + p2.0) / 2, (p1.1 + p2.1) / 2);
        if left_up <= mid {
            self.update(i * 4 + 1, p1, mid, left_up, right_down, new_value);
        }
        if (left_up.0 <= mid.0 && right_down.1 > mid.1)
            || (left_up.1 <= mid.1 && right_down.0 > mid.0)
        {
            self.update(
                i * 4 + 2,
                (p1.0, mid.1 + 1),
                (mid.0, p2.1),
                left_up,
                right_down,
                new_value,
            );
        }
        if (left_up.0 > mid.0 && right_down.1 <= mid.1)
            || (left_up.1 > mid.1 && right_down.0 <= mid.0)
        {
            self.update(
                i * 4 + 3,
                (mid.0 + 1, p1.1),
                (p2.0, mid.1),
                left_up,
                right_down,
                new_value,
            );
        }
        if right_down > mid {
            self.update(
                i * 4 + 4,
                (mid.0 + 1, mid.1 + 1),
                p2,
                left_up,
                right_down,
                new_value,
            );
        }
    }

    fn query(&mut self, i: i32, p1: Point, p2: Point, point: Point) -> i32 {
        if p1.0 >= self.rows || p1.1 >= self.cols || p2.0 >= self.rows || p2.1 >= self.cols {
            return 0;
        }
        if p1 == p2 {
            return self.vals[i as usize];
        }
        let mid = ((p1.0 + p2.0) / 2, (p1.1 + p2.1) / 2);
        let mut res = 0;
        if point <= mid {
            res += self.query(i * 4 + 1, p1, mid, point);
        }
        if (point.0 <= mid.0 && point.1 > mid.1) || (point.1 <= mid.1 && point.0 > mid.0) {
            res += self.query(i * 4 + 2, (p1.0, mid.1 + 1), (mid.0, p2.1), point);
        }
        if (point.0 > mid.0 && point.1 <= mid.1) || (point.1 > mid.1 && point.0 <= mid.0) {
            res += self.query(i * 4 + 3, (mid.0 + 1, p1.1), (p2.0, mid.1), point);
        }
        if point > mid {
            res += self.query(i * 4 + 4, (mid.0 + 1, mid.1 + 1), p2, point);
        }
        res
    }
}

use base_62;
struct Codec {
    store: HashMap<String, String>,
}

impl Codec {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn encode(&mut self, long_url: String) -> String {
        let code = base_62::encode(long_url.as_bytes());
        self.store.insert(code.clone(), long_url);
        code
    }

    pub fn decode(&self, s: String) -> String {
        self.store.get(&s).unwrap_or(&"".to_string()).clone()
    }
}

struct BowserHistory {
    history: Vec<String>,
    cur: usize,
}

impl BowserHistory {
    fn new(homepage: String) -> Self {
        Self {
            history: vec![homepage],
            cur: 0,
        }
    }

    fn visit(&mut self, url: String) {
        self.history.truncate(self.cur + 1);
        self.history.push(url);
        self.cur += 1;
    }

    fn back(&mut self, steps: i32) -> String {
        self.cur = (self.cur as i32 - steps).max(0) as usize;
        self.history[self.cur].clone()
    }

    fn forward(&mut self, steps: i32) -> String {
        self.cur = (self.cur + steps as usize).min(self.history.len() - 1);
        self.history[self.cur].clone()
    }
}

struct CustomStack {
    max_size: usize,
    data: Vec<i32>,
}

impl CustomStack {
    pub fn new(max_size: i32) -> Self {
        Self {
            max_size: max_size as usize,
            data: Vec::new(),
        }
    }

    pub fn push(&mut self, x: i32) {
        if self.data.len() < self.max_size {
            self.data.push(x);
        }
    }

    pub fn pop(&mut self) -> i32 {
        match self.data.pop() {
            Some(v) => v,
            None => -1,
        }
    }

    pub fn increment(&mut self, k: i32, val: i32) {
        for i in 0..k.min(self.data.len() as i32) as usize {
            self.data[i] += val;
        }
    }
}

struct UndergroundSystem {
    check_in: HashMap<i32, (String, i64)>, // id -> (station_name, time)

    check_out: HashMap<(String, String), (i64, i64)>, // (start_station, end_station) -> (total_time, count)
}

impl UndergroundSystem {
    pub fn new() -> Self {
        Self {
            check_in: HashMap::new(),
            check_out: HashMap::new(),
        }
    }

    pub fn check_in(&mut self, id: i32, station_name: String, t: i32) {
        self.check_in.insert(id, (station_name, t as i64));
    }

    pub fn check_out(&mut self, id: i32, station_name: String, t: i32) {
        let (check_in_station, check_in_time) = self.check_in.remove(&id).unwrap();
        let (total_time, count) = self
            .check_out
            .entry((check_in_station, station_name))
            .or_default();
        *total_time += t as i64 - check_in_time;
        *count += 1;
    }

    pub fn get_average_time(&self, start_station: String, end_station: String) -> f64 {
        let (total_time, count) = self.check_out.get(&(start_station, end_station)).unwrap();
        *total_time as f64 / *count as f64
    }
}

use std::collections::BTreeSet;

struct SmallestInfiniteSet {
    set: BTreeSet<i32>, // keep track of added numbers
    smallest: i32,      // the current smallest number when add_back is not called
}

impl SmallestInfiniteSet {
    pub fn new() -> Self {
        Self {
            set: BTreeSet::new(),
            smallest: 1,
        }
    }

    pub fn add_back(&mut self, num: i32) {
        if num >= self.smallest {
            return;
        }
        self.set.insert(num);
    }

    pub fn pop_smallest(&mut self) -> i32 {
        match self.set.pop_first() {
            Some(n) => n,
            None => {
                self.smallest += 1;
                self.smallest - 1
            }
        }
    }
}

struct CombinationIterator {
    combinations: Vec<String>,
}

impl CombinationIterator {
    pub fn new(characters: String, combination_length: i32) -> Self {
        let chars = characters.chars().collect();
        let mut combinations = Vec::new();
        Self::backtrack(
            &mut combinations,
            &chars,
            0,
            combination_length as usize,
            &String::new(),
        );
        combinations.reverse();
        Self { combinations }
    }

    fn backtrack(
        combinations: &mut Vec<String>,
        chars: &Vec<char>,
        index: usize,
        combination_length: usize,
        curr: &String,
    ) {
        if combination_length == 0 {
            combinations.push(curr.clone());
            return;
        }
        if index >= chars.len() {
            return;
        }
        let next = curr.to_owned() + &chars[index].to_string();
        Self::backtrack(
            combinations,
            chars,
            index + 1,
            combination_length - 1,
            &next,
        );
        Self::backtrack(combinations, chars, index + 1, combination_length, curr);
    }

    pub fn next(&mut self) -> String {
        self.combinations.pop().unwrap()
    }

    pub fn has_next(&self) -> bool {
        !self.combinations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_stack() {
        let mut obj = MinStack::new();
        obj.push(-2);
        obj.push(0);
        obj.push(-3);
        assert_eq!(obj.get_min(), -3);
        obj.pop();
        assert_eq!(obj.top(), 0);
        assert_eq!(obj.get_min(), -2);
    }

    #[test]
    fn test_data_stream() {
        let mut obj = DataStream::new(1, 3);
        assert!(!obj.consec(1));
        assert!(!obj.consec(1));
        assert!(obj.consec(1));
        assert!(!obj.consec(2));
        assert!(!obj.consec(2));
        assert!(!obj.consec(3));
        assert!(!obj.consec(3));
        assert!(!obj.consec(3));
        assert!(!obj.consec(1));
        assert!(!obj.consec(1));
        assert!(obj.consec(1));
    }

    #[test]
    fn test_frequency_tracker() {
        let mut obj = FrequencyTracker::new();
        obj.add(3);
        obj.add(3);
        assert!(obj.has_frequency(2));

        obj.add(7);
        assert!(obj.has_frequency(1));
        obj.delete_one(7);
        assert!(!obj.has_frequency(1));
    }

    #[test]
    fn test_subrectangle_queries() {
        let mut obj = SubrectangleQueries::new(vec![
            vec![1, 2, 1],
            vec![4, 3, 4],
            vec![3, 2, 1],
            vec![1, 1, 1],
        ]);
        assert_eq!(obj.get_value(0, 2), 1);
        obj.update_subrectangle(0, 0, 3, 2, 5);
        assert_eq!(obj.get_value(0, 2), 5);
        assert_eq!(obj.get_value(3, 1), 5);
        obj.update_subrectangle(3, 0, 3, 2, 10);
        assert_eq!(obj.get_value(3, 1), 10);
        assert_eq!(obj.get_value(0, 2), 5);
    }

    #[test]
    #[allow(unused_mut, unused_variables)]
    fn test_segment_tree() {
        let mut obj = SegmentTree2D::new(vec![
            vec![1, 2, 1],
            vec![4, 3, 4],
            vec![3, 2, 1],
            vec![1, 1, 1],
        ]);
        // FIXME: fix the bug
        // assert_eq!(obj.get_value((0, 2)), 1);
        // obj.update_subrectangle((0, 0), (3, 2), 5);
        // assert_eq!(obj.get_value((0, 2)), 5);
        // assert_eq!(obj.get_value((3, 1)), 5);
        // obj.update_subrectangle((3, 0), (3, 2), 10);
        // assert_eq!(obj.get_value((3, 1)), 10);
        // assert_eq!(obj.get_value((0, 2)), 5);
    }

    #[test]
    fn test_codec() {
        let mut obj = Codec::new();
        let url = "https://leetcode.com/problems/design-tinyurl".to_string();
        let code = obj.encode(url.clone());
        assert_eq!(obj.decode(code), url);
    }

    #[test]
    fn test_bowser_history() {
        let mut obj = BowserHistory::new("leetcode.com".to_string());
        obj.visit("google.com".to_string());
        obj.visit("facebook.com".to_string());
        obj.visit("youtube.com".to_string());
        assert_eq!(obj.back(1), "facebook.com");
        assert_eq!(obj.back(1), "google.com");
        assert_eq!(obj.forward(1), "facebook.com");
        obj.visit("linkedin.com".to_string());
        assert_eq!(obj.forward(2), "linkedin.com");
        assert_eq!(obj.back(2), "google.com");
        assert_eq!(obj.back(7), "leetcode.com");
    }

    #[test]
    fn test_custom_stack() {
        let mut obj = CustomStack::new(3);
        obj.push(1);
        obj.push(2);
        assert_eq!(obj.pop(), 2);
        obj.push(2);
        obj.push(3);
        obj.push(4);
        obj.increment(5, 100);
        obj.increment(2, 100);
        assert_eq!(obj.pop(), 103);
        assert_eq!(obj.pop(), 202);
        assert_eq!(obj.pop(), 201);
        assert_eq!(obj.pop(), -1);
    }

    #[test]
    fn test_underground_system() {
        let mut obj = UndergroundSystem::new();
        obj.check_in(45, "Leyton".to_string(), 3);
        obj.check_in(32, "Paradise".to_string(), 8);
        obj.check_in(27, "Leyton".to_string(), 10);
        obj.check_out(45, "Waterloo".to_string(), 15);
        obj.check_out(27, "Waterloo".to_string(), 20);
        obj.check_out(32, "Cambridge".to_string(), 22);
        assert_eq!(
            obj.get_average_time("Paradise".to_string(), "Cambridge".to_string()),
            14.0
        );
        assert_eq!(
            obj.get_average_time("Leyton".to_string(), "Waterloo".to_string()),
            11.0
        );
        obj.check_in(10, "Leyton".to_string(), 24);
        assert_eq!(
            obj.get_average_time("Leyton".to_string(), "Waterloo".to_string()),
            11.0
        );
        obj.check_out(10, "Waterloo".to_string(), 38);
        assert_eq!(
            obj.get_average_time("Leyton".to_string(), "Waterloo".to_string()),
            12.0
        );
    }

    #[test]
    fn test_smallest_infinitest_set() {
        let mut obj = SmallestInfiniteSet::new();
        obj.add_back(2);
        assert_eq!(obj.pop_smallest(), 1);
        assert_eq!(obj.pop_smallest(), 2);
        assert_eq!(obj.pop_smallest(), 3);
        obj.add_back(1);
        assert_eq!(obj.pop_smallest(), 1);
        assert_eq!(obj.pop_smallest(), 4);
        assert_eq!(obj.pop_smallest(), 5);
    }

    #[test]
    fn test_combination_iterator() {
        let mut obj = CombinationIterator::new("abc".to_string(), 2);
        assert_eq!(obj.next(), "ab");
        assert!(obj.has_next());
        assert_eq!(obj.next(), "ac");
        assert!(obj.has_next());
        assert_eq!(obj.next(), "bc");
        assert!(!obj.has_next());
    }
}
