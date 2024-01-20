#![allow(dead_code)]

use crate::tree::{Tree, TreeNode};
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
    rc::Rc,
    str::Split,
};

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
        let (check_in_station, check_in_time) = self.check_in.remove(&id).expect("no such id");
        let (total_time, count) = self
            .check_out
            .entry((check_in_station, station_name))
            .or_default();
        *total_time += t as i64 - check_in_time;
        *count += 1;
    }

    pub fn get_average_time(&self, start_station: String, end_station: String) -> f64 {
        let (total_time, count) = self
            .check_out
            .get(&(start_station, end_station))
            .expect("no such station");
        *total_time as f64 / *count as f64
    }
}

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

struct FindElements {
    root: Tree,
    sets: HashSet<i32>,
}

impl FindElements {
    fn new(root: Tree) -> Self {
        match root {
            None => Self {
                root: None,
                sets: HashSet::new(),
            },
            Some(root) => {
                root.borrow_mut().val = 0;
                let mut obj = Self {
                    root: Some(root.clone()),
                    sets: HashSet::new(),
                };
                obj.recover_tree(Some(root.clone()));
                obj
            }
        }
    }

    fn recover_tree(&mut self, root: Tree) {
        match root {
            None => {}
            Some(root) => {
                let root = root.borrow_mut();
                self.sets.insert(root.val);
                if root.left.is_some() {
                    root.left.as_ref().unwrap().borrow_mut().val = root.val * 2 + 1;
                    self.recover_tree(root.left.clone());
                }
                if root.right.is_some() {
                    root.right.as_ref().unwrap().borrow_mut().val = root.val * 2 + 2;
                    self.recover_tree(root.right.clone());
                }
            }
        }
    }

    pub fn find(&self, target: i32) -> bool {
        self.sets.contains(&target)
    }

    #[deprecated = "use sets instead"]
    fn find_in_tree(root: Tree, target: i32) -> bool {
        match root {
            None => false,
            Some(root) => {
                let root = root.borrow();
                if root.val == target {
                    return true;
                }
                Self::find_in_tree(root.left.clone(), target)
                    || Self::find_in_tree(root.right.clone(), target)
            }
        }
    }
}

struct LRUCache {
    capacity: usize,
    cache: HashMap<i32, i32>,
    queue: VecDeque<i32>,
}

impl LRUCache {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            cache: HashMap::new(),
            queue: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: i32) -> i32 {
        match self.cache.get(&key) {
            Some(v) => {
                let idx = self
                    .queue
                    .iter()
                    .position(|&x| x == key)
                    .expect("key not found");
                self.queue.remove(idx);
                self.queue.push_back(key);
                *v
            }
            None => -1,
        }
    }

    pub fn put(&mut self, key: i32, value: i32) {
        match self.cache.contains_key(&key) {
            true => {
                let idx = self
                    .queue
                    .iter()
                    .position(|&x| x == key)
                    .expect("key not found");
                self.queue.remove(idx);
                self.queue.push_back(key);
                self.cache.insert(key, value);
            }
            false => {
                if self.queue.len() == self.capacity {
                    let key = self.queue.pop_front().unwrap();
                    self.cache.remove(&key);
                }
                self.queue.push_back(key);
                self.cache.insert(key, value);
            }
        }
    }
}

struct BSTIterator {
    stack: Vec<Tree>,
}

impl BSTIterator {
    fn new(root: Tree) -> Self {
        let mut stack = Vec::new();
        let mut root = root;
        while root.is_some() {
            stack.push(root.clone());
            root = root.unwrap().borrow().left.clone();
        }

        Self { stack }
    }

    fn next(&mut self) -> i32 {
        let mut root = self.stack.pop().expect("stack is empty");
        let val = root.as_ref().unwrap().borrow().val;
        root = root.unwrap().borrow().right.clone();
        while root.is_some() {
            self.stack.push(root.clone());
            root = root.unwrap().borrow().left.clone();
        }
        val
    }

    fn has_next(&self) -> bool {
        !self.stack.is_empty()
    }
}

struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_word: bool,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_word: false,
        }
    }
}

struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    fn insert(&mut self, word: String) {
        let mut node = &mut self.root;
        for c in word.chars() {
            node = node.children.entry(c).or_insert(TrieNode::new());
        }
        node.is_word = true;
    }

    fn search(&self, word: String) -> bool {
        let mut node = &self.root;
        for c in word.chars() {
            match node.children.get(&c) {
                Some(v) => node = v,
                None => return false,
            }
        }
        node.is_word
    }

    fn starts_with(&self, prefix: String) -> bool {
        let mut node = &self.root;
        for c in prefix.chars() {
            match node.children.get(&c) {
                Some(v) => node = v,
                None => return false,
            }
        }
        true
    }
}

// implement stack using queue
struct MyStackUnderQueue {
    queue: VecDeque<i32>,
}

impl MyStackUnderQueue {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    fn push(&mut self, x: i32) {
        self.queue.push_back(x);
        for _ in 0..self.queue.len() - 1 {
            let x = self.queue.pop_front().unwrap();
            self.queue.push_back(x);
        }
    }

    fn pop(&mut self) -> i32 {
        self.queue.pop_front().expect("queue is empty")
    }

    fn top(&self) -> i32 {
        *self.queue.front().expect("queue is empty")
    }

    fn empty(&self) -> bool {
        self.queue.is_empty()
    }
}

struct MyQueue {
    stack1: Vec<i32>,
    stack2: Vec<i32>,
}

impl MyQueue {
    fn new() -> Self {
        Self {
            stack1: Vec::new(),
            stack2: Vec::new(),
        }
    }

    fn push(&mut self, x: i32) {
        self.stack1.push(x);
    }

    fn pop(&mut self) -> i32 {
        match self.stack2.pop() {
            Some(v) => v,
            None => {
                while let Some(v) = self.stack1.pop() {
                    self.stack2.push(v);
                }
                self.stack2.pop().expect("stack is empty")
            }
        }
    }

    fn peek(&mut self) -> i32 {
        match self.stack2.last() {
            Some(v) => *v,
            None => {
                while let Some(v) = self.stack1.pop() {
                    self.stack2.push(v);
                }
                *self.stack2.last().expect("stack is empty")
            }
        }
    }

    fn empty(&self) -> bool {
        self.stack1.is_empty() && self.stack2.is_empty()
    }
}

struct NumArray {
    nums: Vec<i32>,
    sums: Vec<i32>,
}

impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        Self {
            nums: nums.clone(),
            sums: nums
                .iter()
                .scan(0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect(),
        }
    }

    fn sum_range(&self, left: usize, right: usize) -> i32 {
        self.sums[right] - self.sums[left] + self.nums[left]
    }
}

struct WordDictionary {
    root: TrieNode,
}

impl WordDictionary {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    pub fn add_word(&mut self, word: String) {
        let mut node = &mut self.root;
        for c in word.chars() {
            node = node.children.entry(c).or_insert(TrieNode::new());
        }
        node.is_word = true;
    }

    fn search_in_trie(node: &TrieNode, word: &[u8]) -> bool {
        if word.is_empty() {
            return node.is_word;
        }
        match word[0] {
            b'.' => node
                .children
                .values()
                .any(|v| Self::search_in_trie(v, &word[1..])),
            c => match node.children.get(&(c as char)) {
                Some(v) => Self::search_in_trie(v, &word[1..]),
                None => false,
            },
        }
    }

    pub fn search(&self, word: String) -> bool {
        Self::search_in_trie(&self.root, word.as_bytes())
    }
}

struct MedianFinder {
    low: BinaryHeap<i32>,           // the smaller half of the numbers
    high: BinaryHeap<Reverse<i32>>, // the lagrer half of the numbers
}

impl MedianFinder {
    pub fn new() -> Self {
        Self {
            high: BinaryHeap::new(),
            low: BinaryHeap::new(),
        }
    }

    pub fn add_num(&mut self, num: i32) {
        self.low.push(num);
        self.high.push(Reverse(self.low.pop().unwrap()));
        if self.high.len() > self.low.len() {
            self.low.push(self.high.pop().unwrap().0);
        }
    }

    pub fn find_median(&self) -> f64 {
        if self.low.len() > self.high.len() {
            *self.low.peek().unwrap() as f64
        } else {
            (*self.low.peek().unwrap() as f64 + self.high.peek().unwrap().0 as f64) / 2.0
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Tweet {
    publisher: i32,
    tweet_id: i32,
    order: i32,
}

struct Twitter {
    user_tweet: HashMap<i32, Vec<Tweet>>, // user -> tweets(user, tweet_id)
    followees: HashMap<i32, HashSet<i32>>, // followee -> followers
    feeds: HashMap<i32, Vec<Tweet>>,      // user -> tweets(user, tweet_id)
    order: i32,                           // the global order of tweets
}

impl Twitter {
    pub fn new() -> Self {
        Self {
            user_tweet: HashMap::new(),
            followees: HashMap::new(), // followee -> followers
            feeds: HashMap::new(),
            order: 0,
        }
    }

    pub fn post_tweet(&mut self, user_id: i32, tweet_id: i32) {
        let new_tweet = Tweet {
            publisher: user_id,
            tweet_id,
            order: self.order,
        };
        self.order += 1;
        self.user_tweet.entry(user_id).or_default().push(new_tweet);
        self.order += 1;
        self.feeds.entry(user_id).or_default().push(new_tweet);
        for followee in self.followees.get(&user_id).unwrap_or(&HashSet::new()) {
            self.feeds.entry(*followee).or_default().push(new_tweet);
        }
    }

    pub fn get_news_feed(&self, user_id: i32) -> Vec<i32> {
        self.feeds
            .get(&user_id)
            .unwrap_or(&Vec::new())
            .iter()
            .rev()
            .map(|t| t.tweet_id)
            .take(10)
            .collect()
    }

    pub fn follow(&mut self, follower_id: i32, followee_id: i32) {
        if self
            .followees
            .entry(followee_id)
            .or_default()
            .contains(&follower_id)
        {
            return;
        }
        if follower_id == followee_id {
            return;
        }

        self.followees
            .entry(followee_id)
            .or_default()
            .insert(follower_id);
        self.feeds
            .entry(follower_id)
            .or_default()
            .extend(self.user_tweet.get(&followee_id).unwrap_or(&Vec::new()));
        self.feeds
            .entry(follower_id)
            .or_default()
            .sort_by_key(|t| t.order);
    }

    pub fn unfollow(&mut self, follower_id: i32, followee_id: i32) {
        if !self
            .followees
            .entry(followee_id)
            .or_default()
            .contains(&follower_id)
        {
            return;
        }
        if follower_id == followee_id {
            return;
        }
        self.followees
            .entry(followee_id)
            .or_default()
            .remove(&follower_id);
        self.feeds
            .entry(follower_id)
            .or_default()
            .retain(|t| t.publisher != followee_id);
    }
}

struct KthLargest {
    k: usize,
    nums: BinaryHeap<Reverse<i32>>,
}

impl KthLargest {
    fn new(k: i32, nums: Vec<i32>) -> Self {
        let mut obj = Self {
            k: k as usize,
            nums: BinaryHeap::with_capacity(k as usize + 1),
        };
        for n in nums {
            obj.nums.push(Reverse(n));
        }

        obj
    }

    fn add(&mut self, val: i32) -> i32 {
        self.nums.push(Reverse(val));
        while self.nums.len() > self.k {
            self.nums.pop();
        }
        self.nums.peek().unwrap().0
    }
}

// serialize tree to string, format: [1,2,3,null,null,4,5]
// deserialize string to tree
struct TreeCodec {}

impl TreeCodec {
    pub fn new() -> Self {
        Self {}
    }

    pub fn serialize(&self, root: Tree) -> String {
        let mut res = vec![];
        Self::serialize_helper(&root, &mut res);
        // join the vector with ,
        format!("[{}]", res.join(","))
    }

    fn serialize_helper(root: &Tree, res: &mut Vec<String>) {
        match root {
            None => {
                res.push("null".to_string());
            }
            Some(root) => {
                res.push(root.borrow().val.to_string());
                Self::serialize_helper(&root.borrow().left, res);
                Self::serialize_helper(&root.borrow().right, res);
            }
        }
    }

    pub fn deserialize(&self, data: String) -> Tree {
        // remove [ and ]
        let data = data.trim_start_matches('[').trim_end_matches(']');
        let mut iter = data.split(',');

        Self::deserialize_helper(&mut iter)
    }

    fn deserialize_helper(iter: &mut Split<char>) -> Tree {
        match iter.next().unwrap() {
            "null" => None,
            val => {
                let mut root = TreeNode::new(val.parse().unwrap());
                root.left = Self::deserialize_helper(iter);
                root.right = Self::deserialize_helper(iter);
                Some(Rc::new(RefCell::new(root)))
            }
        }
    }
}

struct LFUCache {
    capacity: usize,
    cache: HashMap<i32, i32>,               // key -> value
    freq: HashMap<i32, i32>,                // key -> freq
    freq_cache: HashMap<i32, HashSet<i32>>, // freq -> keys
    min_freq: i32,                          // the min freq in the cache
}

impl LFUCache {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            cache: HashMap::new(),
            freq: HashMap::new(),
            freq_cache: HashMap::new(),
            min_freq: 0,
        }
    }

    pub fn get(&mut self, key: i32) -> i32 {
        if !self.cache.contains_key(&key) {
            return -1;
        }
        self.update_freq(key);
        *self.cache.get(&key).unwrap()
    }

    fn update_freq(&mut self, key: i32) {
        let freq = *self.freq.get(&key).unwrap();
        self.freq_cache.entry(freq).or_default().remove(&key);
        self.freq_cache.entry(freq + 1).or_default().insert(key);
        self.freq.insert(key, freq + 1);
        if freq == self.min_freq && self.freq_cache.entry(freq).or_default().is_empty() {
            self.min_freq += 1;
        }
    }

    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        if self.cache.contains_key(&key) {
            self.cache.insert(key, value);
            self.update_freq(key);
            return;
        }
        if self.cache.len() == self.capacity {
            self.remove_min_freq();
        }
        self.cache.insert(key, value);
        self.freq.insert(key, 1);
        self.freq_cache.entry(1).or_default().insert(key);
        self.min_freq = 1;
    }

    fn remove_min_freq(&mut self) {
        let key = self
            .freq_cache
            .entry(self.min_freq)
            .or_default()
            .iter()
            .next()
            .unwrap()
            .clone();
        self.cache.remove(&key);
        self.freq.remove(&key);
        self.freq_cache
            .entry(self.min_freq)
            .or_default()
            .remove(&key);
        if self.freq_cache.entry(self.min_freq).or_default().is_empty() {
            self.min_freq += 1;
        }
    }
}

struct CirularQueue {
    data: Vec<i32>,
    head: usize, // the index of the first element
    tail: usize, // the index of the last element
    size: usize, // the number of elements in the queue
    capacity: usize,
}

impl CirularQueue {
    fn new(k: i32) -> Self {
        Self {
            data: vec![0; k as usize],
            head: 0,
            tail: 0,
            size: 0,
            capacity: k as usize,
        }
    }

    fn en_queue(&mut self, value: i32) -> bool {
        if self.is_full() {
            return false;
        }
        // if the queue is empty, set the head to the first element
        // don't have to update tail when there is one element
        // otherwise, set the tail to the next element
        if !self.is_empty() {
            self.tail = (self.tail + 1) % self.capacity;
        }

        self.data[self.tail] = value;
        self.size += 1;
        true
    }

    fn de_queue(&mut self) -> bool {
        if self.is_empty() {
            return false;
        }

        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        if self.is_empty() {
            self.head = 0;
            self.tail = 0;
        }
        true
    }

    fn front(&self) -> i32 {
        if self.is_empty() {
            return -1;
        }
        self.data[self.head]
    }

    fn rear(&self) -> i32 {
        if self.is_empty() {
            return -1;
        }
        self.data[self.tail]
    }

    // empty when head == tail
    fn is_empty(&self) -> bool {
        self.size == 0
    }

    // full when tail is one slot before head
    fn is_full(&self) -> bool {
        self.size == self.capacity
    }
}

struct SnapshotArray {
    data: Vec<BTreeMap<i32, i32>>, // data[i] -> (snap_id, val)
    snap_id: i32,
}

impl SnapshotArray {
    pub fn new(length: i32) -> Self {
        Self {
            data: vec![BTreeMap::new(); length as usize],
            snap_id: 0,
        }
    }

    pub fn set(&mut self, index: i32, val: i32) {
        self.data[index as usize].insert(self.snap_id, val);
    }

    pub fn snap(&mut self) -> i32 {
        self.snap_id += 1;
        self.snap_id - 1
    }

    pub fn get(&self, index: i32, snap_id: i32) -> i32 {
        match self.data[index as usize].range(..=snap_id).next_back() {
            Some((_, v)) => *v,
            None => 0,
        }
    }
}

// get product of last key numbers
struct ProductOfNumbers {
    data: Vec<i32>,
}

impl ProductOfNumbers {
    pub fn new() -> Self {
        Self { data: vec![1] }
    }

    pub fn add(&mut self, num: i32) {
        if num == 0 {
            self.data = vec![1];
        } else {
            self.data.push(self.data.last().unwrap() * num);
        }
    }

    pub fn get_product(&self, k: i32) -> i32 {
        if k >= self.data.len() as i32 {
            return 0;
        }
        self.data.last().unwrap() / self.data[self.data.len() - k as usize - 1]
    }
}

struct ParkingSystem {
    spaces: Vec<i32>,
}

impl ParkingSystem {
    pub fn new(big: i32, medium: i32, small: i32) -> Self {
        Self {
            spaces: vec![big, medium, small],
        }
    }

    pub fn add_car(&mut self, car_type: i32) -> bool {
        if self.spaces[car_type as usize - 1] > 0 {
            self.spaces[car_type as usize - 1] -= 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tree::TreeNode;

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

    #[test]
    fn test_find_elements() {
        let root = TreeNode::from_vec(vec![Some(-1), None, Some(-1)]);
        let obj = FindElements::new(root);
        assert!(!obj.find(1));
        assert!(obj.find(2));
        assert!(!obj.find(3));
    }

    #[test]
    fn test_lru_cache() {
        let mut obj = LRUCache::new(2);
        obj.put(1, 1);
        obj.put(2, 2);
        assert_eq!(obj.get(1), 1);
        obj.put(3, 3);
        assert_eq!(obj.get(2), -1);
        obj.put(4, 4);
        assert_eq!(obj.get(1), -1);
        assert_eq!(obj.get(3), 3);
        assert_eq!(obj.get(4), 4);
    }

    #[test]
    fn test_bst_iterator() {
        let root = TreeNode::from_vec(vec![
            Some(7),
            Some(3),
            Some(15),
            None,
            None,
            Some(9),
            Some(20),
        ]);
        let mut obj = BSTIterator::new(root);
        assert_eq!(obj.next(), 3);
        assert_eq!(obj.next(), 7);
        assert!(obj.has_next());
        assert_eq!(obj.next(), 9);
        assert!(obj.has_next());
        assert_eq!(obj.next(), 15);
        assert!(obj.has_next());
        assert_eq!(obj.next(), 20);
        assert!(!obj.has_next());
    }

    #[test]
    fn test_trie() {
        let mut obj = Trie::new();
        obj.insert("apple".to_string());
        assert!(obj.search("apple".to_string()));
        assert!(!obj.search("app".to_string()));
        assert!(obj.starts_with("app".to_string()));
        obj.insert("app".to_string());
        assert!(obj.search("app".to_string()));
    }

    #[test]
    fn test_mystack() {
        let mut obj = MyStackUnderQueue::new();
        obj.push(1);
        obj.push(2);
        assert_eq!(obj.top(), 2);
        assert_eq!(obj.pop(), 2);
        assert!(!obj.empty());
    }

    #[test]
    fn test_myqueue() {
        let mut obj = MyQueue::new();
        obj.push(1);
        obj.push(2);
        assert_eq!(obj.peek(), 1);
        assert_eq!(obj.pop(), 1);
        assert!(!obj.empty());
    }

    #[test]
    fn test_num_array() {
        let obj = NumArray::new(vec![-2, 0, 3, -5, 2, -1]);
        assert_eq!(obj.sum_range(0, 2), 1);
        assert_eq!(obj.sum_range(2, 5), -1);
        assert_eq!(obj.sum_range(0, 5), -3);
    }

    #[test]
    fn test_word_dictionary() {
        let mut obj = WordDictionary::new();
        obj.add_word("bad".to_string());
        obj.add_word("dad".to_string());
        obj.add_word("mad".to_string());
        assert!(!obj.search("pad".to_string()));
        assert!(obj.search("bad".to_string()));
        assert!(obj.search(".ad".to_string()));
        assert!(obj.search("b..".to_string()));
    }

    #[test]
    fn test_find_median() {
        let mut obj = MedianFinder::new();
        obj.add_num(1);
        obj.add_num(2);
        assert_eq!(obj.find_median(), 1.5);
        obj.add_num(3);
        assert_eq!(obj.find_median(), 2.0);
    }

    #[test]
    fn test_tweet() {
        let mut obj = Twitter::new();
        obj.post_tweet(1, 5);
        assert_eq!(obj.get_news_feed(1), vec![5]);
        obj.follow(1, 2);
        obj.post_tweet(2, 6);
        assert_eq!(obj.get_news_feed(1), vec![6, 5]);
        obj.unfollow(1, 2);
        assert_eq!(obj.get_news_feed(1), vec![5]);

        let mut obj = Twitter::new();
        obj.post_tweet(2, 5);
        obj.follow(1, 2);
        obj.follow(1, 2); // duplicate follow
        assert_eq!(obj.get_news_feed(1), vec![5]);

        let mut obj = Twitter::new();
        obj.post_tweet(1, 4);
        obj.post_tweet(2, 5);
        obj.unfollow(1, 2);
        assert_eq!(obj.get_news_feed(1), vec![4]);

        let mut obj = Twitter::new();
        obj.post_tweet(1, 5);
        obj.post_tweet(1, 3);
        obj.post_tweet(1, 101);
        obj.post_tweet(1, 13);
        obj.post_tweet(1, 10);
        obj.post_tweet(1, 2);
        obj.post_tweet(1, 94);
        obj.post_tweet(1, 505);
        obj.post_tweet(1, 333);
        obj.post_tweet(1, 22);
        obj.post_tweet(1, 11);
        assert_eq!(
            obj.get_news_feed(1),
            vec![11, 22, 333, 505, 94, 2, 10, 13, 101, 3]
        );
    }

    #[test]
    fn test_kth_largest() {
        let mut obj = KthLargest::new(3, vec![4, 5, 8, 2]);
        assert_eq!(obj.add(3), 4);
        assert_eq!(obj.add(5), 5);
        assert_eq!(obj.add(10), 5);
        assert_eq!(obj.add(9), 8);
        assert_eq!(obj.add(4), 8);
    }

    #[test]
    fn test_tree_codec() {
        let obj = TreeCodec::new();
        let root = TreeNode::from_vec(vec![
            Some(1),
            Some(2),
            Some(3),
            None,
            None,
            Some(4),
            Some(5),
        ]);
        let serialized = obj.serialize(root.clone());
        assert_eq!(serialized, "[1,2,null,null,3,4,null,null,5,null,null]");
        assert_eq!(
            obj.serialize(obj.deserialize(serialized.clone())),
            serialized
        );
    }

    #[test]
    fn test_lfu_cahce() {
        let mut obj = LFUCache::new(2);
        obj.put(1, 1);
        obj.put(2, 2);
        assert_eq!(obj.get(1), 1);
        obj.put(3, 3);
        assert_eq!(obj.get(2), -1);
        assert_eq!(obj.get(3), 3);
        obj.put(4, 4);
        // random delete 1 or 3 because they have the same frequency
        assert!(obj.get(1) == -1 || obj.get(3) == -1);
        assert_eq!(obj.get(4), 4);
    }

    #[test]
    fn test_circul_queue() {
        let mut obj = CirularQueue::new(3);
        assert!(obj.en_queue(1));
        assert!(obj.en_queue(2));
        assert!(obj.en_queue(3));
        assert!(!obj.en_queue(4));
        assert_eq!(obj.rear(), 3);
        assert!(obj.is_full());
        assert!(obj.de_queue());
        assert!(obj.en_queue(4));
        assert_eq!(obj.rear(), 4);

        let mut obj = CirularQueue::new(3);
        assert!(obj.en_queue(2));
        assert_eq!(obj.rear(), 2);
        assert_eq!(obj.front(), 2);
        assert!(obj.de_queue());
        assert!(obj.is_empty());
        assert_eq!(obj.front(), -1);
        assert_eq!(obj.rear(), -1);

        let mut obj = CirularQueue::new(81);
        assert!(obj.en_queue(69));
        assert!(obj.de_queue());
        assert!(obj.en_queue(92));
        assert!(obj.en_queue(12));
        assert!(obj.de_queue());
        assert!(!obj.is_full());
        assert!(!obj.is_full());
        assert_eq!(obj.front(), 12);
    }

    #[test]
    fn test_snapshot_array() {
        let mut obj = SnapshotArray::new(3);
        obj.set(0, 5);
        assert_eq!(obj.snap(), 0);
        obj.set(0, 6);
        assert_eq!(obj.get(0, 0), 5);

        let mut obj = SnapshotArray::new(1);
        obj.set(0, 15);
        assert_eq!(obj.snap(), 0);
        assert_eq!(obj.snap(), 1);
        assert_eq!(obj.snap(), 2);
        assert_eq!(obj.get(0, 2), 15);
        assert_eq!(obj.snap(), 3);
        assert_eq!(obj.snap(), 4);
        assert_eq!(obj.get(0, 0), 15);
    }

    #[test]
    fn test_product_of_numbers() {
        let mut obj = ProductOfNumbers::new();
        obj.add(3);
        obj.add(0);
        obj.add(2);
        obj.add(5);
        obj.add(4);
        assert_eq!(obj.get_product(2), 20);
        assert_eq!(obj.get_product(3), 40);
        assert_eq!(obj.get_product(4), 0);
        obj.add(8);
        assert_eq!(obj.get_product(2), 32);
    }

    #[test]
    fn test_parking_system() {
        let mut obj = ParkingSystem::new(1, 1, 0);
        assert!(obj.add_car(1));
        assert!(obj.add_car(2));
        assert!(!obj.add_car(3));
        assert!(!obj.add_car(1));
    }
}
