pub struct ArrarySolution {}

impl ArrarySolution {
    /*
    link: https://leetcode.com/problems/next-permutation/
    next lexicographically greater permutation of its integer.
    operation must be done in place and space complexity must be O(1)

    we can use this algorithm to solve this problem:
    https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
    steps:
    1.Find the largest index k such that a[k] < a[k + 1]. If no such index exists, the permutation is the last permutation.
    2. Find the largest index l greater than k such that a[k] < a[l].
    3. Swap the value of a[k] with that of a[l].
    4. Reverse the sequence from a[k + 1] up to and including the final element a[n].

    time complexity: O(n) space complexity: O(1)
    use
     */

    pub fn next_permutation(nums: &mut Vec<i32>) {
        // rustic way: use windows and rposition to find the largest index k such that a[k] < a[k + 1]
        if let Some(k) = nums.windows(2).rposition(|w| w[0] < w[1]) {
            // use rposition to find the largest index l greater than k such that a[k] < a[l]
            let j = nums.iter().rposition(|&x| x > nums[k]).unwrap();
            nums.swap(k, j);
            nums[k + 1..].reverse();
        } else {
            nums.reverse();
        }
    }
    /*
    link: https://leetcode.com/problems/combination-sum/description/

     */

    pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut result = vec![];
        let mut path = vec![];
        ArrarySolution::backtrack(&mut result, &mut path, &candidates, target, 0);
        result
    }
    fn backtrack(
        result: &mut Vec<Vec<i32>>,
        path: &mut Vec<i32>,
        candidates: &[i32],
        target: i32,
        start: usize,
    ) {
        if target < 0 {
            return;
        }
        if target == 0 {
            result.push(path.clone());
        }
        for i in start..candidates.len() {
            path.push(candidates[i]);
            ArrarySolution::backtrack(result, path, candidates, target-candidates[i], i);
            path.pop();
        }
        return 
    }

    pub fn combination_sum2(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut result = vec![];
        let mut path = vec![];
        let mut candidates = candidates;
        candidates.sort();
        ArrarySolution::backtrack2(&mut result, &mut path, &candidates, target, 0);
        result
    }

    fn backtrack2(
        result: &mut Vec<Vec<i32>>,
        path: &mut Vec<i32>,
        candidates: &[i32],
        target: i32,
        start: usize,
    ) {
        if target < 0 {
            return;
        }
        if target == 0 {
            result.push(path.clone());
        }
        for i in start..candidates.len() {
            // 
            if i > start && candidates[i] == candidates[i-1] {
                continue;
            }
            if candidates[i] > target {
                break
            }
            path.push(candidates[i]);
            ArrarySolution::backtrack2(result, path, candidates, target-candidates[i], i+1);
            path.pop();
        }
        return
    }
}

#[test]
fn test_next_permutation() {
    let mut input = vec![1];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1]);
    let mut input = vec![1, 2, 3];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1, 3, 2]);
    let mut input = vec![3, 2, 1];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1, 2, 3]);
    let mut input = vec![1, 1, 5];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1, 5, 1]);
}

#[test]
fn test_combination_sum() {
    let mut result = ArrarySolution::combination_sum(vec![2, 3, 6, 7], 7);
    result.sort();
    assert_eq!(result, vec![vec![2, 2, 3], vec![7]]);
    let mut result = ArrarySolution::combination_sum(vec![2, 3, 5], 8);
    result.sort();
    assert_eq!(result, vec![vec![2, 2, 2, 2], vec![2, 3, 3], vec![3, 5]]);
}

#[test]
fn test_combination_sum2() {
    let mut result = ArrarySolution::combination_sum2(vec![10, 1, 2, 7, 6, 1, 5], 8);
    result.sort();
    assert_eq!(result, vec![vec![1, 1, 6], vec![1, 2, 5], vec![1, 7], vec![2, 6]]);
    let mut result = ArrarySolution::combination_sum2(vec![2, 5, 2, 1, 2], 5);
    result.sort();
    assert_eq!(result, vec![vec![1, 2, 2], vec![5]]);
}