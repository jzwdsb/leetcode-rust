pub struct BackStrackSolution {}

impl BackStrackSolution {
    /*
    link: https://leetcode.com/problems/combination-sum/description/
    solve this problem by backtracking
    for one candidate, we can choose it or not choose it, have two branches
    we can use recursion to solve it
    save the path until the target is 0, then push the path to the result
     */

    pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut result = vec![];
        let mut path = vec![];
        Self::backtrack(&mut result, &mut path, &candidates, target, 0);
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
            Self::backtrack(result, path, candidates, target - candidates[i], i);
            path.pop();
        }
        return;
    }

    /*
    link: https://leetcode.com/problems/combination-sum-ii/
    solve this problem by backtracking but need to sort the candidates first and skip the same value
    basically the same as the previous one
     */

    pub fn combination_sum2(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut result = vec![];
        let mut path = vec![];
        let mut candidates = candidates;
        candidates.sort();
        Self::backtrack2(&mut result, &mut path, &candidates, target, 0);
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
            if i > start && candidates[i] == candidates[i - 1] {
                continue;
            }
            if candidates[i] > target {
                break;
            }
            path.push(candidates[i]);
            Self::backtrack2(result, path, candidates, target - candidates[i], i + 1);
            path.pop();
        }
        return;
    }
}

pub fn main() {}

#[test]
fn test_combination_sum() {
    let mut result = BackStrackSolution::combination_sum(vec![2, 3, 6, 7], 7);
    result.sort();
    assert_eq!(result, vec![vec![2, 2, 3], vec![7]]);
    let mut result = BackStrackSolution::combination_sum(vec![2, 3, 5], 8);
    result.sort();
    assert_eq!(result, vec![vec![2, 2, 2, 2], vec![2, 3, 3], vec![3, 5]]);
}

#[test]
fn test_combination_sum2() {
    let mut result = BackStrackSolution::combination_sum2(vec![10, 1, 2, 7, 6, 1, 5], 8);
    result.sort();
    assert_eq!(
        result,
        vec![vec![1, 1, 6], vec![1, 2, 5], vec![1, 7], vec![2, 6]]
    );
    let mut result = BackStrackSolution::combination_sum2(vec![2, 5, 2, 1, 2], 5);
    result.sort();
    assert_eq!(result, vec![vec![1, 2, 2], vec![5]]);
}
