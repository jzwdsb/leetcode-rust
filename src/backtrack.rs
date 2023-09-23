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

    /*
    link: https://leetcode.com/problems/combinations/
    return all the possible combination of k numbers out of 1...n
     */

    pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
        let mut res = vec![];
        Self::backtrack3(&mut res, &mut vec![], n, k);
        res
    }

    fn backtrack3(res: &mut Vec<Vec<i32>>, curr: &mut Vec<i32>, n: i32, k: i32) {
        if k == 0 {
            res.push(curr.clone());
            return;
        }
        for i in curr.last().unwrap_or(&0) + 1..=(n - k + 1) {
            curr.push(i);
            Self::backtrack3(res, curr, n, k - 1);
            curr.pop();
        }
    }

    /*
    https://leetcode.com/problems/n-queens/description/

    n queens problem, place n queens on an n*n chessboard
    return all the possible solution that n queens can be placed on an n*n chessboard
    and no two queens attack each other 
    a queen can attack horizontally, vertically and diagonally
    
    */

    // start from the first row, try to place a queen on each column
    pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
        let mut res = vec![];
        let mut curr = vec![];
        Self::queen_backtrack(&mut res, &mut curr, n);
        res
    }

    fn queen_backtrack(res: &mut Vec<Vec<String>>, curr: &mut Vec<String>, n: i32 ) {
        if curr.len() == n as usize {
            res.push(curr.clone());
            return;
        }

        // try to place a queen on each column
        for i in 0..n {
            if Self::is_valid_queen(curr, i, n) {
                let mut s = vec!['.'; n as usize].iter().collect::<String>();
                s.replace_range(i as usize..i as usize + 1, "Q");
                curr.push(s);
                Self::queen_backtrack(res, curr, n);
                curr.pop();
            }
        }
    }

    // the current position is valid if there is no queen on the same column, same row and same diagonal
    fn is_valid_queen(curr: &Vec<String>, col: i32, n: i32) -> bool {
        let row = curr.len() as i32;
        // check if there multiple Q in the same col
        for i in 0..row {
            if curr[i as usize].chars().nth(col as usize) == Some('Q') {
                return false;
            }
        }
        
        // check if there is multiple Q in the left up diagonal
        let mut i = row - 1;
        let mut j = col - 1;
        while i >= 0 && j >= 0 {
            if curr[i as usize].chars().nth(j as usize) == Some('Q') {
                return false;
            }
            i -= 1;
            j -= 1;
        }
        
        // check if there is multiple Q in the right up diagonal
        let mut i = row - 1;
        let mut j = col + 1;
        while i >= 0 && j < n {
            if curr[i as usize].chars().nth(j as usize) == Some('Q') {
                return false;
            }
            i -= 1;
            j += 1;
        }
        true
    }
}

pub fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_combination() {
        let mut result = BackStrackSolution::combine(4, 2);
        result.sort();
        assert_eq!(
            result,
            vec![
                vec![1, 2],
                vec![1, 3],
                vec![1, 4],
                vec![2, 3],
                vec![2, 4],
                vec![3, 4]
            ]
        );
    }

    #[test]
    fn test_solve_n_queens() {
        assert_eq!(
            BackStrackSolution::solve_n_queens(4),
            vec![
                vec![".Q..", "...Q", "Q...", "..Q."],
                vec!["..Q.", "Q...", "...Q", ".Q.."]
            ]
        );
    }
}
