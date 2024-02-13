#![allow(dead_code)]

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

    fn queen_backtrack(res: &mut Vec<Vec<String>>, curr: &mut Vec<String>, n: i32) {
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
    fn is_valid_queen(curr: &[String], col: i32, n: i32) -> bool {
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

    pub fn total_n_queens(n: i32) -> i32 {
        let mut res = 0;
        let mut curr = vec![];
        Self::queen_backtrack2(&mut res, &mut curr, n);
        res
    }

    fn queen_backtrack2(res: &mut i32, curr: &mut Vec<String>, n: i32) {
        if curr.len() == n as usize {
            *res += 1;
            return;
        }

        // try to place a queen on each column
        for i in 0..n {
            if Self::is_valid_queen(curr, i, n) {
                let mut s = vec!['.'; n as usize].iter().collect::<String>();
                s.replace_range(i as usize..i as usize + 1, "Q");
                curr.push(s);
                Self::queen_backtrack2(res, curr, n);
                curr.pop();
            }
        }
    }

    pub fn subsets(num: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let mut curr = vec![];
        Self::subset_backtrack(&mut res, &mut curr, &num, 0);
        res
    }

    fn subset_backtrack(res: &mut Vec<Vec<i32>>, curr: &mut Vec<i32>, num: &[i32], pos: usize) {
        res.push(curr.to_vec());
        if pos == num.len() {
            return;
        }
        for i in pos..num.len() {
            curr.push(num[i]);
            Self::subset_backtrack(res, curr, num, i + 1);
            curr.pop();
        }
    }

    pub fn subsets_with_dup_ii(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = Vec::new();
        let mut path = Vec::new();
        let mut nums = nums;
        nums.sort_unstable();
        Self::backtrack_ii(&nums, 0, &mut path, &mut res);
        res
    }

    fn backtrack_ii(nums: &[i32], start: usize, path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        res.push(path.clone());
        for i in start..nums.len() {
            if i > start && nums[i] == nums[i - 1] {
                // skip duplicate
                continue;
            }
            path.push(nums[i]);
            Self::backtrack_ii(nums, i + 1, path, res);
            path.pop();
        }
    }

    pub fn solve_sudoko(board: &mut Vec<Vec<char>>) {
        Self::sudoko_backtrack(board, 0, 0);
    }

    fn sudoko_backtrack(board: &mut Vec<Vec<char>>, row: usize, col: usize) -> bool {
        // all the row and col are filled
        if row == 9 {
            return true;
        }
        // if the current row is filled, go to the next position
        if col == 9 {
            return Self::sudoko_backtrack(board, row + 1, 0);
        }
        // if the current position is not empty, go to the next position
        if board[row][col] != '.' {
            return Self::sudoko_backtrack(board, row, col + 1);
        }
        // try to fill the current position with 1 to 9
        for i in 1..=9 {
            if Self::is_valid_sudoko(board, row, col, i) {
                board[row][col] = i.to_string().chars().next().unwrap();
                // if the next position is filled, return true
                if Self::sudoko_backtrack(board, row, col + 1) {
                    return true;
                } else {
                    // else rollback the current position
                    board[row][col] = '.';
                }
            }
        }
        false
    }

    fn is_valid_sudoko(board: &[Vec<char>], row: usize, col: usize, num: i32) -> bool {
        let num = (num as u8 + b'0') as char;
        for i in 0..9 {
            if board[row][i] == num {
                return false;
            }
            if board[i][col] == num {
                return false;
            }
            // check the 3*3 block
            // for row, col. the upper left corner is (row / 3 * 3, col / 3 * 3)
            // for the ith element in the block
            // the position is (row / 3 * 3 + i / 3, col / 3 * 3 + i % 3)
            if board[row / 3 * 3 + i / 3][col / 3 * 3 + i % 3] == num {
                return false;
            }
        }
        true
    }

    pub fn restore_ip_addresses(s: String) -> Vec<String> {
        let mut res = Vec::new();
        let mut path = Vec::new();
        Self::ip_backtrack(&s, 0, &mut path, &mut res);
        res
    }

    fn ip_backtrack(s: &str, start: usize, path: &mut Vec<String>, res: &mut Vec<String>) {
        // if the path is already 4 and the start is already the end of the string
        // push the path to the result and return
        if path.len() == 4 && start == s.len() {
            res.push(path.join("."));
            return;
        }

        // if the path is already 4 or the start is already the end of the string
        // return
        if path.len() == 4 || start == s.len() {
            return;
        }
        for i in start..s.len() {
            // the number could not start with 0
            if i > start && s.chars().nth(start) == Some('0') {
                break;
            }
            // if the number is larger than 255, break
            let num = s[start..=i].parse::<i32>().unwrap();
            if num > 255 {
                break;
            }
            path.push(num.to_string());
            Self::ip_backtrack(s, i + 1, path, res);
            path.pop();
        }
    }
}

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

    #[test]
    fn test_solve_n_queen_ii() {
        assert_eq!(BackStrackSolution::total_n_queens(4), 2);
    }

    #[test]
    fn test_subset() {
        let mut result = BackStrackSolution::subsets(vec![1, 2, 3]);
        result.sort();
        assert_eq!(
            result,
            vec![
                vec![],
                vec![1],
                vec![1, 2],
                vec![1, 2, 3],
                vec![1, 3],
                vec![2],
                vec![2, 3],
                vec![3]
            ]
        );
    }

    #[test]
    fn test_solve_sudoko() {
        let mut board = vec![
            vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
            vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
            vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
            vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
            vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
            vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
            vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
            vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
            vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
        ];
        BackStrackSolution::solve_sudoko(&mut board);
        assert_eq!(
            board,
            vec![
                vec!['5', '3', '4', '6', '7', '8', '9', '1', '2'],
                vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
                vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
                vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
                vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
                vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
                vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
                vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
                vec!['3', '4', '5', '2', '8', '6', '1', '7', '9'],
            ]
        );
    }

    #[test]
    fn test_restore_ip_addresses() {
        let mut result = BackStrackSolution::restore_ip_addresses("25525511135".to_string());
        result.sort();
        let mut expect = vec!["255.255.111.35", "255.255.11.135"];
        expect.sort();
        assert_eq!(result, expect);
    }

    #[test]
    fn test_subsets_with_dup_ii() {
        let mut result = BackStrackSolution::subsets_with_dup_ii(vec![1, 2, 2]);
        result.sort();
        assert_eq!(
            result,
            vec![
                vec![],
                vec![1],
                vec![1, 2],
                vec![1, 2, 2],
                vec![2],
                vec![2, 2]
            ]
        );
    }
}
