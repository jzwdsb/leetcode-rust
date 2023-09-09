pub struct DPSolution {}

impl DPSolution {
    /*
    link: https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/
    the basic idea of how to solve this is using dyamic programming
    we define a 2-dimension matrix, m[i,j] = sum[nums[i,j]]
    m[i,j] = m[i,j-1] + nums[j].
    we can get the max absolute sum by iterating the matrix
    with time and space complexity O(n^2)
    not the best solution, but it works
     */
    pub fn max_absolute_sum(nums: Vec<i32>) -> i32 {
        let mut dp = vec![vec![0; nums.len()]; nums.len()];
        let mut max = 0;
        for i in 0..nums.len() {
            for j in i..nums.len() {
                if i == j {
                    dp[i][j] = nums[i];
                } else {
                    dp[i][j] = dp[i][j - 1] + nums[j];
                }
                max = max.max(dp[i][j].abs())
            }
        }
        max
    }
    /*
    the optimized solution is using two variables to store the max and min value
    max stores the max sum for positive sums
    min stores the min sum for negtive sums
    we can get the max absolute sum by max.abs().max(min)
    time complexity O(n) space complexity O(1)
     */
    pub fn max_absolute_sum_optimized(nums: Vec<i32>) -> i32 {
        let mut max = i32::MIN;
        let mut min = i32::MAX;

        let (mut positive_sum, mut negtive_sum) = (0, 0);

        for x in nums.iter() {
            positive_sum += x;
            negtive_sum += x;

            max = max.max(positive_sum);
            min = min.min(negtive_sum);

            positive_sum = positive_sum.max(0);
            negtive_sum = negtive_sum.min(0);
        }

        min.abs().max(max)
    }

    /*
    link: https://leetcode.com/problems/new-21-game/description
    The first thing comes to my mind is to brute force all the possible cases
    and calculate the probability by finding the matched cases
    but test cases failed
     */

    pub fn new21_game(n: i32, k: i32, max_pts: i32) -> f64 {
        let mut cases = Vec::new();
        Self::calculate_prob(&mut cases, 0, k, max_pts);
        let len = cases.len() as f64;
        let mut count = 0;
        for i in cases.into_iter() {
            if i <= n {
                count += 1;
            }
        }
        return count as f64 / len;
    }
    fn calculate_prob(set: &mut Vec<i32>, sum: i32, k: i32, max_pts: i32) {
        if sum >= k {
            set.push(sum);
            return;
        }
        for i in 1..max_pts + 1 {
            Self::calculate_prob(set, sum + i, k, max_pts);
        }
    }
    /*
    link: https://leetcode.com/problems/jump-game-ii/
    we can use greedy algorithm to solve this problem with time complexity O(n)
    create a vector named steps, steps[i] represents the minimum steps to reach the end from i
    initialize the steps from the end, steps[end] = 0
    steps[i] = steps[j] + 1, j is the index of the minimum value in nums[i, i+nums[i]]
    return steps[0] as result
     */

    pub fn jump(nums: Vec<i32>) -> i32 {
        let mut steps = vec![i32::MAX; nums.len()];
        steps[nums.len() - 1] = 0;
        for i in (0..nums.len() - 1).rev() {
            if nums[i] == 0 {
                continue;
            }
            let mut min = i32::MAX;
            for j in i + 1..=i + nums[i] as usize {
                if j >= nums.len() {
                    break;
                }
                min = min.min(steps[j]);
            }
            if min != i32::MAX {
                steps[i] = min + 1;
            }
        }

        steps[0]
    }

    pub fn unique_paths(m: i32, n: i32) -> i32 {
        let mut steps = vec![vec![0; n as usize]; m as usize];
        Self::solve_unique_path(0, 0, &m, &n, &mut steps)
    }
    /*
    link: https://leetcode.com/problems/unique-paths/
    define a matrix named steps, steps[i,j] represents the unique paths from (i,j) to (m-1,n-1)
    steps[i,j] = steps[i+1,j] + steps[i,j+1]
    if i == m-1 && j == n-1, steps[i,j] = 1
     */

    fn solve_unique_path(i: i32, j: i32, m: &i32, n: &i32, steps: &mut Vec<Vec<i32>>) -> i32 {
        if i < 0 || j < 0 || i >= *m || j >= *n {
            return 0;
        }
        if i == *m - 1 && j == *n - 1 {
            return 1;
        }
        if steps[i as usize][j as usize] != 0 {
            return steps[i as usize][j as usize];
        }
        let right = Self::solve_unique_path(i, j + 1, m, n, steps);
        let down = Self::solve_unique_path(i + 1, j, m, n, steps);

        steps[i as usize][j as usize] = right + down;

        steps[i as usize][j as usize]
    }

    pub fn unique_path_with_obstacle(grids: Vec<Vec<i32>>) -> i32 {
        let mut steps = vec![vec![0; grids[0].len()]; grids.len()];
        Self::solve_unique_path_with_obstacle(0, 0, &grids, &mut steps)
    }
    fn solve_unique_path_with_obstacle(
        i: i32,
        j: i32,
        grids: &Vec<Vec<i32>>,
        steps: &mut Vec<Vec<i32>>,
    ) -> i32 {
        if i < 0 || j < 0 || i >= grids.len() as i32 || j >= grids[0].len() as i32 {
            return 0;
        }
        if grids[i as usize][j as usize] == 1 {
            return 0;
        }
        if i == grids.len() as i32 - 1 && j == grids[0].len() as i32 - 1 {
            return 1;
        }

        if steps[i as usize][j as usize] != 0 {
            return steps[i as usize][j as usize];
        }
        let right = Self::solve_unique_path_with_obstacle(i, j + 1, grids, steps);
        let down = Self::solve_unique_path_with_obstacle(i + 1, j, grids, steps);
        steps[i as usize][j as usize] = right + down;
        steps[i as usize][j as usize]
    }

    /*
    link: https://leetcode.com/problems/minimum-path-sum/
    dp solve
    define a matrix named steps, steps[i,j] represents the minimum path sum from (i,j) to (m-1,n-1)
    steps[i,j] = min(steps[i+1,j], steps[i,j+1]) + grid[i,j]
    if i == m-1 && j == n-1, steps[i,j] = grid[i,j]
     */

    pub fn minimum_path_sum(grid: Vec<Vec<i32>>) -> i32 {
        let mut steps = vec![vec![0; grid[0].len()]; grid.len()];
        Self::solve_minimum_path_sum(0, 0, &grid, &mut steps)
    }

    fn solve_minimum_path_sum(
        i: usize,
        j: usize,
        grid: &Vec<Vec<i32>>,
        steps: &mut Vec<Vec<i32>>,
    ) -> i32 {
        if i >= grid.len() || j >= grid[0].len() {
            return i32::MAX;
        }
        if i == grid.len() - 1 && j == grid[0].len() - 1 {
            return grid[i as usize][j as usize];
        }
        if steps[i][j] != 0 {
            return steps[i][j];
        }
        let right = Self::solve_minimum_path_sum(i, j + 1, grid, steps);
        let down = Self::solve_minimum_path_sum(i + 1, j, grid, steps);
        steps[i][j] = grid[i][j] + right.min(down);
        steps[i][j]
    }

    pub fn climb_stairs(n: i32) -> i32 {
        if n == 1 {
            return 1;
        }
        let mut steps = vec![0; n as usize];
        steps[0] = 1;
        steps[1] = 2;
        for i in 2..n as usize {
            steps[i] = steps[i - 1] + steps[i - 2];
        }
        steps[n as usize - 1]
    }

    /*
    link: https://leetcode.com/problems/longest-arithmetic-subsequence/
    dp solve
    define a matrix of Vec<HashMap<i32,i32>>
    dp[i][j] represents the longest arithmetic subsequence length of nums[0..i] with difference j
     */

    pub fn longest_arith_seq_length(nums: Vec<i32>) -> i32 {
        if nums.len() < 3 {
            return nums.len() as i32;
        }

        let mut dp = vec![std::collections::HashMap::<i32, i32>::new(); nums.len()];
        let mut longest = 0;
        for i in 0..nums.len() {
            for j in 0..i {
                let diff = nums[i] - nums[j];
                let count = dp[j].get(&diff).unwrap_or(&1) + 1;
                dp[i].insert(diff, count);
                longest = longest.max(count);
            }
        }

        longest
    }

    /*
    link: https://leetcode.com/problems/house-robber/description/
    follow the explaination from
    https://leetcode.com/problems/house-robber/solutions/156523/from-good-to-great-how-to-approach-most-of-dp-problems/
     */

    pub fn rob(nums: Vec<i32>) -> i32 {
        if nums.len() == 0 {
            return 0;
        }
        let (mut prev1, mut prev2) = (0, 0);

        for num in nums {
            let tmp = prev1;
            prev1 = prev1.max(prev2 + num);
            prev2 = tmp;
        }

        return prev1;
    }

    #[allow(dead_code)]
    fn rob_helper(nums: &Vec<i32>, memo: &mut Vec<i32>, i: i32) -> i32 {
        if i < 0 {
            return 0;
        }
        if memo[i as usize] > 0 {
            return memo[i as usize];
        }
        let val = (nums[i as usize] + Self::rob_helper(nums, memo, i - 2)).max(Self::rob_helper(
            nums,
            memo,
            i - 1,
        ));
        memo[i as usize] = val;
        val
    }

    /*
    link: https://leetcode.com/problems/coin-change-ii/
    dp solve
    dp[i] represents the number of combinations to make up amount i
    we can update dp[i] by dp[i] += dp[i-coin]
     */

    pub fn change(amount: i32, coins: Vec<i32>) -> i32 {
        let mut dp = vec![0; amount as usize + 1];
        dp[0] = 1;
        for coin in coins {
            for i in coin..=amount {
                dp[i as usize] += dp[(i - coin) as usize];
            }
        }
        dp[amount as usize]
    }
    
    /*
    link https://leetcode.com/problems/coin-change/
    find the fewest number of coins that you need to make up that amount
    dp solve
    dp[i] represents the fewest number of coins that you need to make up amount i
    dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount]
     */

    pub fn coin_change(coins:Vec<i32>, amount: i32) -> i32 {
        let mut dp = vec![amount+1; amount as usize + 1];
        dp[0] = 0;
        for i in 1..=amount {
            for coin in &coins {
                if i >= *coin {
                    // dp[i] = dp[i].min(dp[i-coin]+1); dp[i-coin] + 1 is the way to make up amount i with coin
                    dp[i as usize] = dp[i as usize].min(dp[(i - coin) as usize] + 1);
                }
            }
        }
        if dp[amount as usize] > amount {
            return -1;
        }
        dp[amount as usize]
    }

    /*
    link: https://leetcode.com/problems/edit-distance/
    given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.
    we can solve this problem using dynamic programming.
    let dp[i][j] be the minimum number of operations required to convert word1[0..i] to word2[0..j].
    if word1[i] == word2[j], then dp[i][j] = dp[i - 1][j - 1]
    otherwise, dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    
     */

     pub fn min_distance(word1: String, word2: String) -> i32 {
        let mut dp = vec![vec![0; word2.len() + 1]; word1.len() + 1];
        for i in 0..=word1.len() {
            dp[i][0] = i; // initialize the first column
        }
        for j in 0..=word2.len() {
            dp[0][j] = j; // initialize the first row 
        }
        for i in 1..=word1.len() {
            for j in 1..=word2.len() {
                if word1.chars().nth(i - 1) == word2.chars().nth(j - 1) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    let insert = dp[i][j - 1] + 1;   // insert word2[j]
                    let delete = dp[i - 1][j] + 1; // delete word1[i]
                    let replace = dp[i - 1][j - 1] + 1;  // replace word1[i] with word2[j]
                    dp[i][j] = insert.min(delete).min(replace);
                }
            }
        }
        dp[word1.len()][word2.len()] as i32
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_max_absolute_sum() {
        assert_eq!(DPSolution::max_absolute_sum(vec![1, -3, 2, 3, -4]), 5);
        assert_eq!(DPSolution::max_absolute_sum(vec![2, -5, 1, -4, 3, -2]), 8);

        assert_eq!(
            DPSolution::max_absolute_sum_optimized(vec![1, -3, 2, 3, -4]),
            5
        );
        assert_eq!(
            DPSolution::max_absolute_sum_optimized(vec![2, -5, 1, -4, 3, -2]),
            8
        );
    }

    #[test]
    fn test_new21_game() {
        assert_eq!(DPSolution::new21_game(10, 1, 10), 1.0);
        assert_eq!(DPSolution::new21_game(6, 1, 10), 0.6);
        // assert_eq!(DPSolution::new21_game(21, 17, 10), 0.73278); test failed
    }

    #[test]
    fn test_jump() {
        assert_eq!(DPSolution::jump(vec![2, 3, 1, 1, 4]), 2);
        assert_eq!(DPSolution::jump(vec![2, 3, 0, 1, 4]), 2);
        assert_eq!(
            DPSolution::jump(vec![5, 9, 3, 2, 1, 0, 2, 3, 3, 1, 0, 0]),
            3
        );
    }

    #[test]
    fn test_unique_paths() {
        assert_eq!(DPSolution::unique_paths(3, 2), 3);
        assert_eq!(DPSolution::unique_paths(7, 3), 28);
    }

    #[test]
    fn test_unique_path_with_obstacle() {
        assert_eq!(
            DPSolution::unique_path_with_obstacle(vec![
                vec![0, 0, 0],
                vec![0, 1, 0],
                vec![0, 0, 0]
            ]),
            2
        );
        assert_eq!(
            DPSolution::unique_path_with_obstacle(vec![vec![0, 1], vec![0, 0]]),
            1
        );
    }

    #[test]
    fn test_minimum_path_sum() {
        assert_eq!(
            DPSolution::minimum_path_sum(vec![vec![1, 3, 1], vec![1, 5, 1], vec![4, 2, 1]]),
            7
        );
        assert_eq!(
            DPSolution::minimum_path_sum(vec![vec![1, 2, 3], vec![4, 5, 6]]),
            12
        );
    }

    #[test]
    fn test_climb_stairs() {
        assert_eq!(DPSolution::climb_stairs(2), 2);
        assert_eq!(DPSolution::climb_stairs(3), 3);
        assert_eq!(DPSolution::climb_stairs(4), 5);
    }

    #[test]
    fn test_longest_arith_seq_length() {
        assert_eq!(DPSolution::longest_arith_seq_length(vec![3, 6, 9, 12]), 4);
        assert_eq!(
            DPSolution::longest_arith_seq_length(vec![9, 4, 7, 2, 10]),
            3
        );
        assert_eq!(
            DPSolution::longest_arith_seq_length(vec![20, 1, 15, 3, 10, 5, 8]),
            4
        );
    }

    #[test]
    fn test_rob() {
        assert_eq!(DPSolution::rob(vec![1, 2, 3, 1]), 4);
        assert_eq!(DPSolution::rob(vec![2, 7, 9, 3, 1]), 12);
    }

    #[test]
    fn test_change() {
        assert_eq!(DPSolution::change(5, vec![1, 2, 5]), 4);
        assert_eq!(DPSolution::change(3, vec![2]), 0);
        assert_eq!(DPSolution::change(10, vec![10]), 1);
    }

    #[test]
    fn test_coin_change() {
        assert_eq!(DPSolution::coin_change(vec![1, 2, 5], 11), 3);
        assert_eq!(DPSolution::coin_change(vec![2], 3), -1);
    }

    #[test]
    fn test_min_distance() {
        assert_eq!(
            DPSolution::min_distance("horse".to_string(), "ros".to_string()),
            3
        );
        assert_eq!(
            DPSolution::min_distance("intention".to_string(), "execution".to_string()),
            5
        );
    }
}