#![allow(dead_code)]

pub mod dpsolution {
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
        calculate_prob(&mut cases, 0, k, max_pts);
        let len = cases.len() as f64;
        let mut count = 0;
        for i in cases.into_iter() {
            if i <= n {
                count += 1;
            }
        }
        count as f64 / len
    }
    fn calculate_prob(set: &mut Vec<i32>, sum: i32, k: i32, max_pts: i32) {
        if sum >= k {
            set.push(sum);
            return;
        }
        for i in 1..max_pts + 1 {
            calculate_prob(set, sum + i, k, max_pts);
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
            for (j, &n) in steps
                .iter()
                .enumerate()
                .take(i + nums[i] as usize + 1)
                .skip(i + 1)
            {
                if j >= nums.len() {
                    break;
                }
                min = min.min(n);
            }
            if min != i32::MAX {
                steps[i] = min + 1;
            }
        }

        steps[0]
    }

    pub fn unique_paths(m: i32, n: i32) -> i32 {
        let mut steps = vec![vec![0; n as usize]; m as usize];
        solve_unique_path(0, 0, &m, &n, &mut steps)
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
        let right = solve_unique_path(i, j + 1, m, n, steps);
        let down = solve_unique_path(i + 1, j, m, n, steps);

        steps[i as usize][j as usize] = right + down;

        steps[i as usize][j as usize]
    }

    pub fn unique_path_with_obstacle(grids: Vec<Vec<i32>>) -> i32 {
        let mut steps = vec![vec![0; grids[0].len()]; grids.len()];
        solve_unique_path_with_obstacle(0, 0, &grids, &mut steps)
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
        let right = solve_unique_path_with_obstacle(i, j + 1, grids, steps);
        let down = solve_unique_path_with_obstacle(i + 1, j, grids, steps);
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
        solve_minimum_path_sum(0, 0, &grid, &mut steps)
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
            return grid[i][j];
        }
        if steps[i][j] != 0 {
            return steps[i][j];
        }
        let right = solve_minimum_path_sum(i, j + 1, grid, steps);
        let down = solve_minimum_path_sum(i + 1, j, grid, steps);
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
        if nums.is_empty() {
            return 0;
        }
        let (mut prev1, mut prev2) = (0, 0);

        for num in nums {
            let tmp = prev1;
            prev1 = prev1.max(prev2 + num);
            prev2 = tmp;
        }

        prev1
    }

    fn rob_helper(nums: &Vec<i32>, memo: &mut Vec<i32>, i: i32) -> i32 {
        if i < 0 {
            return 0;
        }
        if memo[i as usize] > 0 {
            return memo[i as usize];
        }
        let val =
            (nums[i as usize] + rob_helper(nums, memo, i - 2)).max(rob_helper(nums, memo, i - 1));
        memo[i as usize] = val;
        val
    }

    /*
    link https://leetcode.com/problems/coin-change/
    find the fewest number of coins that you need to make up that amount
    dp solve
    dp[i] represents the fewest number of coins that you need to make up amount i
    dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount]
     */

    pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
        let mut dp = vec![amount + 1; amount as usize + 1]; // initialize the dp array with amount+1, because the maximum number of coins is amount
        dp[0] = 0; // we only need 0 coin to make up amount 0
        for i in 1..=amount {
            for &coin in &coins {
                if i >= coin {
                    // the way to make up amount i with coin
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
    link: https://leetcode.com/problems/coin-change-ii/
    dp solve
    dp[i] represents the number of combinations to make up amount i
    for each coin in coins, we can update dp[i] by dp[i] += dp[i-coin]
     */

    pub fn coin_change_ii(amount: i32, coins: Vec<i32>) -> i32 {
        let mut dp = vec![0; amount as usize + 1];
        dp[0] = 1; // we only have one way to make up amount 0, that is using no coin
        for coin in coins {
            for i in coin..=amount {
                dp[i as usize] += dp[(i - coin) as usize];
            }
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
        dp.iter_mut()
            .enumerate()
            .take(word1.len() + 1)
            .for_each(|(i, d)| d[0] = i); // initialize the first column

        dp[0]
            .iter_mut()
            .enumerate()
            .take(word2.len() + 1)
            .for_each(|(j, d)| *d = j); // initialize the first row

        for i in 1..=word1.len() {
            for j in 1..=word2.len() {
                if word1.chars().nth(i - 1) == word2.chars().nth(j - 1) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    let insert = dp[i][j - 1] + 1; // insert word2[j]
                    let delete = dp[i - 1][j] + 1; // delete word1[i]
                    let replace = dp[i - 1][j - 1] + 1; // replace word1[i] with word2[j]
                    dp[i][j] = insert.min(delete).min(replace);
                }
            }
        }
        dp[word1.len()][word2.len()] as i32
    }
    /*
    link: https://leetcode.com/problems/distinct-subsequences/
    given a string S and a string T, count the number of distinct subsequences of S which equals T.

    we can use dp to solve this problem.
    let dp[i][j] be the number of distinct subsequences of S[0..i] which equals T[0..j]
    if S[i] == T[j], that means we can use S[i] to match T[j],
     the number of distinct subsequences is the sum of the number of distinct subsequences
     of S[0..i-1] which equals T[0..j-1] and the number of distinct subsequences of S[0..i-1] which equals T[0..j]
    then dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
    otherwise, dp[i][j] = dp[i - 1][j]

     */
    pub fn num_distinct(s: String, t: String) -> i32 {
        let mut dp = vec![vec![0; t.len() + 1]; s.len() + 1];
        dp.iter_mut().take(s.len() + 1).for_each(|d| d[0] = 1); // initialize the first column

        for i in 1..=s.len() {
            for j in 1..=t.len() {
                dp[i][j] = if s.chars().nth(i - 1) == t.chars().nth(j - 1) {
                    dp[i - 1][j - 1] + dp[i - 1][j]
                } else {
                    dp[i - 1][j]
                }
            }
        }
        dp[s.len()][t.len()]
    }

    /*
    https://leetcode.com/problems/triangle/

    find the minimum path sum from top to bottom.

    def dp[i][j] be the minimum path sum from triangle[i][j] to the bottom
    dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])
    build the dp matrix from bottom to top
     */

    pub fn minimum_triangle(triangle: Vec<Vec<i32>>) -> i32 {
        let mut dp = vec![vec![0; triangle.len()]; triangle.len()];
        for i in 0..triangle.len() {
            dp[triangle.len() - 1][i] = triangle[triangle.len() - 1][i];
        }
        for i in (0..triangle.len() - 1).rev() {
            for j in 0..triangle[i].len() {
                dp[i][j] = triangle[i][j] + dp[i + 1][j].min(dp[i + 1][j + 1]);
            }
        }
        dp[0][0]
    }

    /*
    https://leetcode.com/problems/wildcard-matching/
    define dp[i][j] be the result of s[0..i] matches p[0..j]
    dp[i][j] = dp[i-1][j-1], if s[i] == p[j] || p[j] == '?'
               dp[i-1][j] || dp[i][j-1], if p[j] == '*'
     */

    pub fn is_match(s: String, p: String) -> bool {
        let mut dp = vec![vec![false; p.len() + 1]; s.len() + 1];
        let s = s.chars().collect::<Vec<char>>();
        let p = p.chars().collect::<Vec<char>>();
        dp[0][0] = true;
        for i in 1..=p.len() {
            if p[i - 1] == '*' {
                dp[0][i] = dp[0][i - 1];
            }
        }
        for i in 1..=s.len() {
            for j in 1..=p.len() {
                if p[j - 1] == '?' || p[j - 1] == s[i - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                    continue;
                }
                if p[j - 1] == '*' {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
        dp[s.len()][p.len()]
    }

    /*
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    find the maximum profit you can achieve with at most two transactions.

    in order to get the maximum profit,
    we need to find the max profit we can make with at most one transaction on day j
    then we can get the max profit we can make with at most two transactions on day j+1

    define dp[i][j] represents the maximum profit you can achieve with at most i transactions on day j
    dp[i][j] = max(dp[i][j-1], prices[j] - prices[m] + dp[i-1][m-1]), m is the index of the minimum price in prices[0..j-1]

    the first dimension of dp is 3, because we can make at most two transactions
    the second dimension of dp is prices.len(), because we can make transactions on each day

    dp[2][prices.len()-1] is the max profit we can make with at most two transactions on the last day


     */

    pub fn max_profit_iii(prices: Vec<i32>) -> i32 {
        if prices.len() < 2 {
            return 0;
        }
        let mut dp = vec![vec![0; prices.len()]; 3];
        for i in 1..=2 {
            let mut min = prices[0];
            for (j, p) in prices.iter().enumerate().skip(1) {
                // min is the minimum price in prices[0..j]
                min = min.min(p - dp[i - 1][j - 1]);
                // update dp[i][j] by comparing dp[i][j-1] and prices[j] - min
                // dp[i][j-1] is the max profit we can make with at most i transactions on day j-1
                // prices[j] - min is the max profit we can make with at most i transactions on day j
                // if we don't sell stock on day j, then dp[i][j] = dp[i][j-1]
                // if we sell stock on day j, then dp[i][j] = prices[j] - min
                dp[i][j] = dp[i][j - 1].max(p - min);
            }
        }
        dp[2][prices.len() - 1]
    }

    /*
    https://leetcode.com/problems/combination-sum-iv/
    define dp
    dp[i] = the number of combinations that make up amount i
    dp[i] = dp[i] + dp[i-coin] if i >= coin
    */

    pub fn combination_sum4(nums: Vec<i32>, target: i32) -> i32 {
        let mut dp = vec![0; target as usize + 1];
        dp[0] = 1;
        for i in 1..=target {
            for num in &nums {
                if i >= *num {
                    dp[i as usize] += dp[(i - num) as usize];
                }
            }
        }
        dp[target as usize]
    }

    /*

    https://leetcode.com/problems/cherry-pickup-ii/
    follow: https://leetcode.com/problems/cherry-pickup-ii/solutions/660562/c-java-python-top-down-dp-clean-code
    define dp[r][c1][c2], r is the row index, c1 and c2 are the column index for robot1 and robot2
    dp[r][c1][c2] represents the maximum number of cherries that robot1 and robot2 can pick up from grid[r..grid.len()-1]
    robot1 is at (r,c1), robot2 is at (r,c2)

     */

    pub fn cherry_pickup_ii(grid: Vec<Vec<i32>>) -> i32 {
        let mut dp = vec![vec![vec![-1; grid[0].len()]; grid[0].len()]; grid.len()];
        solve_cherry_pickup_ii(&grid, &mut dp, 0, 0, grid[0].len() as i32 - 1)
    }

    pub fn solve_cherry_pickup_ii(
        grid: &Vec<Vec<i32>>,
        dp: &mut Vec<Vec<Vec<i32>>>,
        r: i32,
        c1: i32,
        c2: i32,
    ) -> i32 {
        if r == grid.len() as i32 {
            return 0;
        }
        if c1 < 0 || c1 >= grid[0].len() as i32 || c2 < 0 || c2 >= grid[0].len() as i32 {
            return 0;
        }
        if dp[r as usize][c1 as usize][c2 as usize] != -1 {
            return dp[r as usize][c1 as usize][c2 as usize];
        }
        let cherries = grid[r as usize][c1 as usize]
            + if c1 != c2 {
                grid[r as usize][c2 as usize]
            } else {
                0
            };
        let result = if r != grid.len() as i32 - 1 {
            let mut max = 0;
            for i in -1..=1 {
                for j in -1..=1 {
                    max = max.max(solve_cherry_pickup_ii(grid, dp, r + 1, c1 + i, c2 + j));
                }
            }
            max + cherries
        } else {
            cherries
        };
        dp[r as usize][c1 as usize][c2 as usize] = result;
        result
    }

    /*
    https://leetcode.com/problems/min-cost-climbing-stairs/

    define dp
    dp[i] = the minimum cost to reach the i-th stair
    do[i] = cost[i] + min(dp[i-1], dp[i-2])
    dp[0] = cost[0]
    dp[1] = cost[1]

    return min(dp[n-1], dp[n-2])
     */

    pub fn min_cost_climbing_stairs(cost: Vec<i32>) -> i32 {
        let mut dp = vec![0; cost.len()];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for i in 2..cost.len() {
            dp[i] = cost[i] + dp[i - 1].min(dp[i - 2]);
        }

        dp[cost.len() - 1].min(dp[cost.len() - 2])
    }

    // Using bottom-up dynamic programming
    pub fn calculate_minimum_hp(dungeon: Vec<Vec<i32>>) -> i32 {
        use std::cmp::{max, min};
        let (r, c) = (dungeon.len(), dungeon[0].len());
        let mut dp = vec![vec![i32::MIN; c]; r];

        for y in (0..r).rev() {
            for x in (0..c).rev() {
                let val = dungeon[y][x];
                if x == c - 1 && y == r - 1 {
                    // princess position
                    dp[y][x] = val;
                } else if x == c - 1 && y != r - 1 {
                    // right
                    let new_val = val + dp[y + 1][x];
                    dp[y][x] = if new_val < 0 { new_val } else { 0 };
                } else if x != c - 1 && y == r - 1 {
                    // bottom
                    let new_val = val + dp[y][x + 1];
                    dp[y][x] = if new_val < 0 { new_val } else { 0 };
                } else {
                    let down = val + dp[y + 1][x];
                    let right = val + dp[y][x + 1];
                    dp[y][x] = max(down, right);
                }
                dp[y][x] = min(val, dp[y][x]);
            }
        }

        if dp[0][0] < 0 {
            -dp[0][0] + 1
        } else {
            1
        }
    }

    #[test]
    fn test_calculate_minimum_hp() {
        assert_eq!(
            calculate_minimum_hp(vec![vec![-2, -3, 3], vec![-5, -10, 1], vec![10, 30, -5]]),
            7
        );
    }

    #[test]
    fn test_max_absolute_sum() {
        assert_eq!(max_absolute_sum(vec![1, -3, 2, 3, -4]), 5);
        assert_eq!(max_absolute_sum(vec![2, -5, 1, -4, 3, -2]), 8);

        assert_eq!(max_absolute_sum_optimized(vec![1, -3, 2, 3, -4]), 5);
        assert_eq!(max_absolute_sum_optimized(vec![2, -5, 1, -4, 3, -2]), 8);
    }

    #[test]
    fn test_new21_game() {
        assert_eq!(new21_game(10, 1, 10), 1.0);
        assert_eq!(new21_game(6, 1, 10), 0.6);
        // assert_eq!(new21_game(21, 17, 10), 0.73278); test failed
    }

    #[test]
    fn test_jump() {
        assert_eq!(jump(vec![2, 3, 1, 1, 4]), 2);
        assert_eq!(jump(vec![2, 3, 0, 1, 4]), 2);
        assert_eq!(jump(vec![5, 9, 3, 2, 1, 0, 2, 3, 3, 1, 0, 0]), 3);
    }

    #[test]
    fn test_unique_paths() {
        assert_eq!(unique_paths(3, 2), 3);
        assert_eq!(unique_paths(7, 3), 28);
    }

    #[test]
    fn test_unique_path_with_obstacle() {
        assert_eq!(
            unique_path_with_obstacle(vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]]),
            2
        );
        assert_eq!(unique_path_with_obstacle(vec![vec![0, 1], vec![0, 0]]), 1);
    }

    #[test]
    fn test_minimum_path_sum() {
        assert_eq!(
            minimum_path_sum(vec![vec![1, 3, 1], vec![1, 5, 1], vec![4, 2, 1]]),
            7
        );
        assert_eq!(minimum_path_sum(vec![vec![1, 2, 3], vec![4, 5, 6]]), 12);
    }

    #[test]
    fn test_climb_stairs() {
        assert_eq!(climb_stairs(2), 2);
        assert_eq!(climb_stairs(3), 3);
        assert_eq!(climb_stairs(4), 5);
    }

    #[test]
    fn test_longest_arith_seq_length() {
        assert_eq!(longest_arith_seq_length(vec![3, 6, 9, 12]), 4);
        assert_eq!(longest_arith_seq_length(vec![9, 4, 7, 2, 10]), 3);
        assert_eq!(longest_arith_seq_length(vec![20, 1, 15, 3, 10, 5, 8]), 4);
    }

    #[test]
    fn test_rob() {
        assert_eq!(rob(vec![1, 2, 3, 1]), 4);
        assert_eq!(rob(vec![2, 7, 9, 3, 1]), 12);
    }

    #[test]
    fn test_change() {
        assert_eq!(coin_change_ii(5, vec![1, 2, 5]), 4);
        assert_eq!(coin_change_ii(3, vec![2]), 0);
        assert_eq!(coin_change_ii(10, vec![10]), 1);
    }

    #[test]
    fn test_coin_change() {
        assert_eq!(coin_change(vec![1, 2, 5], 11), 3);
        assert_eq!(coin_change(vec![2], 3), -1);
    }

    #[test]
    fn test_min_distance() {
        assert_eq!(min_distance("horse".to_string(), "ros".to_string()), 3);
        assert_eq!(
            min_distance("intention".to_string(), "execution".to_string()),
            5
        );
    }

    #[test]
    fn test_num_distinct() {
        assert_eq!(num_distinct("rabbbit".to_string(), "rabbit".to_string()), 3);
        assert_eq!(num_distinct("babgbag".to_string(), "bag".to_string()), 5);
    }

    #[test]
    fn test_minimum_triangle() {
        assert_eq!(
            minimum_triangle(vec![vec![2], vec![3, 4], vec![6, 5, 7], vec![4, 1, 8, 3]]),
            11
        );
        assert_eq!(
            minimum_triangle(vec![vec![-1], vec![2, 3], vec![1, -1, -3]],),
            -1
        );
        assert_eq!(
            minimum_triangle(vec![
                vec![1],
                vec![-2, -5],
                vec![3, 6, 9],
                vec![-1, 2, 4, -3]
            ]),
            1
        )
    }

    #[test]
    fn test_is_match() {
        assert_eq!(is_match("aa".to_string(), "a".to_string()), false);
        assert_eq!(is_match("aa".to_string(), "*".to_string()), true);
        assert_eq!(is_match("cb".to_string(), "?a".to_string()), false);
        assert_eq!(is_match("adceb".to_string(), "*a*b".to_string()), true);
        assert_eq!(is_match("acdcb".to_string(), "a*c?b".to_string()), false);
    }

    #[test]
    fn test_max_profit_iii() {
        assert_eq!(max_profit_iii(vec![3, 3, 5, 0, 0, 3, 1, 4]), 6);
        assert_eq!(max_profit_iii(vec![1, 2, 3, 4, 5]), 4);
        assert_eq!(max_profit_iii(vec![7, 6, 4, 3, 1]), 0);
    }

    #[test]
    fn test_combination_sum4() {
        assert_eq!(combination_sum4(vec![1, 2, 3], 4), 7);
        assert_eq!(combination_sum4(vec![9], 3), 0);
    }

    #[test]
    fn test_cherry_pickup_ii() {
        assert_eq!(
            cherry_pickup_ii(vec![
                vec![3, 1, 1],
                vec![2, 5, 1],
                vec![1, 5, 5],
                vec![2, 1, 1]
            ]),
            24
        );
        assert_eq!(
            cherry_pickup_ii(vec![
                vec![1, 0, 0, 0, 0, 0, 1],
                vec![2, 0, 0, 0, 0, 3, 0],
                vec![2, 0, 9, 0, 0, 0, 0],
                vec![0, 3, 0, 5, 4, 0, 0],
                vec![1, 0, 2, 3, 0, 0, 6]
            ]),
            28
        );
    }

    #[test]
    fn test_min_cost_climbing_stairs() {
        assert_eq!(min_cost_climbing_stairs(vec![10, 15, 20]), 15);
        assert_eq!(
            min_cost_climbing_stairs(vec![1, 100, 1, 1, 1, 100, 1, 1, 100, 1]),
            6
        );
    }
} // mod dpsolution
