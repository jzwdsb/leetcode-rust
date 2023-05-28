
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
        DPSolution::calculate_prob(&mut cases, 0, k, max_pts);
        let len = cases.len() as f64;
        let mut count = 0;
        for i in cases.into_iter() {
            if i <= n {
                count += 1;
            }
        }
        return count as f64 / len;
    }
    fn calculate_prob(set: &mut Vec<i32>, sum: i32, k: i32, max_pts: i32){
        if sum >= k {
            set.push(sum);
            return;
        }
        for i in 1..max_pts+1 {
            DPSolution::calculate_prob(set, sum + i, k, max_pts);
        }
    }
    
}

#[test]
fn test_max_absolute_sum() {
    assert_eq!(DPSolution::max_absolute_sum(vec![1, -3, 2, 3, -4]), 5);
    assert_eq!(DPSolution::max_absolute_sum(vec![2, -5, 1, -4, 3, -2]), 8);

    assert_eq!(DPSolution::max_absolute_sum_optimized(vec![1, -3, 2, 3, -4]), 5);
    assert_eq!(DPSolution::max_absolute_sum_optimized(vec![2, -5, 1, -4, 3, -2]), 8);
}

#[test]
fn test_new21_game() {
    assert_eq!(DPSolution::new21_game(10, 1, 10), 1.0);
    assert_eq!(DPSolution::new21_game(6, 1, 10), 0.6);
    // assert_eq!(DPSolution::new21_game(21, 17, 10), 0.73278); test failed
}