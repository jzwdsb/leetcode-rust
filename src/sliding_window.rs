#![allow(dead_code)]

use std::cmp::Ordering;

pub struct SlidingWindow {}

impl SlidingWindow {
    pub fn max_vowels(s: String, k: i32) -> i32 {
        // this could do the work, but time limit exceeded
        // s.as_bytes().windows(k as usize).map(|x| {
        //     x.iter().filter(|&c| {
        //         VOEWLS.contains(&(*c as char))
        //     }).count()
        // }).max().unwrap_or(0) as i32
        let s = s.as_bytes();
        let k = k as usize;
        let count: usize = s.iter().take(k).filter(|&x| Self::is_vowel(*x)).count();
        s.iter()
            .skip(k)
            .zip(s.iter())
            .fold((count, count), |a, x| {
                let (max, count) = a;
                let (cur, left) = x;
                match (Self::is_vowel(*left), Self::is_vowel(*cur)) {
                    (false, true) => (max.max(count + 1), count + 1),
                    (true, false) => (max, count - 1),
                    _ => (max, count),
                }
            })
            .0 as i32
    }

    fn is_vowel(c: u8) -> bool {
        matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
    }

    /*
    link: https://leetcode.com/problems/4sum/
    four sum problem

     */

    pub fn four_sum(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut res = Vec::new();
        let n = nums.len();
        if n < 4 {
            return res;
        }

        let mut nums = nums;
        nums.sort_unstable();
        for i in 0..n - 3 {
            if nums[i] as i64 * 4 > target as i64 {
                break;
            }
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }

            for j in i + 1..n - 2 {
                if nums[j] as i64 * 3 > (target - nums[i]) as i64 {
                    continue;
                }
                if j > i + 1 && nums[j] == nums[j - 1] {
                    continue;
                }

                let (mut left, mut right) = (j + 1, n - 1);
                while left < right {
                    let sum =
                        nums[i] as i64 + nums[j] as i64 + nums[left] as i64 + nums[right] as i64;
                    match sum.cmp(&(target as i64)) {
                        Ordering::Equal => {
                            res.push(vec![nums[i], nums[j], nums[left], nums[right]]);
                            while left < right && nums[left] == nums[left + 1] {
                                left += 1;
                            }
                            while left < right && nums[right] == nums[right - 1] {
                                right -= 1;
                            }
                            left += 1;
                            right -= 1;
                        }
                        Ordering::Greater => right -= 1,
                        Ordering::Less => left += 1,
                    }
                }
            }
        }

        res
    }

    /*
    link: https://leetcode.com/problems/maximize-the-confusion-of-an-exam/
     */

    pub fn max_consecutive_answers(answer_key: String, k: i32) -> i32 {
        Self::consecutive_helper(&answer_key, k, 'T').max(Self::consecutive_helper(
            &answer_key,
            k,
            'F',
        ))
    }

    // find the max consecutive answers with a given char
    // k is the number of answer that can be changed
    // c is the char that we can change
    // return the max consecutive answers
    fn consecutive_helper(answer_key: &str, k: i32, c: char) -> i32 {
        let mut res = 0;
        let (mut left, mut right) = (0, 0); // window size = right - left
        let mut count = 0; // the number of charachter in the window that can be changed
        let k = k as usize;
        let s: Vec<char> = answer_key.chars().collect();

        while right < s.len() {
            if s[right] == c {
                count += 1;
            }
            // if count > k, we need to move the left pointer
            while count > k {
                // if the left pointer is the char that we can change, we need to decrease the count
                if s[left] == c {
                    count -= 1;
                }
                left += 1;
            }
            res = res.max(right - left + 1);
            right += 1;
        }
        res as i32
    }

    pub fn get_averages(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let k = k as usize;
        let mut res = vec![-1; nums.len()];
        let len = k * 2 + 1;
        if len > res.len() {
            return res;
        }

        let mut window = nums.iter().take(len).map(|&num| num as i64).sum::<i64>();
        for i in k..nums.len() - k {
            res[i] = (window / len as i64) as i32;
            window -= nums[i - k] as i64;
            if i + k + 1 < nums.len() {
                window += nums[i + k + 1] as i64;
            }
        }

        res
    }

    /*
    link: https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/
     */

    pub fn longest_subarray(nums: Vec<i32>) -> i32 {
        let mut res = 0;
        let mut left = 0;
        let mut right = 0;
        let mut count = 0;
        while right < nums.len() {
            if nums[right] == 0 {
                count += 1;
            }
            while count > 1 {
                if nums[left] == 0 {
                    count -= 1;
                }
                left += 1;
            }
            res = res.max(right - left);
            right += 1;
        }
        res as i32
    }

    /*
    link: https://leetcode.com/problems/minimum-size-subarray-sum/

    use a sliding window to solve this problem
    i,j respectively represent the left and right pointer of the window
    sum is the sum of the window
    if sum >= target, we need to move the left pointer to the right and update the sum
    update result = min(result, j - i + 1) (current window size)
     */
    pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
        let mut res = usize::MAX;
        let mut sum = 0;
        let mut i = 0;
        for j in 0..nums.len() {
            sum += nums[j];
            while sum >= target {
                res = res.min(j - i + 1);
                sum -= nums[i];
                i += 1;
            }
        }
        if res == usize::MAX {
            0
        } else {
            res as i32
        }
    }
}

pub fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_vowels() {
        assert_eq!(SlidingWindow::max_vowels("abciiidef".to_string(), 3), 3);
        assert_eq!(SlidingWindow::max_vowels("aeiou".to_string(), 2), 2);
        assert_eq!(SlidingWindow::max_vowels("leetcode".to_string(), 3), 2);
        assert_eq!(SlidingWindow::max_vowels("rhythms".to_string(), 4), 0);
        assert_eq!(SlidingWindow::max_vowels("tryhard".to_string(), 4), 1);
    }

    #[test]
    fn test_four_sum() {
        let mut res = SlidingWindow::four_sum(vec![1, 0, -1, 0, -2, 2], 0);
        res.sort();
        assert_eq!(
            res,
            vec![vec![-2, -1, 1, 2], vec![-2, 0, 0, 2], vec![-1, 0, 0, 1]]
        );
        let mut res = SlidingWindow::four_sum(vec![-2, -1, -1, 1, 1, 2, 2], 0);
        res.sort();
        assert_eq!(res, vec![vec![-2, -1, 1, 2], vec![-1, -1, 1, 1]]);
    }

    #[test]
    fn test_max_consecutive_answers() {
        assert_eq!(
            SlidingWindow::max_consecutive_answers("TTFF".to_string(), 2),
            4
        );
        assert_eq!(
            SlidingWindow::max_consecutive_answers("TFFT".to_string(), 1),
            3
        );
        assert_eq!(
            SlidingWindow::max_consecutive_answers("TTFTTFTT".to_string(), 1),
            5
        );
    }

    #[test]
    fn test_get_averages() {
        assert_eq!(
            SlidingWindow::get_averages(vec![7, 4, 3, 9, 1, 8, 5, 2, 6], 3),
            vec![-1, -1, -1, 5, 4, 4, -1, -1, -1]
        );
    }

    #[test]
    fn test_longest_sub_array() {
        assert_eq!(SlidingWindow::longest_subarray(vec![1, 1, 0, 1]), 3);
        assert_eq!(
            SlidingWindow::longest_subarray(vec![0, 1, 1, 1, 0, 1, 1, 0, 1]),
            5
        );
        assert_eq!(SlidingWindow::longest_subarray(vec![1, 1, 1]), 2);
        assert_eq!(
            SlidingWindow::longest_subarray(vec![1, 1, 0, 0, 1, 1, 1, 0, 1]),
            4
        );
        assert_eq!(SlidingWindow::longest_subarray(vec![0, 0, 0]), 0);
    }

    #[test]
    fn test_min_sub_array_len() {
        assert_eq!(
            SlidingWindow::min_sub_array_len(7, vec![2, 3, 1, 2, 4, 3]),
            2
        );
        assert_eq!(SlidingWindow::min_sub_array_len(4, vec![1, 4, 4]), 1);
        assert_eq!(
            SlidingWindow::min_sub_array_len(11, vec![1, 1, 1, 1, 1, 1, 1, 1]),
            0
        );
    }
}
