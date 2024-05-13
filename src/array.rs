#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    ops::Div,
};

pub struct ArraySolution {}

impl ArraySolution {
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

    pub fn next_permutation(nums: &mut [i32]) {
        // rustic way: use windows and rposition to find the largest index k such that a[k] < a[k + 1]
        // if let Some(k) = nums.windows(2).rposition(|w| w[0] < w[1]) {
        // use rposition to find the largest index l greater than k such that a[k] < a[l]
        //     let j = nums.iter().rposition(|&x| x > nums[k]).unwrap();
        //     nums.swap(k, j);
        //     nums[k + 1..].reverse();
        // } else {
        //     nums.reverse();
        // }
        if nums.len() <= 1 {
            return;
        }
        let mut k = nums.len() - 2;
        while k > 0 && nums[k] >= nums[k + 1] {
            k -= 1;
        }
        // no such index exists, the permutation is the last permutation,
        // so we can reverse the array
        if k == 0 && nums[k] >= nums[k + 1] {
            nums.reverse();
            return;
        }
        let mut l = nums.len() - 1;
        while l >= k && nums[k] >= nums[l] {
            l -= 1;
        }
        nums.swap(k, l);
        nums[k + 1..].reverse();
    }

    /*
    link: https://leetcode.com/problems/permutations/
    easily solved by the next_permutation solved before
     */
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let mut prev_nums = nums.clone();

        loop {
            let mut nums_copy = prev_nums.clone();
            Self::next_permutation(&mut nums_copy);
            res.push(nums_copy.clone());
            prev_nums = nums_copy;
            if prev_nums == nums {
                break;
            }
        }

        res
    }

    pub fn array_sign(nums: Vec<i32>) -> i32 {
        nums.iter().map(|x| x.signum()).product()
    }

    /*
    link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
     */
    pub fn max_profits(prices: Vec<i32>) -> i32 {
        // rustic way but slower
        // prices
        //     .iter()
        //     .fold((i32::MAX, 0), |(min_price, max_profit), price| {
        //         (min_price.min(*price), max_profit.max(price - min_price))
        //     })
        //     .1
        let mut max_profit = i32::MIN;
        let mut min_price = i32::MAX;
        for price in prices {
            min_price = min_price.min(price);
            max_profit = max_profit.max(price - min_price);
        }
        max_profit
    }

    /*
    link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    On each day, you may decide to buy and/or sell the stock.
    You can only hold at most one share of the stock at any time.
    However, you can buy it then immediately sell it on the same day.

    the max profit at day[i] we can make is prices[i] - min(prices[0..i])
    so we can calculate the max profit at each day
    and sum them up to get the max profit
    we can find the subarray that is increasing
    and sum up the difference between the first and last element
    time complexity: O(n) space complexity: O(1)

     */
    pub fn max_profit_ii(prices: Vec<i32>) -> i32 {
        // one line solution
        // prices
        //     .windows(2)
        //     .fold(0, |acc, w| acc + (w[1] - w[0]).max(0));
        let mut res = 0;
        for i in 0..prices.len() - 1 {
            res += (prices[i + 1] - prices[i]).max(0);
        }

        res
    }

    /*
    link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
    we can not buy on the day and the day after we sell,
    so we can use two variables to represent the max profit at day[i]
    1. hold[i]: the max profit at day[i] we can make if we hold the stock
    2. sold[i]: the max profit at day[i] we can make if we sold the stock
    hold[i] = max(hold[i-1],     hold[i-1] means we do nothing at day[i],
                  sold[i-2] - prices[i]) . sold[i-2] means we sold the stock at day[i-2] and do nothing at day[i-1]

    sold[i] = max(sold[i-1],  sold[i-1] means we do nothing at day[i]
                 hold[i-1] + prices[i]) hold[i-1] means we hold the stock at day[i-1] and sell it at day[i]
    time complexity: O(n) space complexity: O(1)
    at the end, we can get the max profit at day[n] from sold[n]
     */

    pub fn max_profit_with_cool_down(prices: Vec<i32>) -> i32 {
        let mut hold = vec![i32::MIN; prices.len() + 1];
        let mut sold = vec![0; prices.len() + 1];

        for i in 1..=prices.len() {
            hold[i] = hold[i - 1].max(if i >= 2 { sold[i - 2] } else { 0 } - prices[i - 1]);
            sold[i] = sold[i - 1].max(hold[i - 1] + prices[i - 1]);
        }

        sold[prices.len()]
    }

    /*
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    can only complete at most two transactions

    define dp
     */

    pub fn max_profit_iii(prices: Vec<i32>) -> i32 {
        if prices.len() < 2 {
            return 0;
        }
        let mut dp = vec![vec![0; prices.len()]; 3];
        for i in 1..=2 {
            let mut min = prices[0];
            for j in 1..prices.len() {
                min = min.min(prices[j] - dp[i - 1][j - 1]);
                dp[i][j] = dp[i][j - 1].max(prices[j] - min);
            }
        }
        dp[2][prices.len() - 1]
    }

    /*
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/
    can only complete at most k transactions

    if k >= prices.len()/2, we can use the solution of best-time-to-buy-and-sell-stock-ii
    else we can use dp to solve this problem
    dp[i][j] represents the max profit at day[j] we can make if we complete at most i transactions
    dp[i][j] = max(dp[i][j-1], max_diff + prices[j])
    max_diff = max(max_diff, dp[i-1][j] - prices[j])
    max_diff means the max profit at day[j] we can make if we complete at most i-1 transactions
    time complexity: O(n*k) space complexity: O(n*k)

    */
    pub fn max_profit_iv(k: i32, prices: Vec<i32>) -> i32 {
        let k = k as usize;
        // if k >= prices.len() / 2 {
        //     return prices
        //         .windows(2)
        //         .fold(0, |acc, w| acc + (w[1] - w[0]).max(0));
        // }
        let mut dp = vec![vec![0; prices.len()]; k + 1];
        for i in 1..=k {
            let mut max_diff = -prices[0];
            for j in 1..prices.len() {
                dp[i][j] = dp[i][j - 1].max(max_diff + prices[j]);
                max_diff = max_diff.max(dp[i - 1][j] - prices[j]);
            }
        }
        dp[k][prices.len() - 1]
    }

    /*
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
    we can buy and sell the stock at the same day,
    so we can use two variables to represent the max profit at day[i]
    1. cash[i]: the max profit at day[i] we can make if we do not hold the stock
    2. hold[i]: the max profit at day[i] we can make if we hold the stock
    cash[i] = max(cash[i-1], hold[i-1] + prices[i] - fee)
    hold[i] = max(hold[i-1], cash[i-1] - prices[i])
    time complexity: O(n) space complexity: O(1)
    at the end, we can get the max profit at day[n] from cash[n]
    since we don't need to hold the stock at the end
    we can get the max profit at day[n] from cash[n]
     */

    pub fn max_profit_with_fee(prices: Vec<i32>, fee: i32) -> i32 {
        let mut cash = 0;
        let mut hold = -prices[0];
        for price in prices.iter().skip(1) {
            cash = cash.max(hold + price - fee);
            hold = hold.max(cash - price);
        }
        cash
    }

    /*
    link: https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/
     */

    pub fn average(nums: Vec<i32>) -> f64 {
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        let mut sum = 0;
        let len = nums.len() - 2;
        for num in nums {
            min = min.min(num);
            max = max.max(num);
            sum += num;
        }
        (sum - min - max) as f64 / len as f64
    }

    /*
    link: https://leetcode.com/problems/remove-element/
    remove all the element that is equal to val in place
    we can use two pointers to travel the array
    pointer i is the slow pointer points to the element that is not equal to val
    pointer j is the fast pointer points to the element that is equal to val
    every time nums[j] != val, we copy nums[j] to nums[i] and i += 1
    so that we can remove all the element that is equal to val
    time complexity: O(n) space complexity: O(1)
     */

    pub fn remove_element(nums: &mut [i32], val: i32) -> i32 {
        let mut i = 0;
        let mut j = 0;
        while j < nums.len() {
            if nums[j] != val {
                nums[i] = nums[j];
                i += 1;
            }
            j += 1;
        }
        i as i32
    }

    /*
    link: https://leetcode.com/problems/rotate-image/description/
    rotate the image in place
    we can use two steps to solve this problem
    1. reverse up to down
    2. swap the symmetry
     */

    pub fn rotate(image: &mut [Vec<i32>]) {
        let n = image.len();
        // transpose
        for i in 0..n.div(2) {
            image.swap(i, n - i - 1);
        }
        // swap the symmetry
        for i in 0..n {
            for j in i + 1..n {
                image[i][j] ^= image[j][i];
                image[j][i] ^= image[i][j];
                image[i][j] ^= image[j][i];
            }
        }
    }

    /*
    link: https://leetcode.com/problems/maximum-subarray/
    for nums[i], we have 2 options, either add it to the previous subarray or start a new subarray
    1. add to the previous subarray: sum = sum + nums[i]
    2. start a new subarray: sum = nums[i]
    this two options can be represented by sum = max(sum, 0) + nums[i]
    if sum < 0, we can start a new subarray, else we can add it to the previous subarray
    time complexity: O(n) space complexity: O(1)
     */
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut max_sum = i32::MIN;
        let mut sum = 0;
        for num in nums {
            sum = sum.max(0) + num;
            max_sum = max_sum.max(sum);
        }
        max_sum
    }

    /*
    link: https://leetcode.com/problems/spiral-matrix/

     */
    pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
        let mut res = vec![];
        if matrix.is_empty() {
            return res;
        }
        let mut length_y = matrix[0].len();
        let mut length_x = matrix.len();
        let total_num = length_x * length_y;
        let (mut x, mut y) = (0, 0); // pointing to the current pos
        loop {
            // 4 steps in a loop
            // 1. to the right, y++ until y == length_y-1
            // 2. to the down, x++ until x == length_x-1
            // 3. to the left, y-- until y == (y+1)(when the loop starts)
            // 4. to the up, x-- until x == (x) (when the loop starts)
            // if result.len() == length_x * length_y, we can break the loop
            if res.len() >= total_num {
                break;
            }

            // to the right
            while y < length_y {
                res.push(matrix[x][y]);
                y += 1;
            }
            y -= 1;

            if res.len() >= total_num {
                break;
            }

            // to the down
            x += 1;
            while x < length_x {
                res.push(matrix[x][y]);
                x += 1;
            }
            x -= 1;

            if res.len() >= total_num {
                break;
            }

            // to the left
            y -= 1;
            while y > matrix[0].len() - length_y {
                res.push(matrix[x][y]);
                y -= 1;
            }
            res.push(matrix[x][y]);

            if res.len() >= total_num {
                break;
            }

            // to the up
            x -= 1;
            while x > matrix.len() - length_x {
                res.push(matrix[x][y]);
                x -= 1;
            }
            x += 1;
            y += 1;

            if res.len() >= total_num {
                break;
            }

            // update the length_x and length_y
            length_x -= 1;
            length_y -= 1;
        }

        res
    }

    /*
    link: https://leetcode.com/problems/spiral-matrix-ii/
     */

    pub fn spiral_matrix_ii(n: i32) -> Vec<Vec<i32>> {
        let mut matrix = vec![vec![0i32; n as usize]; n as usize];
        let mut length_y = n as usize;
        let mut length_x = n as usize;
        let total_num = n * n;
        let (mut x, mut y) = (0, 0); // pointing to the current pos
        let mut num = 1i32;
        loop {
            // 4 steps in a loop
            // 1. to the right, y++ until y == length_y-1
            // 2. to the down, x++ until x == length_x-1
            // 3. to the left, y-- until y == (y+1)(when the loop starts)
            // 4. to the up, x-- until x == (x) (when the loop starts)
            // if result.len() == length_x * length_y, we can break the loop
            if num > total_num {
                break;
            }

            // to the right
            while y < length_y {
                matrix[x][y] = num;
                num += 1;
                y += 1;
            }
            y -= 1;

            if num > total_num {
                break;
            }

            // to the down
            x += 1;
            while x < length_x {
                matrix[x][y] = num;
                num += 1;
                x += 1;
            }
            x -= 1;

            if num > total_num {
                break;
            }

            // to the left
            y -= 1;
            while y > matrix[0].len() - length_y {
                matrix[x][y] = num;
                num += 1;
                y -= 1;
            }
            matrix[x][y] = num;
            num += 1;

            if num > total_num {
                break;
            }

            // to the up
            x -= 1;
            while x > matrix.len() - length_x {
                matrix[x][y] = num;
                num += 1;
                x -= 1;
            }
            x += 1;
            y += 1;

            // update the length_x and length_y
            length_x -= 1;
            length_y -= 1;
        }
        matrix
    }

    /*
    link: https://leetcode.com/problems/jump-game/description/
    max val represents the max index we can reach at current position
    if i > max_val, that means we can't reach the end
     */

    pub fn can_jump(steps: Vec<i32>) -> bool {
        let mut max_val = 0;
        for (i, v) in steps.iter().enumerate() {
            if i > max_val {
                return false;
            }
            max_val = max_val.max(i + *v as usize);
        }
        true
    }

    /*
    link: https://leetcode.com/problems/merge-intervals/
     */

    pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut res = Vec::new();
        if intervals.is_empty() {
            return res;
        }
        let mut intervals = intervals;
        intervals.sort_unstable_by_key(|val| val[0]);
        let mut prev = intervals[0].clone();
        for i in intervals.iter().skip(1) {
            if i[0] <= prev[1] {
                prev = Self::merge_intervals(prev, i.clone());
            } else {
                res.push(prev);
                prev = i.clone();
            }
        }
        res.push(prev);
        res
    }

    fn merge_intervals(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {
        vec![a[0].min(b[0]), a[1].max(b[1])]
    }

    pub fn length_of_last_word(s: String) -> i32 {
        s.split_ascii_whitespace().last().unwrap_or("").len() as i32
    }

    pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
        let mut digits = digits;
        let mut i = digits.len() - 1;
        loop {
            if digits[i] == 9 {
                digits[i] = 0;
                if i == 0 {
                    digits.insert(0, 1);
                    break;
                }
                i -= 1;
            } else {
                digits[i] += 1;
                break;
            }
        }
        digits
    }

    pub fn insert_interval(intervals: Vec<Vec<i32>>, new_intervals: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = Vec::new();
        let mut intervals = intervals;
        intervals.push(new_intervals);
        intervals.sort_unstable_by_key(|val| val[0]);
        let mut prev = intervals[0].clone();
        for i in intervals.iter().skip(1) {
            if i[0] <= prev[1] {
                prev = Self::merge_intervals(prev, i.clone());
            } else {
                res.push(prev);
                prev = i.clone();
            }
        }
        res.push(prev);
        res
    }

    /*
    link: https://leetcode.com/problems/move-zeroes/
    same solution at remove element
     */

    pub fn move_zeros(nums: &mut [i32]) {
        let mut i = 0;
        let mut j = 0;
        while j < nums.len() {
            if nums[j] != 0 {
                nums[i] = nums[j];
                i += 1;
            }
            j += 1;
        }
        while i < nums.len() {
            nums[i] = 0;
            i += 1;
        }
    }

    /*
    link: https://leetcode.com/problems/majority-element/
    https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm
     */

    pub fn majority_element(nums: Vec<i32>) -> i32 {
        let mut count = 0;
        let mut candidate = 0;
        for num in nums {
            if count == 0 {
                candidate = num;
            }
            count += if num == candidate { 1 } else { -1 };
        }
        candidate
    }

    /*
    link: https://leetcode.com/problems/longest-consecutive-sequence/

     */

    pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
        let mut set = HashSet::new();
        for num in nums.iter() {
            set.insert(num);
        }
        let mut res = 0;
        for &num in nums.iter() {
            if !set.contains(&(num - 1)) {
                let mut cur = num;
                let mut cur_len = 1;
                while set.contains(&(cur + 1)) {
                    cur += 1;
                    cur_len += 1;
                }
                res = res.max(cur_len);
            }
        }
        res
    }

    pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
        let mut flowerbed = flowerbed;
        let mut i = 0;
        let mut count = 0;
        while i < flowerbed.len() {
            if flowerbed[i] == 0
                && (i == 0 || flowerbed[i - 1] == 0)
                && (i == flowerbed.len() - 1 || flowerbed[i + 1] == 0)
            {
                flowerbed[i] = 1;
                count += 1;
            }
            i += 1
        }
        count >= n
    }

    pub fn erase_overlap_intervals(intervals: Vec<Vec<i32>>) -> i32 {
        let mut intervals = intervals;
        intervals.sort_unstable_by_key(|val| val[1]);
        let mut count = 0;
        let mut prev = intervals[0].clone();
        for i in intervals.iter().skip(1) {
            if i[0] < prev[1] {
                count += 1;
            } else {
                prev = i.clone();
            }
        }
        count
    }

    /*
    link: https://leetcode.com/problems/product-of-array-except-self/
    we use two variables to store the product of the left and right
    res[i] = left * right
    from the left to right with index i
    for left side, res[i], we can get the product of the left
    for right side, res[nums.len()-i-1], we can get the product of the right
    update the result in two directions, we can solve it in one pass
    when it reaches to the end, we can get the result
     */

    pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
        let mut res = vec![1; nums.len()];
        let mut left = 1;
        let mut right = 1;
        for i in 0..nums.len() {
            res[i] *= left;
            left *= nums[i];
            res[nums.len() - i - 1] *= right;
            right *= nums[nums.len() - i - 1];
        }
        res
    }

    /*
    link: https://leetcode.com/problems/asteroid-collision/
     */
    pub fn asteroid_collision(asteroids: Vec<i32>) -> Vec<i32> {
        let mut stack: Vec<i32> = Vec::new();

        for asteroid in asteroids {
            if stack.is_empty() || (stack.last().unwrap().is_positive() == asteroid.is_positive()) {
                stack.push(asteroid);
            } else {
                let survives = loop {
                    match stack.last() {
                        Some(&last) if last.is_positive() != asteroid.is_positive() => {
                            match last.abs().cmp(&asteroid.abs()) {
                                Ordering::Less => {
                                    stack.pop();
                                }
                                Ordering::Equal => {
                                    stack.pop();
                                    break false;
                                }
                                Ordering::Greater => {
                                    break false;
                                }
                            }
                        }
                        _ => break true,
                    }
                };
                if survives {
                    stack.push(asteroid);
                }
            }
        }
        stack
    }

    /*
    https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
     */

    pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
        let (mut front, mut back) = (0, numbers.len() - 1);
        while front < back {
            let sum = numbers[front] + numbers[back];
            match target.cmp(&sum) {
                Ordering::Less => back -= 1,
                Ordering::Equal => break,
                Ordering::Greater => front += 1,
            }
        }

        vec![front as i32 + 1, back as i32 + 1]
    }

    /*
    link: https://leetcode.com/problems/maximum-product-subarray/
    dp solve
    define two variables curr_min and curr_max
    curr_min stores the minimum product of subarray ending with nums[i]
    curr_max stores the maximum product of subarray ending with nums[i]
    curr_min = min(curr_min * nums[i], curr_max * nums[i], nums[i])
    curr_max = max(curr_min * nums[i], curr_max * nums[i], nums[i])
     */

    pub fn max_product(nums: Vec<i32>) -> i32 {
        let first_num = nums[0];
        let (mut curr_min, mut curr_max) = (1, 1);
        nums.into_iter().fold(first_num, |max_prod, n| {
            let tmp = n * curr_max;
            curr_max = n.max(tmp).max(n * curr_min);
            curr_min = n.min(tmp).min(n * curr_min);

            std::cmp::max(max_prod, curr_max)
        })
    }

    pub fn single_number(nums: Vec<i32>) -> i32 {
        let mut res = 0;
        for num in nums {
            res ^= num;
        }
        res
    }

    pub fn single_number_ii(nums: Vec<i32>) -> i32 {
        let mut res = 0;
        for i in 0..32 {
            let mut count = 0;
            for num in nums.iter() {
                if num >> i & 1 == 1 {
                    count += 1;
                }
            }
            if count % 3 != 0 {
                res |= 1 << i;
            }
        }
        res
    }

    pub fn missing_number(nums: Vec<i32>) -> i32 {
        let mut res = 0;
        for (i, num) in nums.iter().enumerate() {
            res ^= i as i32 ^ num;
        }
        res ^ nums.len() as i32
    }

    /*
    link: https://leetcode.com/problems/make-two-arrays-equal-by-reversing-subarrays/
     */
    pub fn can_be_equal(target: Vec<i32>, arr: Vec<i32>) -> bool {
        let mut target = target;
        let mut arr = arr;
        target.sort_unstable();
        arr.sort_unstable();
        target == arr
    }

    /*
    link: https://leetcode.com/problems/trapping-rain-water/
    steps:
    1. find the max height of the left and right
    2. if height[left] < height[right], we can trap the water from the left
    3. else we can trap the water from the right
    4. update the left and right
    5. repeat 1-4 until left >= right

    for position[i], we can trap the water min(left_max, right_max) - height[i]
     */

    pub fn trap_rain(height: Vec<i32>) -> i32 {
        let mut res = 0; // the result, res += min(left_max, right_max) - height[i]
        let mut left = 0; // left pointer
        let mut right = height.len() - 1; // right pointer
        let mut left_max = 0; // the max height of the left
        let mut right_max = 0; // the max height of the right
        while left < right {
            if height[left] < height[right] {
                if height[left] >= left_max {
                    left_max = height[left];
                } else {
                    res += left_max - height[left];
                }
                left += 1;
            } else {
                if height[right] >= right_max {
                    right_max = height[right];
                } else {
                    res += right_max - height[right];
                }
                right -= 1;
            }
        }
        res
    }

    /*
    link: https://leetcode.com/problems/maximum-sum-of-an-hourglass/
    the brut force solution is pretty straightforward
    the size and the shape of the hourglass is fixed with 3x3
    but the input is not,
    so we can use two for loops to traverse the input
    and calculate the sum of the hourglass
     */

    pub fn max_sum(grid: Vec<Vec<i32>>) -> i32 {
        let mut res = i32::MIN;
        for i in 0..grid.len() - 2 {
            for j in 0..grid[0].len() - 2 {
                let sum = grid[i][j]
                    + grid[i][j + 1]
                    + grid[i][j + 2]
                    + grid[i + 1][j + 1]
                    + grid[i + 2][j]
                    + grid[i + 2][j + 1]
                    + grid[i + 2][j + 2];
                res = res.max(sum);
            }
        }
        res
    }

    /*
    link: https://leetcode.com/problems/stone-game-vii/
    at turn, alice can choose the left or right stone, and get the score of the rest of the stones
    bob can do the same thing

    max the difference between alice and bob

    we can use dp to solve this problem
    dp[i][j] represents the max difference between alice and bob at stones[i..j]
    i, j means the index of the stones, where stones i,j remains
    dp[i][j] = max(sum[i..j] - dp[i+1][j],   if alice choose the left stone
                    sum[i..j] - dp[i][j-1])     if alice choose the right stone
    time complexity: O(n^2) space complexity: O(n^2)

     */
    pub fn stone_game_vii(stones: Vec<i32>) -> i32 {
        let mut dp = vec![vec![0; stones.len()]; stones.len()];
        let sum = {
            let mut sum = vec![0; stones.len() + 1];
            for i in 0..stones.len() {
                sum[i + 1] = sum[i] + stones[i];
            }
            sum
        };

        for i in (0..stones.len() - 1).rev() {
            for j in i + 1..stones.len() {
                dp[i][j] =
                    (sum[j + 1] - sum[i + 1] - dp[i + 1][j]).max(sum[j] - sum[i] - dp[i][j - 1]);
            }
        }
        dp[0][stones.len() - 1]
    }

    /*
    link: https://leetcode.com/problems/sort-colors/
    we can use three pointers to solve this problem
    i points to the element that is not equal to 0
    j points to the element that is not equal to 1
    k points to the element that is not equal to 2
    we can use two pointers to travel the array
    pointer j is the slow pointer points to the element that is not equal to 1
    pointer k is the fast pointer points to the element that is equal to 2
    every time nums[j] != val, we copy nums[j] to nums[i] and i += 1
    so that we can remove all the element that is equal to val
    time complexity: O(n) space complexity: O(1)

    but there is a more straightforward solution
    we can count sort the array
    same time complexity, but more straightforward

     */
    pub fn sort_colors(nums: &mut [i32]) {
        let (mut i, mut j, mut k) = (0, 0usize, nums.len() - 1);
        while j <= k {
            match nums[j] {
                0 => {
                    nums.swap(i, j);
                    i += 1;
                    j += 1;
                }
                1 => {
                    j += 1;
                }
                2 => {
                    nums.swap(j, k);
                    if k > 0 {
                        k -= 1;
                    } else {
                        break;
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
        let mut rows = vec![vec![false; 9]; 9];
        let mut cols = vec![vec![false; 9]; 9];
        let mut boxes = vec![vec![false; 9]; 9];
        for i in 0..9 {
            for (j, col) in cols.iter_mut().enumerate().take(9) {
                if board[i][j] != '.' {
                    let num = board[i][j] as usize - '1' as usize;
                    let box_index = (i / 3) * 3 + j / 3;
                    if rows[i][num] || col[num] || boxes[box_index][num] {
                        return false;
                    }
                    rows[i][num] = true;
                    col[num] = true;
                    boxes[box_index][num] = true;
                }
            }
        }
        true
    }

    /*
    link: https://leetcode.com/problems/set-matrix-zeroes/description/
    we can use the first row and first col to store the information
    if matrix[i][j] == 0, we can set matrix[i][0] = 0 and matrix[0][j] = 0
    then we can use the first row and first col to set the matrix to zero
     */

    pub fn set_zero(matrix: &mut [Vec<i32>]) {
        let mut first_row = false;
        let mut first_col = false;
        for row in matrix.iter_mut() {
            if row[0] == 0 {
                first_col = true;
                break;
            }
        }
        for j in 0..matrix[0].len() {
            if matrix[0][j] == 0 {
                first_row = true;
                break;
            }
        }
        for i in 1..matrix.len() {
            for j in 1..matrix[0].len() {
                if matrix[i][j] == 0 {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for i in 1..matrix.len() {
            for j in 1..matrix[0].len() {
                if matrix[i][0] == 0 || matrix[0][j] == 0 {
                    matrix[i][j] = 0;
                }
            }
        }
        if first_row {
            for j in 0..matrix[0].len() {
                matrix[0][j] = 0;
            }
        }
        if first_col {
            for row in matrix {
                row[0] = 0;
            }
        }
    }

    /*
    https://leetcode.com/problems/first-missing-positive/description/

    must solve with O(n) time complexity and O(1) space complexity

    the basic idea is to treat the num in the array as an index.
    if the array is sorted, the first missing positive number should be the first
    non-positive and non-consecutive number.
    this number should fall into the range [1, nums.len()]
    for all the number that negative and greater than nums.len()
    we can simply set them to nums.len() + 1 for they left some solt in the array.
    for the rest of the positive number, use them as index and set corresponding positive as negative,
    then we can traverse the array and found the first positive number, the index of that position is
    the first missing positive number.
    f we couldn't find such a number, the means the numbers in the input array all falls into the
    range [1,nums.len()], return nums.len() + 1 as result


     */
    pub fn first_missing_positive(nums: Vec<i32>) -> i32 {
        let mut nums = nums;

        // remove all the non-positive and greater than nums.len() elements
        // and replace them with nums.len() + 1 so we can ignore them
        for i in 0..nums.len() {
            if nums[i] <= 0 || nums[i] > nums.len() as i32 {
                nums[i] = nums.len() as i32 + 1;
            }
        }
        // use the index to store the information
        // nums[i] = -nums[i].abs() means the num i+1 exists in the array
        // nums[i] = nums.len() + 1 means the num i+1 doesn't exist in the array
        // if nums[i] == num, we can set nums[num-1] = -nums[num-1].abs() so
        for i in 0..nums.len() {
            let num = nums[i].unsigned_abs();
            if num as usize <= nums.len() {
                nums[num as usize - 1] = -nums[num as usize - 1].abs();
            }
        }
        // find the first num that is greater than 0
        // if we can't find it, that means all the nums in the array exists
        // if we found a num that is > 0, that means there is an empty slot in the array,
        // i represents the index of the empty slot
        for (i, &num) in nums.iter().enumerate() {
            if num > 0 {
                return i as i32 + 1;
            }
        }

        nums.len() as i32 + 1
    }

    /*
    https://leetcode.com/problems/number-of-good-pairs/

    a pair is good if nums[i] == nums[j] and i < j

    we can use a hashmap to store the repetitions of each number
    and calculate the number of good pairs.
    the good number is the combination of the index of the number
    we pick two index from n index
    so the combination of the index is C(n,2) = n * (n-1) / 2
    n is the repetitions of the number

    get the sum of all the good pairs
    return the sum as result
     */

    fn number_of_good_pairs(numbers: Vec<i32>) -> i32 {
        let mut res = 0;
        let mut map = HashMap::new();
        for num in numbers {
            *map.entry(num).or_insert(0) += 1;
        }

        for (_, v) in map {
            res += v * (v - 1) / 2;
        }
        res
    }

    /*
    https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/

     */

    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        *nums = nums
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| {
                if i > 1 && nums[i - 2] == n {
                    None
                } else {
                    Some(n)
                }
            })
            .collect();
        nums.len() as i32
    }

    /*
    https://leetcode.com/problems/average-value-of-even-numbers-that-are-divisible-by-three/

     */

    pub fn average_value(nums: Vec<i32>) -> i32 {
        let (mut sum, mut cnt) = (0, 0);
        for num in nums {
            if num % 2 == 0 && num % 3 == 0 {
                sum += num;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return 0;
        }
        sum / cnt
    }

    pub fn count_donut(grids: Vec<Vec<char>>) -> usize {
        let directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ];
        let mut ans = 0;

        for i in 0..grids.len() {
            for j in 0..grids[i].len() {
                if grids[i][j] == '.' {
                    let mut is_donut = true;
                    for (dx, dy) in directions {
                        let (nx, ny) = (i as i32 + dx, j as i32 + dy);
                        if nx < 0
                            || nx >= grids.len() as i32
                            || ny < 0
                            || ny >= grids[0].len() as i32
                            || grids[nx as usize][ny as usize] != '#'
                        {
                            is_donut = false;
                            break;
                        }
                    }
                    if is_donut {
                        ans += 1;
                    }
                }
            }
        }

        ans
    }

    /*
    https://leetcode.com/problems/convert-an-array-into-a-2d-array-with-conditions/description/
     */

    pub fn find_matrix(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        nums.sort_unstable();
        let mut res = Vec::new();
        let mut i = 0;
        while i < nums.len() {
            let mut j = i + 1;
            while j < nums.len() && nums[j] == nums[i] {
                j += 1;
            }
            for k in 0..j - i {
                if k >= res.len() {
                    res.push(Vec::new());
                }
                res[k].push(nums[i]);
            }
            i = j;
        }

        res
    }

    /*
    https://leetcode.com/problems/surrounded-regions/description/
     */

    pub fn surround_regions(board: &mut [Vec<char>]) {
        let mut visited = vec![vec![false; board[0].len()]; board.len()];
        for i in 1..board.len() - 1 {
            for j in 1..board[i].len() - 1 {
                if board[i][j] == 'O' && !visited[i][j] {
                    let mut is_surrounded = true;
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((i, j));
                    visited[i][j] = true;
                    while let Some((x, y)) = queue.pop_front() {
                        if x == 0 || x == board.len() - 1 || y == 0 || y == board[0].len() - 1 {
                            is_surrounded = false;
                        }
                        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)].iter() {
                            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                            if nx >= 0
                                && nx < board.len() as i32
                                && ny >= 0
                                && ny < board[0].len() as i32
                                && board[nx as usize][ny as usize] == 'O'
                                && !visited[nx as usize][ny as usize]
                            {
                                queue.push_back((nx as usize, ny as usize));
                                visited[nx as usize][ny as usize] = true;
                            }
                        }
                    }
                    if is_surrounded {
                        for i in 0..board.len() {
                            for j in 0..board[i].len() {
                                if visited[i][j] {
                                    board[i][j] = 'X';
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /*
    https://leetcode.com/problems/gas-station/
     */

    pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let mut total = 0;
        let mut curr = 0;
        let mut start = 0;
        for i in 0..gas.len() {
            total += gas[i] - cost[i];
            curr += gas[i] - cost[i];
            // if we run out of gas at station i
            // all the station between start and i can't be the start station
            // because we can't reach station i from these stations
            // so we can start from station i+1
            // and reset the curr to 0
            if curr < 0 {
                start = i + 1;
                curr = 0;
            }
        }
        // if total < 0, that means we can't reach the end
        if total < 0 {
            -1
        } else {
            start as i32
        }
    }

    /*
    https://leetcode.com/problems/find-the-duplicate-number/description/

    nums contains n + 1 integers where each integer is in the range [1, n] inclusive.
    There is only one repeated number in nums, return this repeated number.

    since the range is [1, n] and there is only one duplicate number
    we can use index to store the information.
    if we sort the array, the duplicate number should be nums[i] == nums[i+1]

    if nums[i] == i, that means the num i exists in the array
    if nums[i] != i, we can swap nums[i] and nums[nums[i]]
    if nums[i] == nums[nums[i]], that means we found the duplicate number
    if nums[i] != nums[nums[i]], we swap nums[i] and nums[nums[i]] and continue

     */

    pub fn find_duplicate(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        let mut i = 0;
        while i < nums.len() {
            if nums[i] == i as i32 {
                i += 1;
            } else {
                let num = nums[i] as usize;
                if nums[num] == nums[i] {
                    return nums[i];
                }
                nums.swap(i, num);
            }
        }
        -1
    }

    pub fn largest_number(nums: Vec<i32>) -> String {
        let mut nums = nums.iter().map(|&n| n.to_string()).collect::<Vec<_>>();
        nums.sort_by(|a, b| {
            let ab = a.to_owned() + b;
            let ba = b.to_owned() + a;
            ab.cmp(&ba)
        });
        nums.reverse();
        if nums[0] == "0" {
            return "0".to_owned();
        }
        nums.join("")
    }

    pub fn divide_array(mut nums: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        nums.chunks(3)
            .map(|v| {
                if v[2] - v[0] <= k {
                    Some(v.to_vec())
                } else {
                    None
                }
            })
            .collect::<Option<Vec<Vec<i32>>>>()
            .unwrap_or_default()
    }

    /*
    https://leetcode.com/problems/group-the-people-given-the-group-size-they-belong-to/

    position i can join a group with size group_sizes[i]

    */

    pub fn group_the_people(group_sizes: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = Vec::new();
        let mut size2people = HashMap::new();
        for (i, &size) in group_sizes.iter().enumerate() {
            size2people.entry(size).or_insert(Vec::new()).push(i as i32);
            if size2people[&size].len() == size as usize {
                res.push(size2people.remove(&size).unwrap());
            }
        }
        res
    }

    pub fn candy(ratings: Vec<i32>) -> i32 {
        let mut candies = vec![1; ratings.len()];
        for i in 1..ratings.len() {
            if ratings[i] > ratings[i - 1] {
                candies[i] = candies[i - 1] + 1;
            }
        }
        for i in (0..ratings.len() - 1).rev() {
            if ratings[i] > ratings[i + 1] {
                candies[i] = candies[i].max(candies[i + 1] + 1);
            }
        }
        candies.iter().sum()
    }

    pub fn zigzag_convert(s: String, num_rows: i32) -> String {
        if s.len() <= 1 || num_rows == 1 {
            return s;
        }
        let mut rows = vec![String::new(); num_rows as usize];
        let mut curr_row = 0;
        let mut go_down = false;
        for c in s.chars() {
            rows[curr_row as usize].push(c);
            if curr_row == 0 || curr_row == num_rows - 1 {
                go_down = !go_down;
            }
            curr_row += if go_down { 1 } else { -1 };
        }
        rows.concat()
    }

    pub fn rearrange_array(nums: Vec<i32>) -> Vec<i32> {
        let pos = nums.iter().filter(|&&n| n >= 0);
        let negs = nums.iter().filter(|&&n| n < 0);
        let mut res = Vec::new();
        for (p, n) in pos.zip(negs) {
            res.push(*p);
            res.push(*n);
        }
        res
    }

    pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
        let mut res = vec![vec![0; n as usize]; n as usize];
        let mut layer = 0;
        let mut num = 1;
        while num <= n * n {
            for i in layer..n - layer {
                res[layer as usize][i as usize] = num;
                num += 1;
            }
            for i in layer + 1..n - layer {
                res[i as usize][(n - layer - 1) as usize] = num;
                num += 1;
            }
            for i in (layer..n - layer - 1).rev() {
                res[(n - layer - 1) as usize][i as usize] = num;
                num += 1;
            }
            for i in (layer + 1..n - layer - 1).rev() {
                res[i as usize][layer as usize] = num;
                num += 1;
            }
            layer += 1;
        }
        res
    }

    pub fn largest_perimeter(nums: Vec<i32>) -> i64 {
        let mut nums = nums;
        nums.sort_unstable_by_key(|&n| Reverse(n));
        let mut sum = nums.iter().map(|v| *v as i64).sum();
        let mut count = nums.len();
        for n in nums {
            let n = n as i64;
            let remain = sum - n;
            if remain > n && count >= 3 {
                return sum;
            }
            count -= 1;
            sum -= n;
        }
        -1
    }

    /*
    remove the k number of elements to make at least number of unique elements

    a pretty straight forward solution is to remove the element that has the least repetitions
    after k iterations, the remaining elements are the least number of unique elements

    we can use a hashmap to store the repetitions of each element
    and a BTreeMap to store the element that has the same repetitions
    BTreeMap can help us to iterate the repetitions in ascending order
     */

    pub fn find_least_num_of_unique_ints(arr: Vec<i32>, k: i32) -> i32 {
        let mut ele2cnt: HashMap<i32, i32> = HashMap::new();
        let mut cnt2ele: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for num in arr {
            *ele2cnt.entry(num).or_default() += 1;
        }
        for (&num, &count) in ele2cnt.iter() {
            cnt2ele.entry(count).or_default().push(num);
        }
        let mut res = ele2cnt.len() as i32;
        let iter = cnt2ele.iter();
        let mut k = k;

        for (_, nums) in iter {
            for &num in nums.iter() {
                if k >= ele2cnt[&num] {
                    k -= ele2cnt[&num];
                    res -= 1;
                } else {
                    break;
                }
            }
        }

        res
    }

    pub fn furthest_building(heights: Vec<i32>, bricks: i32, ladders: i32) -> i32 {
        if heights.is_empty() {
            return 0;
        }
        let mut bricks = bricks;
        let mut heap = BinaryHeap::new();
        let mut ladders = ladders;
        let mut res = 0;
        let mut prev = heights[0];

        for &curr in heights.iter().skip(1) {
            if curr <= prev {
                prev = curr;
                res += 1;
                continue;
            }
            let diff = curr - prev;
            bricks -= diff;

            heap.push(diff);
            if bricks < 0 {
                bricks += heap.pop().expect("heap is empty");
                if ladders > 0 {
                    ladders -= 1;
                } else {
                    return res;
                }
            }
            res += 1;
            prev = curr;
        }

        res
    }

    pub fn most_booked_iii(n: i32, mut meetings: Vec<(i64, i64)>) -> i32 {
        meetings.sort();
        let mut rooms: BinaryHeap<Reverse<(i64, i32)>> = BinaryHeap::new();
        let mut ready: BinaryHeap<Reverse<i32>> =
            BinaryHeap::from((0..n).map(Reverse).collect::<Vec<_>>());
        let mut freqs = vec![0; n as usize];

        for (start, end) in meetings {
            while !rooms.is_empty() && rooms.peek().unwrap().0 .0 <= start {
                ready.push(Reverse(rooms.pop().unwrap().0 .1));
            }
            if let Some(Reverse(room_id)) = ready.pop() {
                rooms.push(Reverse((end, room_id)));
                freqs[room_id as usize] += 1;
            } else if let Some(Reverse((t, room_id))) = rooms.pop() {
                rooms.push(Reverse((t + end - start, room_id)));
                freqs[room_id as usize] += 1;
            }
        }

        let mut max_room = 0;
        for (r, &c) in freqs.iter().enumerate().skip(1) {
            if c > freqs[max_room] {
                max_room = r
            }
        }

        max_room as i32
    }

    pub fn find_judge(n: i32, trust: Vec<Vec<i32>>) -> i32 {
        let mut trusts: HashMap<i32, Vec<_>> = HashMap::new();
        let mut trust_by: HashMap<i32, i32> = HashMap::new();
        for t in trust {
            let (a, b) = (t[0], t[1]);
            trusts.entry(a).or_default().push(b);
            *trust_by.entry(b).or_default() += 1;
        }
        for i in 1..=n {
            if trusts.get(&i).is_none() && trust_by.get(&i).unwrap_or(&0) == &(n - 1) {
                return i;
            }
        }
        -1
    }

    pub fn is_monotonic(nums: Vec<i32>) -> bool {
        let mut inc = true;
        let mut dec = true;
        for i in 1..nums.len() {
            match nums[i].cmp(&nums[i - 1]) {
                Ordering::Less => inc = false,
                Ordering::Greater => dec = false,
                _ => {}
            }
        }
        inc || dec
    }

    pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
        // bad performance
        // let mut heap = BinaryHeap::new();
        // for n in nums {
        //     heap.push(Reverse(n as i64 * n as i64));
        // }
        // let mut res = Vec::new();
        // while let Some(Reverse(n)) = heap.pop() {
        //     res.push(n as i32);
        // }
        // res
        if nums.len() == 1 {
            return vec![nums[0] * nums[0]];
        }

        let mut location = nums.len();
        let mut res = vec![0; nums.len()];

        let (mut left, mut right) = (0, nums.len() - 1);

        let (mut left_square, mut right_square) =
            (nums[left] * nums[left], nums[right] * nums[right]);

        while left != right {
            location -= 1;
            if left_square > right_square {
                res[location] = left_square;
                left += 1;
                left_square = nums[left] * nums[left];
            } else {
                res[location] = right_square;
                right -= 1;
                right_square = nums[right] * nums[right];
            }
        }

        res[0] = right_square;
        res
    }

    /*
    https://leetcode.com/problems/bag-of-tokens/

    greedy algorithm
     */

    pub fn bag_of_tokens_score(mut tokens: Vec<i32>, mut power: i32) -> i32 {
        if tokens.is_empty() {
            return 0;
        }
        let mut score = 0;

        tokens.sort_unstable();
        let (mut left, mut right) = (0, tokens.len() - 1);

        // if we have enough power, we can use the power to gain score
        // if we don't have enough power, we can use the score to gain power
        // the greedy algorithm is to use less token to gain more score
        // and use gain more power to gain more score
        while left < right {
            if power >= tokens[left] {
                power -= tokens[left];
                score += 1;
                left += 1;
            } else if score > 0 {
                power += tokens[right];
                score -= 1;
                right -= 1;
            } else {
                break;
            }
        }
        // if we still have enough power, we can use the power to gain score
        while left < right + 1 && power >= tokens[left] {
            power -= tokens[left];
            score += 1;
            left += 1;
        }

        score
    }

    /*
    https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/


     */

    pub fn min_operations(mut nums: Vec<i32>) -> i32 {
        let n = nums.len() as i32;
        nums.sort_unstable();
        nums.dedup(); // remove the duplicate elements

        nums.iter().enumerate().fold(n, |ops, (i, l)| {
            ops.min(n - (nums.partition_point(|&m| m < l + n) - i) as i32)
        })
    }

    pub fn max_frequency_elements(nums: Vec<i32>) -> i32 {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        for n in nums {
            *map.entry(n).or_default() += 1;
        }
        let max = *map.values().max().unwrap();
        map.values().filter(|&&v| v == max).sum()
    }

    pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        let mut set1: HashSet<i32> = nums1.into_iter().collect();
        let set2: HashSet<i32> = nums2.into_iter().collect();
        set1.retain(|n| set2.contains(n));
        set1.into_iter().collect()
    }

    pub fn num_subarray_with_sum(nums: Vec<i32>, goal: i32) -> i32 {
        // sum -> count, the occurrences of the prefix sum.
        // use sum - goal to get the number of subarray that has the sum equal to goal.
        let mut map: HashMap<i32, i32> = HashMap::new();
        let mut sum = 0; // sum of the prefix
        let mut res = 0; // the number of subarray that has the sum equal to goal
        for n in nums {
            sum += n;
            if sum == goal {
                res += 1;
            }
            // when sum >= goal, for current index I, what we are looking for is the number
            // of subarray that has the sum equal to goal, not to sum.
            // but the number of subarray that sums to goal equals to hte number of subarray that
            // sums to sum - goal
            res += *map.get(&(sum - goal)).unwrap_or(&0);
            // increase the occurrences of the prefix sum
            *map.entry(sum).or_default() += 1;
        }
        res
    }

    pub fn find_max_length(nums: Vec<i32>) -> i32 {
        let mut count = 0; // the number of 1s minus the number of 0s
        let mut map = HashMap::new(); // count -> index
        let mut max = 0;
        for i in 0..nums.len() {
            count += if nums[i] == 1 { 1 } else { -1 };

            if count == 0 {
                // if count == 0, that means the subarray from 0 to i is a valid subarray
                max = max.max(i + 1);
            } else if let Some(&j) = map.get(&count) {
                // if we found a subarray that has the same count,
                // that means the subarray from j+1 to i is a valid subarray
                max = max.max(i - j);
            } else {
                map.insert(count, i);
            }
        }

        max as i32
    }

    pub fn find_min_arrow_shots(mut points: Vec<Vec<i32>>) -> i32 {
        if points.is_empty() {
            return 0;
        }
        points.sort_unstable_by_key(|v| v[1]);
        let mut arrow_pos = points[0][1];
        let mut res = 1;
        for balloon in points.iter().skip(1) {
            if balloon[0] > arrow_pos {
                arrow_pos = balloon[1];
                res += 1;
            }
        }
        res
    }

    /*
    https://leetcode.com/problems/task-scheduler/
    arrange the task in such a way that the same task is at least n intervals apart
    how many intervals we need to schedule all the tasks.

    we don't actually need to arrange the task, we just need to calculate the number of intervals
    the result should be the least number of idle slots + the number of tasks

    we can initialize the idle slots to be the maximum number of idle slots
    the maximum number of idle slots is (max - 1) * n, max is the maximum number of the same task
    then we can update the idle slots by subtracting the number of the same task
    these tasks use the idle slots
     */

    pub fn least_interval(tasks: Vec<char>, n: i32) -> i32 {
        let mut map = [0; 26];
        for &task in &tasks {
            map[(task as u8 - b'A') as usize] += 1;
        }
        map.sort_unstable();
        let max = map.last().unwrap() - 1;
        let mut idle_slots = max * n;
        // skip last one
        for i in 0..25 {
            // update idle slots
            // subtract the number of the same task
            idle_slots -= map[i].min(max);
        }
        idle_slots.max(0) + tasks.len() as i32
    }

    pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
        let mut res = Vec::new();
        let mut nums = nums;
        for i in 0..nums.len() {
            let index = nums[i].unsigned_abs() as usize - 1;
            if nums[index] < 0 {
                // if the number is negative, that means we have seen the number
                res.push(index as i32 + 1);
            } else {
                // mark the number as negative
                nums[index] = -nums[index];
            }
        }
        res
    }

    pub fn number_of_boomerangs(points: Vec<Vec<i32>>) -> i32 {
        let mut res = 0;
        for p in &points {
            let mut map = HashMap::new(); // distance -> count

            // calculate the distance between each pair of points
            for q in &points {
                let d = (p[0] - q[0]).pow(2) + (p[1] - q[1]).pow(2);
                *map.entry(d).or_insert(0) += 1;
            }
            // for each pair of points, we can form a boomerang
            // the number of boomerangs that can be formed is the number of pairs
            // that has the same distance
            for &v in map.values() {
                res += v * (v - 1);
            }
        }
        res
    }

    pub fn matrix_score(mut grid: Vec<Vec<i32>>) -> i32 {
        let (m, n) = (grid.len(), grid[0].len());
        for i in 0..m {
            // if the first element of the row is 0, we can flip the row
            if grid[i][0] == 0 {
                for j in 0..n {
                    grid[i][j] ^= 1;
                }
            }
        }
        // if the number of 1s in the column is less than the number of 0s
        // we can flip the column
        for j in 1..n {
            let mut cnt = 0;
            for i in 0..m {
                cnt += grid[i][j];
            }
            if cnt * 2 < m as i32 {
                for i in 0..m {
                    grid[i][j] ^= 1;
                }
            }
        }
        grid.iter().fold(0, |acc, row| {
            acc + row.iter().fold(0, |acc, &v| acc * 2 + v)
        })
    }
} // impl ArraySolution

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_permutation() {
        let mut input = vec![1];
        ArraySolution::next_permutation(&mut input);
        assert_eq!(input, vec![1]);
        let mut input = vec![1, 2, 3];
        ArraySolution::next_permutation(&mut input);
        assert_eq!(input, vec![1, 3, 2]);
        let mut input = vec![3, 2, 1];
        ArraySolution::next_permutation(&mut input);
        assert_eq!(input, vec![1, 2, 3]);
        let mut input = vec![1, 1, 5];
        ArraySolution::next_permutation(&mut input);
        assert_eq!(input, vec![1, 5, 1]);
    }

    #[test]
    fn test_permutation() {
        let input = vec![1, 2, 3];
        let mut output = ArraySolution::permute(input);
        output.sort();
        let mut expected = vec![
            vec![1, 2, 3],
            vec![1, 3, 2],
            vec![2, 1, 3],
            vec![2, 3, 1],
            vec![3, 1, 2],
            vec![3, 2, 1],
        ];
        expected.sort();
        assert_eq!(output, expected);
        let input = vec![0, 1];
        let mut output = ArraySolution::permute(input);
        output.sort();
        let mut expected = vec![vec![0, 1], vec![1, 0]];
        expected.sort();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_array_sign() {
        assert_eq!(ArraySolution::array_sign(vec![-1, -2, -3, -4, 3, 2, 1]), 1);
        assert_eq!(ArraySolution::array_sign(vec![1, 5, 0, 2, -3]), 0);
        assert_eq!(ArraySolution::array_sign(vec![-1, 1, -1, 1, -1]), -1);
        assert_eq!(
            ArraySolution::array_sign(vec![i32::MAX / 2, i32::MAX / 2, i32::MIN]),
            -1
        );
    }

    #[test]
    fn test_remove_element() {
        let mut input = vec![3, 2, 2, 3];
        assert_eq!(ArraySolution::remove_element(&mut input, 3), 2);
        assert_eq!(input[0..2], vec![2, 2]);
        let mut input = vec![0, 1, 2, 2, 3, 0, 4, 2];
        assert_eq!(ArraySolution::remove_element(&mut input, 2), 5);
        assert_eq!(input[0..5], vec![0, 1, 3, 0, 4]);
    }

    #[test]
    fn test_rotate() {
        let mut input = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        ArraySolution::rotate(&mut input);
        assert_eq!(input, vec![vec![7, 4, 1], vec![8, 5, 2], vec![9, 6, 3]]);
        let mut input = vec![
            vec![5, 1, 9, 11],
            vec![2, 4, 8, 10],
            vec![13, 3, 6, 7],
            vec![15, 14, 12, 16],
        ];
        ArraySolution::rotate(&mut input);
        assert_eq!(
            input,
            vec![
                vec![15, 13, 2, 5],
                vec![14, 3, 4, 1],
                vec![12, 6, 8, 9],
                vec![16, 7, 10, 11],
            ]
        );
    }

    #[test]
    fn test_max_sub_array() {
        assert_eq!(
            ArraySolution::max_sub_array(vec![-2, 1, -3, 4, -1, 2, 1, -5, 4]),
            6
        );
        assert_eq!(ArraySolution::max_sub_array(vec![1]), 1);
        assert_eq!(ArraySolution::max_sub_array(vec![0]), 0);
        assert_eq!(ArraySolution::max_sub_array(vec![-1]), -1);
        assert_eq!(ArraySolution::max_sub_array(vec![-100000]), -100000);
    }

    #[test]
    fn test_spiral_order() {
        assert_eq!(
            ArraySolution::spiral_order(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]),
            vec![1, 2, 3, 6, 9, 8, 7, 4, 5]
        );
        assert_eq!(
            ArraySolution::spiral_order(vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
            ]),
            vec![1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
        );
    }

    #[test]
    fn test_can_jump() {
        assert_eq!(ArraySolution::can_jump(vec![2, 3, 1, 1, 4]), true);
        assert_eq!(ArraySolution::can_jump(vec![3, 2, 1, 0, 4]), false);
        assert_eq!(ArraySolution::can_jump(vec![0]), true);
        assert_eq!(ArraySolution::can_jump(vec![2, 0, 0]), true);
        assert_eq!(ArraySolution::can_jump(vec![1, 1, 2, 2, 0, 1, 1]), true);
        assert_eq!(ArraySolution::can_jump(vec![1, 2, 0, 1]), true);
    }

    #[test]
    fn test_merge() {
        assert_eq!(
            ArraySolution::merge(vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]]),
            vec![vec![1, 6], vec![8, 10], vec![15, 18]]
        );
    }

    #[test]
    fn test_length_of_last_word() {
        assert_eq!(
            ArraySolution::length_of_last_word("Hello World".to_string()),
            5
        );
        assert_eq!(ArraySolution::length_of_last_word(" ".to_string()), 0);
        assert_eq!(ArraySolution::length_of_last_word("a ".to_string()), 1);
    }

    #[test]
    fn test_plus_one() {
        assert_eq!(ArraySolution::plus_one(vec![1, 2, 3]), vec![1, 2, 4]);
        assert_eq!(ArraySolution::plus_one(vec![4, 3, 2, 1]), vec![4, 3, 2, 2]);
        assert_eq!(ArraySolution::plus_one(vec![0]), vec![1]);
        assert_eq!(ArraySolution::plus_one(vec![9]), vec![1, 0]);
        assert_eq!(ArraySolution::plus_one(vec![9, 9]), vec![1, 0, 0]);
        assert_eq!(ArraySolution::plus_one(vec![9, 9, 9]), vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_insert_interval() {
        assert_eq!(
            ArraySolution::insert_interval(vec![vec![1, 3], vec![6, 9]], vec![2, 5]),
            vec![vec![1, 5], vec![6, 9]]
        );
        assert_eq!(
            ArraySolution::insert_interval(
                vec![
                    vec![1, 2],
                    vec![3, 5],
                    vec![6, 7],
                    vec![8, 10],
                    vec![12, 16],
                ],
                vec![4, 8],
            ),
            vec![vec![1, 2], vec![3, 10], vec![12, 16]]
        );
        assert_eq!(
            ArraySolution::insert_interval(vec![vec![1, 5]], vec![2, 3]),
            vec![vec![1, 5]]
        );
    }

    #[test]
    fn test_spiral_matrix_ii() {
        assert_eq!(
            ArraySolution::spiral_matrix_ii(3),
            vec![vec![1, 2, 3], vec![8, 9, 4], vec![7, 6, 5]]
        );
        assert_eq!(ArraySolution::spiral_matrix_ii(1), vec![vec![1]]);
    }

    #[test]
    fn test_move_zeros() {
        let mut input = vec![0, 1, 0, 3, 12];
        ArraySolution::move_zeros(&mut input);
        assert_eq!(input, vec![1, 3, 12, 0, 0]);
        input = vec![0, 0, 1];
        ArraySolution::move_zeros(&mut input);
        assert_eq!(input, vec![1, 0, 0]);
    }

    #[test]
    fn test_majority_element() {
        assert_eq!(ArraySolution::majority_element(vec![3, 2, 3]), 3);
        assert_eq!(
            ArraySolution::majority_element(vec![2, 2, 1, 1, 1, 2, 2]),
            2
        );
    }

    #[test]
    fn test_longest_consecutive_sequence() {
        assert_eq!(
            ArraySolution::longest_consecutive(vec![100, 4, 200, 1, 3, 2]),
            4
        );
        assert_eq!(
            ArraySolution::longest_consecutive(vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1]),
            9
        );
    }

    #[test]
    fn test_can_place_flowers() {
        assert_eq!(
            ArraySolution::can_place_flowers(vec![1, 0, 0, 0, 1], 1),
            true
        );
        assert_eq!(
            ArraySolution::can_place_flowers(vec![1, 0, 0, 0, 1], 2),
            false
        );
        assert_eq!(
            ArraySolution::can_place_flowers(vec![1, 0, 0, 0, 0, 1], 2),
            false
        );
        assert_eq!(
            ArraySolution::can_place_flowers(vec![0, 0, 1, 0, 1], 1),
            true
        );
        assert_eq!(
            ArraySolution::can_place_flowers(vec![0, 0, 1, 0, 1], 2),
            false
        );
    }

    #[test]
    fn test_erase_overlap_intervals() {
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![vec![1, 2], vec![2, 3], vec![3, 4]]),
            0
        );
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![vec![1, 2], vec![1, 2], vec![1, 2]]),
            2
        );
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![vec![1, 2], vec![2, 3]]),
            0
        );
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![
                vec![1, 100],
                vec![11, 22],
                vec![1, 11],
                vec![2, 12],
            ]),
            2
        );
    }

    #[test]
    fn test_product_except_self() {
        assert_eq!(
            ArraySolution::product_except_self(vec![1, 2, 3, 4]),
            vec![24, 12, 8, 6]
        );
    }

    #[test]
    fn test_asteroid_collision() {
        assert_eq!(
            ArraySolution::asteroid_collision(vec![5, 10, -5]),
            vec![5, 10]
        );
        assert_eq!(
            ArraySolution::asteroid_collision(vec![8, -8]),
            Vec::<i32>::new()
        );
        assert_eq!(ArraySolution::asteroid_collision(vec![10, 2, -5]), vec![10]);
        assert_eq!(
            ArraySolution::asteroid_collision(vec![-2, -1, 1, 2]),
            vec![]
        );
    }

    #[test]
    fn test_two_sum() {
        assert_eq!(ArraySolution::two_sum(vec![2, 7, 11, 15], 9), vec![1, 2]);
        assert_eq!(ArraySolution::two_sum(vec![2, 3, 4], 6), vec![1, 3]);
        assert_eq!(ArraySolution::two_sum(vec![-1, 0], -1), vec![1, 2]);
    }

    #[test]
    fn test_max_product() {
        assert_eq!(ArraySolution::max_product(vec![2, 3, -2, 4]), 6);
        assert_eq!(ArraySolution::max_product(vec![-2, 0, -1]), 0);
        assert_eq!(ArraySolution::max_product(vec![-4, -3, -2]), 12)
    }

    #[test]
    pub fn test_single_number() {
        assert_eq!(ArraySolution::single_number(vec![2, 2, 1]), 1);
        assert_eq!(ArraySolution::single_number(vec![4, 1, 2, 1, 2]), 4);
    }

    #[test]
    pub fn test_single_number_ii() {
        assert_eq!(ArraySolution::single_number_ii(vec![2, 2, 3, 2]), 3);
        assert_eq!(
            ArraySolution::single_number_ii(vec![0, 1, 0, 1, 0, 1, 99]),
            99
        );
    }

    #[test]
    fn test_missing_number() {
        assert_eq!(ArraySolution::missing_number(vec![3, 0, 1]), 2);
        assert_eq!(ArraySolution::missing_number(vec![0, 1]), 2);
        assert_eq!(
            ArraySolution::missing_number(vec![9, 6, 4, 2, 3, 5, 7, 0, 1]),
            8
        );
        assert_eq!(ArraySolution::missing_number(vec![0]), 1);
    }

    #[test]
    fn test_trap_rain() {
        assert_eq!(
            ArraySolution::trap_rain(vec![0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2]),
            6
        );
        assert_eq!(ArraySolution::trap_rain(vec![4, 2, 0, 3, 2, 5]), 9);
    }

    #[test]
    fn test_max_sum_hourglass() {
        assert_eq!(
            ArraySolution::max_sum(vec![
                vec![-9, -9, -9, 1, 1, 1],
                vec![0, -9, 0, 4, 3, 2],
                vec![-9, -9, -9, 1, 2, 3],
                vec![0, 0, 8, 6, 6, 0],
                vec![0, 0, 0, -2, 0, 0],
                vec![0, 0, 1, 2, 4, 0],
            ]),
            28
        );
        assert_eq!(
            ArraySolution::max_sum(vec![
                vec![1, 1, 1, 0, 0, 0],
                vec![0, 1, 0, 0, 0, 0],
                vec![1, 1, 1, 0, 0, 0],
                vec![0, 0, 2, 4, 4, 0],
                vec![0, 0, 0, 2, 0, 0],
                vec![0, 0, 1, 2, 4, 0],
            ]),
            19
        );
    }

    #[test]
    fn test_max_profits() {
        assert_eq!(ArraySolution::max_profits(vec![7, 1, 5, 3, 6, 4]), 5);
        assert_eq!(ArraySolution::max_profits(vec![7, 6, 4, 3, 1]), 0);
    }

    #[test]
    fn test_max_profit_ii() {
        assert_eq!(ArraySolution::max_profit_ii(vec![7, 1, 5, 3, 6, 4]), 7);
        assert_eq!(ArraySolution::max_profit_ii(vec![1, 2, 3, 4, 5]), 4);
        assert_eq!(ArraySolution::max_profit_ii(vec![7, 6, 4, 3, 1]), 0);
    }

    #[test]
    fn test_max_profit_iii() {
        assert_eq!(
            ArraySolution::max_profit_iii(vec![3, 3, 5, 0, 0, 3, 1, 4]),
            6
        );
        assert_eq!(ArraySolution::max_profit_iii(vec![1, 2, 3, 4, 5]), 4);
        assert_eq!(ArraySolution::max_profit_iii(vec![7, 6, 4, 3, 1]), 0);
    }

    #[test]
    fn test_max_profit_with_cool_down() {
        assert_eq!(
            ArraySolution::max_profit_with_cool_down(vec![1, 2, 3, 0, 2]),
            3
        );
    }

    #[test]
    fn test_max_profit_iv() {
        assert_eq!(ArraySolution::max_profit_iv(2, vec![2, 4, 1]), 2);
        assert_eq!(ArraySolution::max_profit_iv(2, vec![3, 2, 6, 5, 0, 3]), 7);
    }

    #[test]
    fn test_stone_game_ii() {
        assert_eq!(ArraySolution::stone_game_vii(vec![5, 3, 1, 4, 2]), 6);
        assert_eq!(
            ArraySolution::stone_game_vii(vec![7, 90, 5, 1, 100, 10, 10, 2]),
            122
        );
    }

    #[test]
    fn test_sort_colors() {
        let mut input = vec![2, 0, 2, 1, 1, 0];
        ArraySolution::sort_colors(&mut input);
        assert_eq!(input, vec![0, 0, 1, 1, 2, 2]);
        let mut input = vec![2, 0, 1];
        ArraySolution::sort_colors(&mut input);
        assert_eq!(input, vec![0, 1, 2]);
        let mut input = vec![0];
        ArraySolution::sort_colors(&mut input);
        assert_eq!(input, vec![0]);
        let mut input = vec![1];
        ArraySolution::sort_colors(&mut input);
        assert_eq!(input, vec![1]);
        let mut input = vec![2];
        ArraySolution::sort_colors(&mut input);
        assert_eq!(input, vec![2]);
    }

    #[test]
    fn test_is_valid_sudoki() {
        assert_eq!(
            ArraySolution::is_valid_sudoku(vec![
                vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
                vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
                vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
                vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
                vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
                vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
                vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
                vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
                vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
            ]),
            true
        );
        assert_eq!(
            ArraySolution::is_valid_sudoku(vec![
                vec!['8', '3', '.', '.', '7', '.', '.', '.', '.'],
                vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
                vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
                vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
                vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
                vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
                vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
                vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
                vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
            ]),
            false
        );
    }

    #[test]
    fn test_set_zero() {
        let mut input = vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]];
        ArraySolution::set_zero(&mut input);
        assert_eq!(input, vec![vec![1, 0, 1], vec![0, 0, 0], vec![1, 0, 1]]);
        let mut input = vec![vec![0, 1, 2, 0], vec![3, 4, 5, 2], vec![1, 3, 1, 5]];
        ArraySolution::set_zero(&mut input);
        assert_eq!(
            input,
            vec![vec![0, 0, 0, 0], vec![0, 4, 5, 0], vec![0, 3, 1, 0]]
        );
    }

    #[test]
    fn test_missing_positive() {
        assert_eq!(ArraySolution::first_missing_positive(vec![1, 2, 0]), 3);
        assert_eq!(ArraySolution::first_missing_positive(vec![3, 4, -1, 1]), 2);
        assert_eq!(
            ArraySolution::first_missing_positive(vec![7, 8, 9, 11, 12]),
            1
        );
        assert_eq!(ArraySolution::first_missing_positive(vec![1]), 2);
    }

    #[test]
    fn test_number_of_good_pairs() {
        assert_eq!(
            ArraySolution::number_of_good_pairs(vec![1, 2, 3, 1, 1, 3]),
            4
        );
        assert_eq!(ArraySolution::number_of_good_pairs(vec![1, 1, 1, 1]), 6);
        assert_eq!(ArraySolution::number_of_good_pairs(vec![1, 2, 3]), 0);
    }

    #[test]
    fn test_remove_duplicates() {
        let mut input = vec![1, 1, 1, 2, 2, 3];
        assert_eq!(ArraySolution::remove_duplicates(&mut input), 5);
        assert_eq!(input[0..5], vec![1, 1, 2, 2, 3]);
        let mut input = vec![0, 0, 1, 1, 1, 1, 2, 3, 3];
        assert_eq!(ArraySolution::remove_duplicates(&mut input), 7);
        assert_eq!(input[0..7], vec![0, 0, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_average_value() {
        assert_eq!(ArraySolution::average_value(vec![1, 3, 6, 10, 12, 15]), 9);
        assert_eq!(ArraySolution::average_value(vec![1, 2, 4, 7, 10]), 0);
    }

    #[test]
    fn test_count_donut() {
        assert_eq!(
            ArraySolution::count_donut(vec![
                "####.".to_string().chars().collect::<Vec<char>>(),
                "#.###".to_string().chars().collect::<Vec<char>>(),
                "###.#".to_string().chars().collect::<Vec<char>>(),
                "#.###".to_string().chars().collect::<Vec<char>>(),
                "###.#".to_string().chars().collect::<Vec<char>>(),
            ]),
            3
        );
    }

    #[test]
    fn test_find_matrix() {
        let mut res = ArraySolution::find_matrix(vec![1, 3, 4, 1, 2, 3, 1]);
        for row in res.iter_mut() {
            row.sort();
        }
        res.sort();
        assert_eq!(res, vec![vec![1], vec![1, 2, 3, 4], vec![1, 3]]);
    }

    #[test]
    fn test_surround_regions() {
        let mut input = vec![
            vec!['X', 'X', 'X', 'X'],
            vec!['X', 'O', 'O', 'X'],
            vec!['X', 'X', 'O', 'X'],
            vec!['X', 'O', 'X', 'X'],
        ];
        ArraySolution::surround_regions(&mut input);
        assert_eq!(
            input,
            vec![
                vec!['X', 'X', 'X', 'X'],
                vec!['X', 'X', 'X', 'X'],
                vec!['X', 'X', 'X', 'X'],
                vec!['X', 'O', 'X', 'X'],
            ]
        );
        let mut input = vec![
            vec!['X', 'O', 'X', 'O', 'X', 'O'],
            vec!['O', 'X', 'O', 'X', 'O', 'X'],
            vec!['X', 'O', 'X', 'O', 'X', 'O'],
            vec!['O', 'X', 'O', 'X', 'O', 'X'],
        ];
        ArraySolution::surround_regions(&mut input);
        assert_eq!(
            input,
            vec![
                vec!['X', 'O', 'X', 'O', 'X', 'O'],
                vec!['O', 'X', 'X', 'X', 'X', 'X'],
                vec!['X', 'X', 'X', 'X', 'X', 'O'],
                vec!['O', 'X', 'O', 'X', 'O', 'X'],
            ]
        );
        let mut input = vec![
            vec!['O', 'O', 'O', 'O', 'X', 'X'],
            vec!['O', 'O', 'O', 'O', 'O', 'O'],
            vec!['O', 'X', 'O', 'X', 'O', 'O'],
            vec!['O', 'X', 'O', 'O', 'X', 'O'],
            vec!['O', 'X', 'O', 'X', 'O', 'O'],
            vec!['O', 'X', 'O', 'O', 'O', 'O'],
        ];
        ArraySolution::surround_regions(&mut input);
        assert_eq!(
            input,
            vec![
                vec!['O', 'O', 'O', 'O', 'X', 'X'],
                vec!['O', 'O', 'O', 'O', 'O', 'O'],
                vec!['O', 'X', 'O', 'X', 'O', 'O'],
                vec!['O', 'X', 'O', 'O', 'X', 'O'],
                vec!['O', 'X', 'O', 'X', 'O', 'O'],
                vec!['O', 'X', 'O', 'O', 'O', 'O'],
            ]
        );
    }

    #[test]
    fn test_can_complete_circuit() {
        assert_eq!(
            ArraySolution::can_complete_circuit(vec![1, 2, 3, 4, 5], vec![3, 4, 5, 1, 2]),
            3
        );
        assert_eq!(
            ArraySolution::can_complete_circuit(vec![2, 3, 4], vec![3, 4, 3]),
            -1
        );
        assert_eq!(
            ArraySolution::can_complete_circuit(vec![5, 1, 2, 3, 4], vec![4, 4, 1, 5, 1]),
            4
        );
    }

    #[test]
    fn test_find_duplicate() {
        assert_eq!(ArraySolution::find_duplicate(vec![1, 3, 4, 2, 2]), 2);
        assert_eq!(ArraySolution::find_duplicate(vec![3, 1, 3, 4, 2]), 3);
        assert_eq!(ArraySolution::find_duplicate(vec![1, 1]), 1);
        assert_eq!(ArraySolution::find_duplicate(vec![1, 1, 2]), 1);
    }

    #[test]
    fn test_largest_number() {
        assert_eq!(ArraySolution::largest_number(vec![10, 2]), "210");
        assert_eq!(
            ArraySolution::largest_number(vec![3, 30, 34, 5, 9]),
            "9534330"
        );
        assert_eq!(ArraySolution::largest_number(vec![1]), "1");
        assert_eq!(ArraySolution::largest_number(vec![10]), "10");
        assert_eq!(ArraySolution::largest_number(vec![0, 0]), "0");
    }

    #[test]
    fn test_divide_array() {
        assert_eq!(
            ArraySolution::divide_array(vec![1, 3, 4, 8, 7, 9, 3, 5, 1], 23),
            vec![vec![1, 1, 3], vec![3, 4, 5], vec![7, 8, 9]]
        )
    }

    #[test]
    fn test_group_the_people() {
        let mut res = ArraySolution::group_the_people(vec![3, 3, 3, 3, 3, 1, 3]);
        for row in res.iter_mut() {
            row.sort();
        }
        res.sort();
        assert_eq!(res, vec![vec![0, 1, 2], vec![3, 4, 6], vec![5]]);
        let mut res = ArraySolution::group_the_people(vec![2, 1, 3, 3, 3, 2]);
        for row in res.iter_mut() {
            row.sort();
        }
        res.sort();
        assert_eq!(res, vec![vec![0, 5], vec![1], vec![2, 3, 4]]);
    }

    #[test]
    fn test_candy() {
        assert_eq!(ArraySolution::candy(vec![1, 0, 2]), 5);
        assert_eq!(ArraySolution::candy(vec![1, 2, 2]), 4);
    }

    #[test]
    fn test_zigzag_convert() {
        assert_eq!(
            ArraySolution::zigzag_convert("PAYPALISHIRING".to_string(), 3),
            "PAHNAPLSIIGYIR"
        );
        assert_eq!(
            ArraySolution::zigzag_convert("PAYPALISHIRING".to_string(), 4),
            "PINALSIGYAHRPI"
        );
        assert_eq!(ArraySolution::zigzag_convert("A".to_string(), 1), "A");
    }

    #[test]
    fn test_rearrange_number() {
        assert_eq!(
            ArraySolution::rearrange_array(vec![3, 1, -2, -5, 2, -4]),
            vec![3, -2, 1, -5, 2, -4]
        );
    }

    #[test]
    fn test_generate_matrix() {
        assert_eq!(
            ArraySolution::generate_matrix(3),
            vec![vec![1, 2, 3], vec![8, 9, 4], vec![7, 6, 5]]
        );
        assert_eq!(ArraySolution::generate_matrix(1), vec![vec![1]]);
    }

    #[test]
    fn test_largest_perimeter() {
        assert_eq!(ArraySolution::largest_perimeter(vec![2, 1, 2]), 5);
        assert_eq!(ArraySolution::largest_perimeter(vec![5, 5, 5]), 15);
        assert_eq!(
            ArraySolution::largest_perimeter(vec![1, 12, 1, 2, 5, 50, 3]),
            12
        );

        assert_eq!(ArraySolution::largest_perimeter(vec![5, 5, 50]), -1);
    }

    #[test]
    fn test_find_least_num_of_unique_ints() {
        assert_eq!(
            ArraySolution::find_least_num_of_unique_ints(vec![4, 3, 1, 1, 3, 3, 2], 3),
            2
        );
        assert_eq!(
            ArraySolution::find_least_num_of_unique_ints(vec![5, 5, 4], 1),
            1
        );
        assert_eq!(
            ArraySolution::find_least_num_of_unique_ints(vec![2, 1, 1, 3, 3, 3], 3),
            1
        );
        assert_eq!(ArraySolution::find_least_num_of_unique_ints(vec![1], 1), 0,)
    }

    #[test]
    fn test_furthest_building() {
        assert_eq!(
            ArraySolution::furthest_building(vec![4, 2, 7, 6, 9, 14, 12], 5, 1),
            4
        );
        assert_eq!(
            ArraySolution::furthest_building(vec![4, 12, 2, 7, 3, 18, 20, 3, 19], 10, 2),
            7
        );
        assert_eq!(
            ArraySolution::furthest_building(vec![14, 3, 19, 3], 17, 0),
            3
        );
    }

    #[test]
    fn test_most_booked_iii() {
        assert_eq!(
            ArraySolution::most_booked_iii(2, vec![(0, 10), (1, 5), (2, 7), (3, 4)]),
            0
        );
        assert_eq!(
            ArraySolution::most_booked_iii(3, vec![(1, 20), (2, 10), (3, 5), (4, 9), (6, 8)]),
            1
        );
    }

    #[test]
    fn test_find_judge() {
        assert_eq!(ArraySolution::find_judge(2, vec![vec![1, 2]]), 2);
        assert_eq!(
            ArraySolution::find_judge(3, vec![vec![1, 3], vec![2, 3]]),
            3
        );
        assert_eq!(
            ArraySolution::find_judge(3, vec![vec![1, 3], vec![2, 3], vec![3, 1]]),
            -1
        );
    }

    #[test]
    fn test_is_monotonic() {
        assert_eq!(ArraySolution::is_monotonic(vec![1, 2, 2, 3]), true);
        assert_eq!(ArraySolution::is_monotonic(vec![6, 5, 4, 4]), true);
        assert_eq!(ArraySolution::is_monotonic(vec![1, 3, 2]), false);
        assert_eq!(ArraySolution::is_monotonic(vec![1, 2, 4, 5]), true);
        assert_eq!(ArraySolution::is_monotonic(vec![1, 1, 1]), true);
    }

    #[test]
    fn test_sort_squares() {
        assert_eq!(
            ArraySolution::sorted_squares(vec![-4, -1, 0, 3, 10]),
            vec![0, 1, 9, 16, 100]
        );
        assert_eq!(
            ArraySolution::sorted_squares(vec![-7, -3, 2, 3, 11]),
            vec![4, 9, 9, 49, 121]
        );
    }

    #[test]
    fn test_bag_of_tokens_score() {
        assert_eq!(ArraySolution::bag_of_tokens_score(vec![100], 50), 0);
        assert_eq!(ArraySolution::bag_of_tokens_score(vec![100, 200], 150), 1);
        assert_eq!(
            ArraySolution::bag_of_tokens_score(vec![100, 200, 300, 400], 200),
            2
        );
    }

    #[test]
    fn test_min_operations() {
        assert_eq!(ArraySolution::min_operations(vec![4, 2, 5, 3]), 0);
        assert_eq!(ArraySolution::min_operations(vec![1, 2, 3, 5, 6]), 1);
        assert_eq!(ArraySolution::min_operations(vec![1, 10, 100, 1000]), 3);
    }

    #[test]
    fn test_max_frequency_elements() {
        assert_eq!(
            ArraySolution::max_frequency_elements(vec![1, 2, 2, 3, 1, 4]),
            4
        );
        assert_eq!(
            ArraySolution::max_frequency_elements(vec![1, 2, 3, 4, 5]),
            5
        );
    }

    #[test]
    fn test_array_intersections() {
        let mut res = ArraySolution::intersection(vec![1, 2, 2, 1], vec![2, 2]);
        res.sort();
        assert_eq!(res, vec![2]);
        let mut res = ArraySolution::intersection(vec![4, 9, 5], vec![9, 4, 9, 8, 4]);
        res.sort();
        assert_eq!(res, vec![4, 9]);
    }

    #[test]
    fn test_num_subarray_with_sum() {
        assert_eq!(
            ArraySolution::num_subarray_with_sum(vec![1, 0, 1, 0, 1], 2),
            4
        );
        assert_eq!(ArraySolution::num_subarray_with_sum(vec![1, 1, 1], 2), 2);
    }

    #[test]
    fn test_find_max_length() {
        assert_eq!(ArraySolution::find_max_length(vec![0, 1]), 2);
        assert_eq!(ArraySolution::find_max_length(vec![0, 1, 0]), 2);
    }

    #[test]
    fn test_find_min_arrow_shots() {
        assert_eq!(
            ArraySolution::find_min_arrow_shots(vec![
                vec![10, 16],
                vec![2, 8],
                vec![1, 6],
                vec![7, 12]
            ]),
            2
        );
        assert_eq!(
            ArraySolution::find_min_arrow_shots(vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
                vec![7, 8]
            ]),
            4
        );
        assert_eq!(
            ArraySolution::find_min_arrow_shots(vec![
                vec![1, 2],
                vec![2, 3],
                vec![3, 4],
                vec![4, 5]
            ]),
            2
        );
    }

    #[test]
    fn test_least_interval() {
        assert_eq!(
            ArraySolution::least_interval(vec!['A', 'A', 'A', 'B', 'B', 'B'], 2),
            8
        );
        assert_eq!(
            ArraySolution::least_interval(vec!['A', 'A', 'A', 'B', 'B', 'B'], 0),
            6
        );
        assert_eq!(
            ArraySolution::least_interval(
                vec!['A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
                2
            ),
            16
        );
    }

    #[test]
    fn test_find_duplicates() {
        assert_eq!(
            ArraySolution::find_duplicates(vec![4, 3, 2, 7, 8, 2, 3, 1]),
            vec![2, 3]
        );
        assert_eq!(ArraySolution::find_duplicates(vec![1, 1, 2]), vec![1]);
        assert_eq!(ArraySolution::find_duplicates(vec![1]), Vec::<i32>::new());
    }

    #[test]
    fn test_number_of_boomerangs() {
        assert_eq!(
            ArraySolution::number_of_boomerangs(vec![vec![0, 0], vec![1, 0], vec![2, 0]]),
            2
        );
        assert_eq!(
            ArraySolution::number_of_boomerangs(vec![vec![1, 1], vec![2, 2], vec![3, 3]]),
            2
        );
        assert_eq!(ArraySolution::number_of_boomerangs(vec![vec![1, 1]]), 0);
    }

    #[test]
    fn test_matrix_score() {
        assert_eq!(
            ArraySolution::matrix_score(vec![vec![0, 0, 1, 1], vec![1, 0, 1, 0], vec![1, 1, 0, 0]]),
            39
        );
        assert_eq!(ArraySolution::matrix_score(vec![vec![0]]), 1);
    }
}
