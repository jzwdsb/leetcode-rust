use std::ops::Div;

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

    #[allow(dead_code)]
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
    link: https://leetcode.com/problems/permutations/
    easily solved by the next_permutation solved before
     */
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn array_sign(nums: Vec<i32>) -> i32 {
        nums.iter().map(|x| x.signum()).product()
    }

    /*
    link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
     */
    #[allow(dead_code)]
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
    link: https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/
     */

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
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

    #[allow(dead_code)]
    pub fn rotate(image: &mut Vec<Vec<i32>>) {
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn spiral_matrix_ii(n: i32) -> Vec<Vec<i32>> {
        let mut matrix = vec![vec![0 as i32; n as usize]; n as usize];
        let mut length_y = n as usize;
        let mut length_x = n as usize;
        let total_num = n * n;
        let (mut x, mut y) = (0, 0); // pointing to the current pos
        let mut num = 1 as i32;
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn length_of_last_word(s: String) -> i32 {
        s.split_ascii_whitespace().last().unwrap_or("").len() as i32
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    same solution at remove elelement
     */

    #[allow(dead_code)]
    pub fn move_zeros(nums: &mut Vec<i32>) {
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
        use std::collections::HashSet;
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    when it reach to the end, we can get the result
     */

    #[allow(dead_code)]
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
                                std::cmp::Ordering::Less => {
                                    stack.pop();
                                }
                                std::cmp::Ordering::Equal => {
                                    stack.pop();
                                    break false;
                                }
                                std::cmp::Ordering::Greater => {
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
                std::cmp::Ordering::Less => back -= 1,
                std::cmp::Ordering::Equal => break,
                std::cmp::Ordering::Greater => front += 1,
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

    pub fn sinlge_number(nums: Vec<i32>) -> i32 {
        let mut res = 0;
        for num in nums {
            res ^= num;
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
        let mut res = 0;  // the result, res += min(left_max, right_max) - height[i]
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
}

pub fn main() {}

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
    fn test_max_profits() {
        assert_eq!(ArraySolution::max_profits(vec![7, 1, 5, 3, 6, 4]), 5);
        assert_eq!(ArraySolution::max_profits(vec![7, 6, 4, 3, 1]), 0);
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
                vec![16, 7, 10, 11]
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
                vec![9, 10, 11, 12]
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
            ArraySolution::insert_interval(vec![vec![1, 3], vec![6, 9]], vec![2, 5],),
            vec![vec![1, 5], vec![6, 9]]
        );
        assert_eq!(
            ArraySolution::insert_interval(
                vec![
                    vec![1, 2],
                    vec![3, 5],
                    vec![6, 7],
                    vec![8, 10],
                    vec![12, 16]
                ],
                vec![4, 8],
            ),
            vec![vec![1, 2], vec![3, 10], vec![12, 16]]
        );
        assert_eq!(
            ArraySolution::insert_interval(vec![vec![1, 5]], vec![2, 3],),
            vec![vec![1, 5]]
        );
    }

    #[test]
    fn test_sprial_matrix_ii() {
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
    fn test_earse_overlap_intervals() {
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![vec![1, 2], vec![2, 3], vec![3, 4],]),
            0
        );
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![vec![1, 2], vec![1, 2], vec![1, 2],]),
            2
        );
        assert_eq!(
            ArraySolution::erase_overlap_intervals(vec![vec![1, 2], vec![2, 3],]),
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
    fn test_missing_number() {
        assert_eq!(ArraySolution::missing_number(vec![3, 0, 1]), 2);
        assert_eq!(ArraySolution::missing_number(vec![0, 1]), 2);
        assert_eq!(ArraySolution::missing_number(vec![9, 6, 4, 2, 3, 5, 7, 0, 1]), 8);
        assert_eq!(ArraySolution::missing_number(vec![0]), 1);
    }

    #[test]
    fn test_trap_rain() {
        assert_eq!(ArraySolution::trap_rain(vec![0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2]), 6);
        assert_eq!(ArraySolution::trap_rain(vec![4, 2, 0, 3, 2, 5]), 9);
    }
}
