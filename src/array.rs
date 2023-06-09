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
        let mut matrix = vec![vec![0 as i32; n as usize];n as usize];
        let mut length_y = n as usize;
        let mut length_x = n as usize;
        let total_num = n*n;
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
    same solution at remove elelement
     */

    pub fn move_zeros(nums: &mut Vec<i32>) {
        let mut i = 0;
        let mut j = 0;
        while j < nums.len() {
            if nums[j] != 0 {
                nums[i] = nums[j];
                i += 1;
            }
            j+=1;
        }
        while i < nums.len() {
            nums[i] = 0;
            i += 1;
        }
        
    }

}

pub fn main() {}

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
