use std::ops::Div;

pub struct ArrarySolution {}

impl ArrarySolution {
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
            ArrarySolution::next_permutation(&mut nums_copy);
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
}

pub fn main() {}

#[test]
fn test_next_permutation() {
    let mut input = vec![1];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1]);
    let mut input = vec![1, 2, 3];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1, 3, 2]);
    let mut input = vec![3, 2, 1];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1, 2, 3]);
    let mut input = vec![1, 1, 5];
    ArrarySolution::next_permutation(&mut input);
    assert_eq!(input, vec![1, 5, 1]);
}

#[test]
fn test_permutation() {
    let input = vec![1, 2, 3];
    let mut output = ArrarySolution::permute(input);
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
    let mut output = ArrarySolution::permute(input);
    output.sort();
    let mut expected = vec![vec![0, 1], vec![1, 0]];
    expected.sort();

    assert_eq!(output, expected);
}

#[test]
fn test_array_sign() {
    assert_eq!(ArrarySolution::array_sign(vec![-1, -2, -3, -4, 3, 2, 1]), 1);
    assert_eq!(ArrarySolution::array_sign(vec![1, 5, 0, 2, -3]), 0);
    assert_eq!(ArrarySolution::array_sign(vec![-1, 1, -1, 1, -1]), -1);
    assert_eq!(
        ArrarySolution::array_sign(vec![i32::MAX / 2, i32::MAX / 2, i32::MIN]),
        -1
    );
}

#[test]
fn test_max_profits() {
    assert_eq!(ArrarySolution::max_profits(vec![7, 1, 5, 3, 6, 4]), 5);
    assert_eq!(ArrarySolution::max_profits(vec![7, 6, 4, 3, 1]), 0);
}

#[test]
fn test_remove_element() {
    let mut input = vec![3, 2, 2, 3];
    assert_eq!(ArrarySolution::remove_element(&mut input, 3), 2);
    assert_eq!(input[0..2], vec![2, 2]);
    let mut input = vec![0, 1, 2, 2, 3, 0, 4, 2];
    assert_eq!(ArrarySolution::remove_element(&mut input, 2), 5);
    assert_eq!(input[0..5], vec![0, 1, 3, 0, 4]);
}

#[test]
fn test_rotate() {
    let mut input = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    ArrarySolution::rotate(&mut input);
    assert_eq!(input, vec![vec![7, 4, 1], vec![8, 5, 2], vec![9, 6, 3]]);
    let mut input = vec![
        vec![5, 1, 9, 11],
        vec![2, 4, 8, 10],
        vec![13, 3, 6, 7],
        vec![15, 14, 12, 16],
    ];
    ArrarySolution::rotate(&mut input);
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
