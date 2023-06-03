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
