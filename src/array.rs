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
}

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
    let mut expected = vec![vec![1, 2, 3], vec![1, 3, 2], vec![2, 1, 3], vec![2, 3, 1], vec![3, 1, 2], vec![3, 2, 1]];
    expected.sort();
    assert_eq!(output, expected);
    let input = vec![0,1];
    let mut output = ArrarySolution::permute(input);
    output.sort();
    let mut expected = vec![vec![0, 1], vec![1, 0]];
    expected.sort();

    assert_eq!(output, expected);
}