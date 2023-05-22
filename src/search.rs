pub struct SearchSolution {}

impl SearchSolution {
    /*
    link: https://leetcode.com/problems/search-in-rotated-sorted-array/
    in the description, it requires us to solve this problem in O(logn) time complexity
    apparently, we can use binary search to solve this problem
     */
    pub fn search_in_rotated_sorted_array(nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        while left <= right {
            let mid = (right+left)/2;
            if nums[mid] == target {
                return mid as i32;
            }
            // if nums[left] <= nums[mid], it means the left part is sorted
            if nums[left] <= nums[mid] {
                if nums[left] <= target && target < nums[mid] {
                    right = mid - 1;
                }
                else {
                    left = mid + 1;
                }
            }
            // if nums[mid] <= nums[right], it means the right part is sorted
            else {
                if nums[mid] < target && target <= nums[right] {
                    left = mid + 1;
                }
                else {
                    right = mid - 1;
                }
            }
        }
        -1
    }
}

#[test]
fn test_search_in_rotated_sorted_arrary() {
    assert_eq!(SearchSolution::search_in_rotated_sorted_array(vec![4, 5, 6, 7, 0, 1, 2], 0), 4);
    assert_eq!(SearchSolution::search_in_rotated_sorted_array(vec![4, 5, 6, 7, 0, 1, 2], 3), -1);
    assert_eq!(SearchSolution::search_in_rotated_sorted_array(vec![1], 0), -1);
    assert_eq!(SearchSolution::search_in_rotated_sorted_array(vec![1, 3], 3), 1);
}