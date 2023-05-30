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

    /*
    link: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    we can solve this with binary search
    expand the first and last position of the target
    normally we can use the [)  to cover the edge case
     */
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut left = 0;
        let mut right = nums.len();
        while left < right {
            let mid = (left+right)/2;
            if nums[mid] == target {
                let mut start = mid;
                let mut end = mid;
                while start > 0 && nums[start-1] == target {
                    start -= 1;
                }
                while end < nums.len()-1 && nums[end+1] == target {
                    end += 1;
                }
                return vec![start as i32, end as i32];
            } else {
                if nums[mid] < target {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
        }

        vec![-1, -1]
    }
    pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len();
        while left < right {
            let mid = (left+right)/2;
            if nums[mid] == target {
                return mid as i32;
            } else {
                if nums[mid] < target {
                    left = mid+1;
                } else {
                    right = mid;
                }
            }
        }
        left as i32
    }
    pub fn binary_search(nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len();
        while left < right {
            let mid = (left+right)/2;
            if nums[mid] == target {
                return mid as i32;
            } else {
                if nums[mid] < target {
                    left = mid+1;
                } else {
                    right = mid;
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

#[test]
fn test_search_range() {
    assert_eq!(SearchSolution::search_range(vec![5,7,7,8,8,10], 8), vec![3, 4]);
    assert_eq!(SearchSolution::search_range(vec![5,7,7,8,8,10], 6), vec![-1, -1]);
    assert_eq!(SearchSolution::search_range(vec![], 0), vec![-1, -1]);
    assert_eq!(SearchSolution::search_range(vec![1], 1), vec![0, 0]);
    assert_eq!(SearchSolution::search_range(vec![1], 0), vec![-1, -1]); 
    assert_eq!(SearchSolution::search_range(vec![5,7,7,8,8,10], 6), vec![-1, -1]);
}

#[test]
fn test_search_insert() {
    assert_eq!(SearchSolution::search_insert(vec![1,3,5,6], 5), 2);
    assert_eq!(SearchSolution::search_insert(vec![1,3,5,6], 2), 1);
    assert_eq!(SearchSolution::search_insert(vec![1,3,5,6], 7), 4);
    assert_eq!(SearchSolution::search_insert(vec![1,3,5,6], 0), 0);
}