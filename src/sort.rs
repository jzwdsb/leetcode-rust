#![allow(dead_code)]

pub struct SortSolution {}

impl SortSolution {
    pub fn top_k_fequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut map = std::collections::HashMap::new();
        for num in nums {
            *map.entry(num).or_insert(0) += 1;
        }
        let mut vec: Vec<(i32, i32)> = map.into_iter().collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1));
        vec.into_iter()
            .take(k as usize)
            .map(|(num, _)| num)
            .collect()
    }

    pub fn merge_sorted_array(nums1: &mut Vec<i32>, nums2: &mut Vec<i32>) {
        let mut i = 0;
        let mut j = 0;
        let mut len1 = nums1.len() - nums2.len();
        let mut len2 = nums2.len();
        loop {
            if i >= len1 {
                nums1[i..].clone_from_slice(&nums2[j..]);
                break;
            }
            if j >= nums2.len() {
                break;
            }
            if nums1[i] > nums2[j] {
                nums1.insert(i, nums2[j]);
                nums1.truncate(len1 + len2);
                len1 += 1;
                len2 -= 1;
                j += 1;
            }
            i += 1;
        }
    }

    pub fn heap_sort(data: &mut Vec<i32>) {
        let len = data.len();
        for i in (0..len / 2).rev() {
            Self::heapify(data, len, i);
        }
        for i in (1..len).rev() {
            data.swap(0, i);
            Self::heapify(data, i, 0);
        }
    }

    // heapify the input data to a max heap
    // definition: heapify the subtree rooted at index i
    // so that the subtree is a max heap
    fn heapify(data: &mut Vec<i32>, len: usize, i: usize) {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < len && data[left] > data[largest] {
            largest = left;
        }
        if right < len && data[right] > data[largest] {
            largest = right;
        }
        if largest != i {
            data.swap(i, largest);
            Self::heapify(data, len, largest);
        }
    }

    pub fn quick_sort(nums: &mut Vec<i32>) {
        Self::quick_helper(nums, 0, nums.len().wrapping_sub(1))
    }

    fn quick_helper(nums: &mut Vec<i32>, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let pivot = Self::quick_partition(nums, start, end);
        Self::quick_helper(nums, start, if pivot == 0 { 0 } else { pivot - 1 });
        Self::quick_helper(nums, pivot + 1, end)
    }

    fn quick_partition(nums: &mut Vec<i32>, start: usize, end: usize) -> usize {
        let piviot = nums[end];
        let mut i = start;
        for j in start..end {
            if nums[j] < piviot {
                nums.swap(i, j);
                i += 1;
            }
        }
        nums.swap(i, end);
        i
    }

    pub fn merge_sort(nums1: &mut Vec<i32>) {
        let len = nums1.len();
        Self::merge_helper(nums1, 0, len - 1);
    }

    fn merge_helper(nums1: &mut Vec<i32>, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let mid = (start + end) / 2;
        Self::merge_helper(nums1, start, mid);
        Self::merge_helper(nums1, mid + 1, end);
        Self::merge(nums1, start, mid, end);
    }

    fn merge(nums1: &mut Vec<i32>, start: usize, mid: usize, end: usize) {
        let mut i = start;
        let mut j = mid + 1;
        let mut tmp = Vec::new();
        while i <= mid && j <= end {
            if nums1[i] < nums1[j] {
                tmp.push(nums1[i]);
                i += 1;
            } else {
                tmp.push(nums1[j]);
                j += 1;
            }
        }
        while i <= mid {
            tmp.push(nums1[i]);
            i += 1;
        }
        while j <= end {
            tmp.push(nums1[j]);
            j += 1;
        }
        nums1[start..=end].clone_from_slice(&tmp);
    }

    pub fn bubble_sort(nums1: &mut Vec<i32>) {
        let len = nums1.len();
        for i in 0..len {
            for j in 0..len - i - 1 {
                if nums1[j] > nums1[j + 1] {
                    nums1.swap(j, j + 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_top_k_fequent() {
        assert_eq!(
            SortSolution::top_k_fequent(vec![1, 1, 1, 2, 2, 3], 2),
            vec![1, 2]
        );
    }

    #[test]
    fn test_merge_sorted_array() {
        let mut nums1 = vec![1, 2, 3, 0, 0, 0];
        let mut nums2 = vec![2, 5, 6];
        SortSolution::merge_sorted_array(&mut nums1, &mut nums2);
        assert_eq!(nums1, vec![1, 2, 2, 3, 5, 6]);
    }

    #[test]
    fn test_heap_sort() {
        let mut data = vec![4, 10, 3, 5, 1];
        SortSolution::heap_sort(&mut data);
        assert_eq!(data, vec![1, 3, 4, 5, 10]);
    }

    #[test]
    fn test_quick_sort() {
        let mut data = vec![4, 10, 3, 5, 1];
        SortSolution::quick_sort(&mut data);
        assert_eq!(data, vec![1, 3, 4, 5, 10]);
    }

    #[test]
    fn test_merge_sort() {
        let mut data = vec![4, 10, 3, 5, 1];
        SortSolution::merge_sort(&mut data);
        assert_eq!(data, vec![1, 3, 4, 5, 10]);
    }

    #[test]
    fn test_bubble_sort() {
        let mut data = vec![4, 10, 3, 5, 1];
        SortSolution::bubble_sort(&mut data);
        assert_eq!(data, vec![1, 3, 4, 5, 10]);
    }
}
