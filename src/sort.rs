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
}
