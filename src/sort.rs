
pub struct SortSolution{}

impl SortSolution {
    pub fn top_k_fequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut map = std::collections::HashMap::new();
        for num in nums {
            *map.entry(num).or_insert(0) += 1;
        }
        let mut vec: Vec<(i32, i32)> = map.into_iter().collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1));
        vec.into_iter().take(k as usize).map(|(num, _)| num).collect()
    }
}

pub fn main() {

}

#[test]
fn test_top_k_fequent() { 
    assert_eq!(SortSolution::top_k_fequent(vec![1,1,1,2,2,3], 2), vec![1,2]);
}