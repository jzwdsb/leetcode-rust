use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub struct GraphSolution {}

impl GraphSolution {
    /*
    link: https://leetcode.com/problems/course-schedule/
    check whether it is possible to finish all courses.
    travel all the nodes of the graph using DFS and check if there is a cycle or unvisited node.
     */

    pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let mut graph = vec![vec![]; num_courses as usize];
        let mut visited = vec![0; num_courses as usize];

        for edge in prerequisites {
            graph[edge[0] as usize].push(edge[1] as usize);
        }

        for i in 0..num_courses {
            if !Self::dfs(&graph, &mut visited, i as usize) {
                return false;
            }
        }
        true
    }

    fn dfs(graph: &Vec<Vec<usize>>, visited: &mut Vec<i32>, i: usize) -> bool {
        if visited[i] == 1 {
            return false;
        }
        if visited[i] == -1 {
            return true;
        }
        visited[i] = 1;
        for &j in &graph[i] {
            if !Self::dfs(graph, visited, j) {
                return false;
            }
        }
        visited[i] = -1;
        true
    }

    /*
    link: https://leetcode.com/problems/maximal-network-rank/
    find the maximal network rank of the given cities.
    network rank of two different cities is defined as the total number of directly connected roads to either city.
    the maximal network rank is the maximum network rank of all pairs of different cities.

    1. create a graph using adjacency list
    2. find the network rank of each pair of cities
    3. return the maximum network rank
     */

    pub fn maximal_network_rank(n: i32, roads: Vec<Vec<i32>>) -> i32 {
        // adjacency list
        let mut graph = vec![vec![]; n as usize];
        let mut max = 0;
        for road in roads {
            graph[road[0] as usize].push(road[1] as usize);
            graph[road[1] as usize].push(road[0] as usize);
        }

        // find the network rank of each pair of cities
        // update the maximum network rank
        for i in 0..n {
            for j in i + 1..n {
                // find the network rank of i and j
                let mut count = graph[i as usize].len() + graph[j as usize].len();
                if graph[i as usize].contains(&(j as usize)) {
                    // if i and j are directly connected, the the road is counted twice
                    // so we need to subtract 1
                    count -= 1;
                }
                max = max.max(count);
            }
        }
        max as i32
    }

    /*
    link: https://leetcode.com/problems/path-with-minimum-effort
    plain dfs with visited array will cause TLE

    we can use dijkstra with heap to solve this problem
    use heap to store the cost and position

     */

    pub fn minimum_effort_path(heights: Vec<Vec<i32>>) -> i32 {
        let direction = [(0, 1), (0, -1i32), (1, 0), (-1i32, 0)];
        let mut visited = vec![vec![false; heights[0].len()]; heights.len()];
        // binary heap will sort the elements in the heap by the first element of the tuple
        // so we need to reverse the cost to make the heap sort the cost in ascending order
        let mut heap = BinaryHeap::<(Reverse<i32>, usize, usize)>::new();
        heap.push((Reverse(0), 0, 0));
        loop {
            let (Reverse(cost), x, y) = heap.pop().unwrap();
            if x == heights.len() - 1 && y == heights[0].len() - 1 {
                // reach the destination, return the cost
                break cost;
            }
            if visited[x][y] {
                continue;
            }
            visited[x][y] = true;
            for (dx, dy) in direction {
                let nx = x.wrapping_add_signed(dx as isize);
                let ny = y.wrapping_add_signed(dy as isize);
                if nx < heights.len() && ny < heights[0].len() && visited[nx][ny] == false {
                    // the maximum cost of the path from (x, y) to (nx, ny) is the maximum of the current cost
                    // and the difference between the heights of (nx, ny) and (x, y)
                    let cost2 = cost.max((heights[nx][ny] - heights[x][y]).abs());
                    heap.push((Reverse(cost2), nx, ny));
                }
            }
        }
    }
    /*
    https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

    implemetation of dijkstra algorithm
     */
    pub fn dijkstra() -> i32 {
        todo!("dijkstra")
    }

    // a[i]: the start of road i
    // b[i]: the end of road i
    // n: citys
    pub fn max_network_rank(a: Vec<i32>, b: Vec<i32>, _n: i32) -> i32 {
        let mut max = 0;
        let mut count = HashMap::<i32, i32>::new();
        for i in 0..a.len() {
            count.entry(a[i]).and_modify(|v| *v += 1).or_insert(1);
            count.entry(b[i]).and_modify(|v| *v += 1).or_insert(1);
        }
        for i in 0..a.len() {
            max = max.max(count.get(&a[i]).unwrap() + count.get(&b[i]).unwrap() - 1)
        }

        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_finish() {
        assert_eq!(GraphSolution::can_finish(2, vec![vec![1, 0]]), true);
        assert_eq!(
            GraphSolution::can_finish(2, vec![vec![1, 0], vec![0, 1]]),
            false
        );
    }

    #[test]
    fn test_maximal_network_rank() {
        assert_eq!(
            GraphSolution::maximal_network_rank(
                4,
                vec![vec![0, 1], vec![0, 3], vec![1, 2], vec![1, 3]]
            ),
            4
        );
        assert_eq!(
            GraphSolution::maximal_network_rank(
                5,
                vec![
                    vec![0, 1],
                    vec![0, 3],
                    vec![1, 2],
                    vec![1, 3],
                    vec![2, 3],
                    vec![2, 4]
                ],
            ),
            5
        );
    }

    #[test]
    fn test_minimum_effort_path() {
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![vec![1, 2, 2], vec![3, 8, 2], vec![5, 3, 5]]),
            2
        );
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![vec![1, 2, 3], vec![3, 8, 4], vec![5, 3, 5]]),
            1
        );
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![
                vec![1, 2, 1, 1, 1],
                vec![1, 2, 1, 2, 1],
                vec![1, 2, 1, 2, 1],
                vec![1, 2, 1, 2, 1],
                vec![1, 1, 1, 2, 1]
            ]),
            0
        );
        assert_eq!(
            GraphSolution::minimum_effort_path(vec![vec![1, 10, 6, 7, 9, 10, 4, 9]]),
            9
        );
    }

    #[test]
    fn test_max_network_rank() {
        todo!("add test case")
    }
}
