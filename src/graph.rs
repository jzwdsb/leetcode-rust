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
            GraphSolution::maximal_network_rank(4, vec![vec![0, 1], vec![0, 3], vec![1, 2], vec![1, 3]]),
            4
        );
        assert_eq!(
            GraphSolution::maximal_network_rank(
                5,
                vec![vec![0, 1], vec![0, 3], vec![1, 2], vec![1, 3], vec![2, 3], vec![2, 4]]
            ),
            5
        );
    }
}
