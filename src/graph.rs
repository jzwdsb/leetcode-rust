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
}
